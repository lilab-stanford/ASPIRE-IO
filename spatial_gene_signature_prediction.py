"""
ASPIRE-IO Integrated Inference Pipeline

Loads the complete model with:
- MUSK backbone for vision-language embeddings
- Organ-specific MLP heads for gene signature prediction
- Organ classifier for automatic routing (optional)

Compatible with checkpoints from:
- pretrain.py / finetune_mlp.py (MultiHeadModel)
- train_moe.py / MoE_training.py (OrganClassifier)
- Integrated checkpoint with all components
"""

import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

# MUSK dependencies
from musk import utils, modeling
from timm.models import create_model
from transformers import XLMRobertaTokenizer
from huggingface_hub import login

# ============================================================================
# Configuration
# ============================================================================

# Default organ mapping (can be overridden by checkpoint)
ORGAN_TO_IDX = {
    "Bladder": 0, "Bowel": 1, "Breast": 2, "Lung": 3,
    "Lymph node": 4, "Prostate": 5, "Skin": 6
}

# Modality modes
MODALITY_MODES = {
    "image": 1024,
    "text": 1024,
    "both": 2048,
    "multimodal": 2048,
}

# ============================================================================
# MUSK Model Loading
# ============================================================================

def load_pretrained_musk_model():
    """Load pre-trained MUSK model from HuggingFace Hub."""
    model = create_model("musk_large_patch16_384")
    utils.load_model_and_may_interpolate(
        "hf_hub:xiangjx/musk", model, "model|module", ""
    )
    return model

# ============================================================================
# Model Definitions
# ============================================================================

class OrganSpecificHead(nn.Module):
    """Organ-specific prediction head for immune score and gene expression."""
    
    def __init__(self, gene_dim, in_dim=2048, hidden_dim=1024, dropout=0.3):
        super().__init__()
        self.immune_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        self.gene_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, gene_dim)
        )

    def forward(self, fused):
        immune = self.immune_head(fused).squeeze(-1)
        gene = self.gene_head(fused)
        return immune, gene


class MultiHeadModel(nn.Module):
    """Multi-organ model with MUSK backbone and organ-specific heads."""
    
    def __init__(self, musk_model, organ_to_gene_dim, organ2idx, 
                 hidden_dim=1024, dropout=0.3, use_fuse_proj=False):
        super().__init__()
        self.musk = musk_model
        self.organ_heads = nn.ModuleDict()
        self.organ_to_gene_dim = organ_to_gene_dim
        self.organ2idx = organ2idx
        self.idx2organ = {v: k for k, v in organ2idx.items()}
        self.use_fuse_proj = use_fuse_proj
        
        # Optional fusion projection (3072 -> 2048 for ms_aug=True)
        if use_fuse_proj:
            self.fuse_proj = nn.Linear(3072, 2048)
        
        for organ, gene_dim in organ_to_gene_dim.items():
            head = OrganSpecificHead(gene_dim, hidden_dim=hidden_dim, dropout=dropout)
            self.organ_heads[organ] = head.to(dtype=torch.bfloat16)

    def extract_embeddings(self, patch, text_ids, paddings):
        """Extract MUSK CLS embeddings."""
        cur_device = patch.device
        text_ids = text_ids.to(cur_device, dtype=torch.long)
        paddings = paddings.to(cur_device, dtype=torch.long)
        
        # Vision branch
        vision_cls = self.musk(
            image=patch, with_head=False, out_norm=False,
            ms_aug=False, return_global=True
        )[0]
        
        # Text branch
        text_cls = self.musk(
            text_description=text_ids, padding_mask=paddings,
            with_head=False, out_norm=False, ms_aug=False, return_global=True
        )[1]
        
        return vision_cls, text_cls

    def forward(self, patch, text_ids, paddings, organs):
        """
        Forward pass with organ-specific routing.
        
        Args:
            organs: List of organ names or tensor of organ indices
        """
        cur_device = patch.device
        
        # Extract embeddings
        vision_cls, text_cls = self.extract_embeddings(patch, text_ids, paddings)
        
        # Fuse embeddings
        fused = torch.cat([vision_cls, text_cls], dim=-1)
        
        # Apply projection if needed (for checkpoints with fuse_proj)
        if self.use_fuse_proj and hasattr(self, 'fuse_proj'):
            fused = self.fuse_proj(fused)
        
        B = fused.size(0)
        
        # Convert organ indices to names if needed
        if isinstance(organs, torch.Tensor):
            organs_local = [self.idx2organ.get(int(x), list(self.organ_heads.keys())[0]) 
                          for x in organs.cpu().tolist()]
        elif isinstance(organs, list) and len(organs) > 0 and isinstance(organs[0], int):
            organs_local = [self.idx2organ.get(x, list(self.organ_heads.keys())[0]) for x in organs]
        else:
            organs_local = organs
        
        # Route to organ-specific heads
        out_dict = {}
        for org in set(organs_local):
            indices = [i for i, o in enumerate(organs_local) if o == org]
            if not indices:
                continue
            
            group_fused = fused[indices]
            
            if org in self.organ_heads:
                head = self.organ_heads[org]
                group_immune, group_gene = head(group_fused)
            else:
                # Fallback for unknown organs
                group_immune = torch.zeros(len(indices), device=cur_device, dtype=torch.bfloat16)
                group_gene = torch.zeros((len(indices), 100), device=cur_device, dtype=torch.bfloat16)
            
            for local_idx, original_idx in enumerate(indices):
                out_dict[original_idx] = (group_immune[local_idx:local_idx+1],
                                          group_gene[local_idx:local_idx+1])
        
        # Assemble outputs
        immune_list, gene_list = [], []
        max_gene_dim = max(self.organ_to_gene_dim.values()) if self.organ_to_gene_dim else 100
        
        for i in range(B):
            if i in out_dict:
                immune_list.append(out_dict[i][0])
                pred = out_dict[i][1]
                if pred.shape[1] < max_gene_dim:
                    pad_tensor = torch.zeros((1, max_gene_dim - pred.shape[1]), 
                                     device=pred.device, dtype=pred.dtype)
                    pred = torch.cat([pred, pad_tensor], dim=1)
                gene_list.append(pred)
            else:
                immune_list.append(torch.zeros(1, device=cur_device, dtype=torch.bfloat16))
                gene_list.append(torch.zeros((1, max_gene_dim), device=cur_device, dtype=torch.bfloat16))
        
        return torch.cat(immune_list, dim=0), torch.cat(gene_list, dim=0), vision_cls, text_cls


class OrganClassifier(nn.Module):
    """Simple organ classifier for routing (matches MoE_training.py)."""
    
    def __init__(self, input_dim=2048, num_organs=7, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_organs)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return self.fc2(x)


class IntegratedASPIREModel(nn.Module):
    """
    Complete ASPIRE-IO model with MultiHeadModel + OrganClassifier.
    
    Can run in two modes:
    1. With known organs: Routes directly to specified organ heads
    2. With classifier: Predicts organ first, then routes
    """
    
    def __init__(self, multihead, organ_classifier=None, organ2idx=None):
        super().__init__()
        self.multihead = multihead
        self.organ_classifier = organ_classifier
        self.organ2idx = organ2idx or ORGAN_TO_IDX
        self.idx2organ = {v: k for k, v in self.organ2idx.items()}
        
    def forward(self, patch, text_ids, paddings, organs=None, use_classifier=True):
        """
        Forward pass.
        
        Args:
            organs: Known organ names/indices. If None and use_classifier=True, 
                   predicts organs using classifier.
            use_classifier: Whether to use classifier for routing
        """
        # Get base predictions from multihead
        immune_pred, gene_pred, vision_cls, text_cls = self.multihead(
            patch, text_ids, paddings, 
            organs if organs is not None else ["Lung"] * patch.size(0)  # Default
        )
        
        organ_info = None
        
        # Use classifier for automatic routing if available
        if use_classifier and self.organ_classifier is not None:
            # Prepare embeddings for classifier
            embeddings = torch.cat([vision_cls, text_cls], dim=-1)
            
            # Classify organs (ensure same dtype as classifier)
            organ_logits = self.organ_classifier(embeddings)
            organ_pred_idx = torch.argmax(organ_logits, dim=-1)
            organ_pred_names = [self.idx2organ.get(idx.item(), "Unknown") 
                               for idx in organ_pred_idx]
            
            # Re-route predictions based on classifier output
            immune_pred, gene_pred, _, _ = self.multihead(
                patch, text_ids, paddings, organ_pred_names
            )
            
            organ_info = {
                "organ_logits": organ_logits,
                "organ_pred_idx": organ_pred_idx,
                "organ_pred_names": organ_pred_names,
            }
        
        return immune_pred, gene_pred, organ_info

# ============================================================================
# Dataset
# ============================================================================

class InferenceDataset(Dataset):
    """Dataset for inference."""
    
    def __init__(self, csv_file, patch_dir, text_dir, tokenizer, max_len=100):
        self.df = pd.read_csv(csv_file)
        self.patch_dir = patch_dir
        self.text_dir = text_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transforms.ToTensor()
        
        print(f"Inference Dataset: {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def _load_patch(self, patient_id, spot_id):
        """Load patch from PNG or H5."""
        # Try PNG
        png_path = os.path.join(self.patch_dir, f"{patient_id}_{spot_id}.png")
        if os.path.exists(png_path):
            img = Image.open(png_path).convert("RGB")
            img = img.resize((384, 384), Image.BICUBIC)
            return self.transform(img)
        
        # Try H5
        h5_path = os.path.join(self.patch_dir, f"{patient_id}.h5")
        if not os.path.exists(h5_path):
            return None
        
        with h5py.File(h5_path, "r") as f:
            barcodes = [b.decode() if isinstance(b, bytes) else str(b) 
                       for b in f["barcode"][:].flatten()]
            idx = next((i for i, b in enumerate(barcodes) if b == spot_id), None)
            if idx is None:
                return None
            patch = f["img"][idx]
        
        img = Image.fromarray(patch).convert("RGB")
        img = img.resize((384, 384), Image.BICUBIC)
        return self.transform(img)

    def _load_text(self, patient_id):
        text_path = os.path.join(self.text_dir, f"{patient_id}_text.txt")
        if not os.path.exists(text_path):
            return None, None
        
        with open(text_path, "r") as f:
            text = f.read().strip()
        
        if not text:
            return None, None
        
        txt_ids, padding = utils.xlm_tokenizer(text, self.tokenizer, max_len=self.max_len)
        return txt_ids, padding

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = str(row["Sample_id"])
        spot_id = str(row["Spot_id"])
        organ = row.get("organ", "Unknown")
        
        patch = self._load_patch(patient_id, spot_id)
        text_ids, padding = self._load_text(patient_id)
        
        if patch is None or text_ids is None:
            return None
        
        return {
            "patch": patch,
            "text_ids": torch.tensor(text_ids, dtype=torch.long),
            "paddings": torch.tensor(padding, dtype=torch.long),
            "patient_id": patient_id,
            "spot_id": spot_id,
            "organ_name": organ
        }


def collate_fn(batch):
    batch = [s for s in batch if s is not None]
    if not batch:
        return None
    return {
        "patch": torch.stack([s["patch"] for s in batch]),
        "text_ids": torch.stack([s["text_ids"] for s in batch]),
        "paddings": torch.stack([s["paddings"] for s in batch]),
        "patient_id": [s["patient_id"] for s in batch],
        "spot_id": [s["spot_id"] for s in batch],
        "organ_name": [s["organ_name"] for s in batch]
    }

# ============================================================================
# Model Loading
# ============================================================================

def strip_prefix(state_dict, prefix):
    """Remove prefix from state dict keys."""
    return {k.replace(prefix, "", 1) if k.startswith(prefix) else k: v 
            for k, v in state_dict.items()}


def load_integrated_model(checkpoint_path, device, classifier_checkpoint=None):
    """
    Load integrated ASPIRE-IO model from checkpoint.
    
    Supports multiple checkpoint formats:
    1. Integrated checkpoint with all components (multihead_model.*, organ_classifier.*)
    2. Separate multihead + classifier checkpoints
    3. Simple pretrain/finetune checkpoint (model_state_dict)
    """
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Detect checkpoint format
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
        organ2idx = checkpoint.get("organ2idx", ORGAN_TO_IDX)
        removed_organs = checkpoint.get("removed_organs", [])
    else:
        state_dict = checkpoint
        organ2idx = ORGAN_TO_IDX
        removed_organs = []
    
    # Check for integrated checkpoint (has multihead_model.* and organ_classifier.*)
    has_multihead_prefix = any(k.startswith("multihead_model.") for k in state_dict.keys())
    has_classifier = any(k.startswith("organ_classifier.") for k in state_dict.keys())
    has_fuse_proj = any("fuse_proj" in k for k in state_dict.keys())
    
    print(f"Checkpoint format: integrated={has_multihead_prefix}, classifier={has_classifier}, fuse_proj={has_fuse_proj}")
    print(f"Organs: {list(organ2idx.keys())}")
    if removed_organs:
        print(f"Removed organs: {removed_organs}")
    
    # Load MUSK backbone
    print("Loading MUSK backbone...")
    musk_model = load_pretrained_musk_model()
    musk_model = musk_model.to(device, dtype=torch.bfloat16)
    
    # Create MultiHeadModel
    organ_to_gene_dim = {organ: 100 for organ in organ2idx.keys()}
    multihead = MultiHeadModel(
        musk_model, organ_to_gene_dim, organ2idx,
        use_fuse_proj=has_fuse_proj
    ).to(device, dtype=torch.bfloat16)
    
    # Load multihead weights
    if has_multihead_prefix:
        # Integrated checkpoint - extract multihead_model.* keys
        multihead_state = {}
        for k, v in state_dict.items():
            if k.startswith("multihead_model."):
                new_key = k.replace("multihead_model.", "", 1)
                multihead_state[new_key] = v
        
        missing, unexpected = multihead.load_state_dict(multihead_state, strict=False)
        if missing:
            print(f"  Missing keys in multihead: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys in multihead: {len(unexpected)}")
    else:
        # Simple checkpoint - direct loading
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = strip_prefix(state_dict, "module.")
        
        missing, unexpected = multihead.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
    
    print(f"Loaded MultiHeadModel")
    
    # Load organ classifier
    organ_classifier = None
    
    if has_classifier:
        # Extract classifier from integrated checkpoint
        classifier_state = {}
        for k, v in state_dict.items():
            if k.startswith("organ_classifier."):
                new_key = k.replace("organ_classifier.", "", 1)
                classifier_state[new_key] = v
        
        # Determine classifier dimensions from weights
        input_dim = classifier_state["fc1.weight"].shape[1]
        hidden_dim = classifier_state["fc1.weight"].shape[0]
        num_organs = classifier_state["fc2.weight"].shape[0]
        
        organ_classifier = OrganClassifier(
            input_dim=input_dim,
            num_organs=num_organs,
            hidden_dim=hidden_dim
        ).to(device, dtype=torch.bfloat16)
        
        organ_classifier.load_state_dict(classifier_state)
        print(f"Loaded OrganClassifier: in={input_dim}, hidden={hidden_dim}, organs={num_organs}")
    
    elif classifier_checkpoint and os.path.exists(classifier_checkpoint):
        # Load from separate checkpoint
        print(f"Loading classifier from: {classifier_checkpoint}")
        cls_ckpt = torch.load(classifier_checkpoint, map_location=device, weights_only=False)
        
        classifier_state = cls_ckpt.get("classifier_state", cls_ckpt.get("model_state_dict", cls_ckpt))
        config = cls_ckpt.get("config", {})
        
        input_dim = config.get("embedding_dim", classifier_state["fc1.weight"].shape[1])
        hidden_dim = config.get("hidden_dim", classifier_state["fc1.weight"].shape[0])
        num_organs = classifier_state["fc2.weight"].shape[0]
        
        organ_classifier = OrganClassifier(
            input_dim=input_dim,
            num_organs=num_organs,
            hidden_dim=hidden_dim
        ).to(device, dtype=torch.bfloat16)
        
        organ_classifier.load_state_dict(classifier_state)
        print(f"Loaded OrganClassifier: in={input_dim}, hidden={hidden_dim}, organs={num_organs}")
    
    # Create integrated model
    model = IntegratedASPIREModel(multihead, organ_classifier, organ2idx)
    model.eval()
    
    return model

# ============================================================================
# Inference
# ============================================================================

def run_inference(model, loader, device, output_path, gene_names=None, use_classifier=True):
    """Run inference and save predictions."""
    
    model.eval()
    results = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Running inference"):
            if batch is None:
                continue
            
            patches = batch["patch"].to(device, dtype=torch.bfloat16)
            text_ids = batch["text_ids"].to(device)
            paddings = batch["paddings"].to(device)
            organ_names = batch["organ_name"]
            
            # Run model
            if use_classifier and model.organ_classifier is not None:
                immune_pred, gene_pred, organ_info = model(
                    patches, text_ids, paddings, use_classifier=True
                )
                predicted_organs = organ_info["organ_pred_names"] if organ_info else organ_names
            else:
                immune_pred, gene_pred, _ = model(
                    patches, text_ids, paddings, organs=organ_names, use_classifier=False
                )
                predicted_organs = organ_names
            
            # Collect results
            for i in range(len(batch["patient_id"])):
                result = {
                    "patient_id": batch["patient_id"][i],
                    "spot_id": batch["spot_id"][i],
                    "organ": batch["organ_name"][i],
                    "predicted_organ": predicted_organs[i] if use_classifier else batch["organ_name"][i],
                    "immune_score": immune_pred[i].float().cpu().item(),
                }
                
                # Add gene predictions
                gene_values = gene_pred[i].float().cpu().numpy()
                if gene_names:
                    for j, name in enumerate(gene_names[:len(gene_values)]):
                        result[f"gene_{name}"] = gene_values[j]
                else:
                    for j, val in enumerate(gene_values):
                        result[f"gene_{j}"] = val
                
                results.append(result)
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} predictions to: {output_path}")
    
    return df

# ============================================================================
# Main
# ============================================================================

def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # HuggingFace login
    login(token=args.hf_token)
    
    # Load tokenizer
    tokenizer = XLMRobertaTokenizer(args.tokenizer_path)
    
    print("=" * 70)
    print("ASPIRE-IO Inference")
    print("=" * 70)
    
    # Load model
    model = load_integrated_model(
        args.checkpoint,
        device,
        classifier_checkpoint=args.classifier_checkpoint
    )
    
    use_classifier = args.use_classifier and model.organ_classifier is not None
    if use_classifier:
        print("Using organ classifier for automatic routing")
    else:
        print("Using provided organ labels for routing")
    
    # Create dataset
    dataset = InferenceDataset(
        args.input_csv, args.patch_dir, args.text_dir,
        tokenizer, max_len=100
    )
    
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    # Load gene names if provided
    gene_names = None
    if args.gene_names_file and os.path.exists(args.gene_names_file):
        with open(args.gene_names_file, "r") as f:
            gene_names = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(gene_names)} gene names")
    
    # Run inference
    print("\nRunning inference...")
    df = run_inference(model, loader, device, args.output_path, 
                      gene_names=gene_names, use_classifier=use_classifier)
    
    print("\n" + "=" * 70)
    print("Inference complete!")
    print(f"Total samples: {len(df)}")
    print(f"Output saved to: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASPIRE-IO Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to integrated checkpoint or multihead checkpoint")
    parser.add_argument("--classifier_checkpoint", type=str, default=None,
                        help="Optional separate organ classifier checkpoint")
    parser.add_argument("--use_classifier", action="store_true", default=True,
                        help="Use organ classifier for automatic routing")
    parser.add_argument("--no_classifier", dest="use_classifier", action="store_false",
                        help="Don't use classifier, use provided organ labels")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="CSV with Sample_id, Spot_id, organ columns")
    parser.add_argument("--patch_dir", type=str, required=True)
    parser.add_argument("--text_dir", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--hf_token", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="predictions.csv")
    parser.add_argument("--gene_names_file", type=str, default=None,
                        help="Optional file with gene names (one per line)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    main(args)
