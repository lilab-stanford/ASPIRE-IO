"""
ASPIRE-IO Organ-Specific MLP Fine-tuning Script

Fine-tunes a single organ-specific MLP head within the pre-trained
MultiHeadModel. This script trains one organ's MLP at a time while
keeping the MUSK backbone and other organ heads frozen.

Input: Customized MUSK + MultiHeadModel checkpoint
Output: Same model structure with the target organ's MLP fine-tuned
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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance
from scipy.stats import pearsonr
import torchvision.transforms as transforms
from tqdm import tqdm
import random

# MUSK dependencies
from musk import utils, modeling
from timm.models import create_model
from transformers import XLMRobertaTokenizer
from huggingface_hub import login

# ============================================================================
# Configuration
# ============================================================================

ORGAN_TO_IDX = {
    "Bladder": 0, "Bowel": 1, "Breast": 2, "Lung": 3,
    "Lymph node": 4, "Prostate": 5, "Skin": 6
}

DEFAULT_CONFIG = {
    "batch_size": 108,
    "num_epochs": 100,
    "learning_rate": 5e-5,
    "betas": (0.9, 0.98),
    "hidden_dim": 1024,
    "gene_dim": 100,
    "dropout": 0.3,
    "patience": 10,
    "max_len": 100,
    # Fine-tuning target: immune (gene signature) only
    "lambda_immune_mse": 1.0, #adjust as needed during training to keep the RVD loss and the mse loss in balance
    "lambda_immune_rvd": 1.0, #adjust as needed during training to keep the RVD loss and the mse loss in balance
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
# Data Augmentation
# ============================================================================

def apply_he_color_augmentation(pil_img):
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    color_jitter = transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.01,
    )
    timg = to_tensor(pil_img)
    timg = color_jitter(timg)
    return to_pil(timg)


# ============================================================================
# Dataset (Organ-Specific)
# ============================================================================

class OrganSTDataset(Dataset):
    """Dataset filtered for a single organ for fine-tuning."""
    
    def __init__(self, csv_file, patch_dir, gene_dir, text_dir,
                 tokenizer, organ, gene_dim=100, max_len=100):
        self.df = pd.read_csv(csv_file)
        # Filter for specific organ only
        self.df = self.df[self.df["organ"] == organ].reset_index(drop=True)
        
        self.patch_dir = patch_dir
        self.gene_dir = gene_dir
        self.text_dir = text_dir
        self.tokenizer = tokenizer
        self.gene_dim = gene_dim
        self.max_len = max_len
        self.organ = organ
        self.transform = transforms.ToTensor()
        
        print(f"Organ Dataset [{organ}]: {len(self.df)} spots")

    def __len__(self):
        return len(self.df)

    def _load_patch(self, patient_id, spot_id):
        """Load and preprocess a patch from PNG or H5 file."""
        # Try PNG
        png_path = os.path.join(self.patch_dir, f"{patient_id}_{spot_id}.png")
        if os.path.exists(png_path):
            img = Image.open(png_path).convert("RGB")
            img = img.resize((384, 384), Image.BICUBIC)
            return img
        
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
        return img

    def _load_gene(self, patient_id, spot_id):
        # NOTE: Gene expression targets are not used for immune-score fine-tuning.
        # Kept only for backward compatibility and flexibility.
        if not self.gene_dir:
            return None
        gene_path = os.path.join(self.gene_dir, f"{patient_id}.csv")
        if not os.path.exists(gene_path):
            return None
        
        df = pd.read_csv(gene_path, index_col=0)
        if spot_id not in df.columns:
            return None
        
        arr = df[spot_id].values.astype(np.float32)
        if len(arr) != self.gene_dim:
            return None
        return arr

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
        patient_id = row["Sample_id"]
        spot_id = row["Spot_id"]
        immune_score = row.get("Predicted_Score", 0.0)
        
        img = self._load_patch(patient_id, spot_id)
        text_ids, padding = self._load_text(patient_id)
        
        if img is None or text_ids is None:
            return None

        # Augmentation
        angle = random.choice([0, 90, 180, 270])
        img = img.rotate(angle, expand=True)
        img = apply_he_color_augmentation(img)
        img = img.resize((384, 384), Image.BICUBIC)
        
        return {
            "patch": self.transform(img),
            "text_ids": torch.tensor(text_ids, dtype=torch.long),
            "paddings": torch.tensor(padding, dtype=torch.long),
            "immune_score": torch.tensor(immune_score, dtype=torch.bfloat16),
            "organ": self.organ,
            "unique_id": f"{patient_id}_{spot_id}"
        }


def collate_fn(batch):
    batch = [s for s in batch if s is not None]
    if not batch:
        return None
    return {
        "patch": torch.stack([s["patch"] for s in batch]),
        "text_ids": torch.stack([s["text_ids"] for s in batch]),
        "paddings": torch.stack([s["paddings"] for s in batch]),
        "immune_score": torch.stack([s["immune_score"] for s in batch]),
        "organ": torch.tensor([ORGAN_TO_IDX[s["organ"]] for s in batch]),
        "unique_id": [s["unique_id"] for s in batch]
    }

# ============================================================================
# Model Architecture (Same as pretrain.py - MultiHeadModel)
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
    """Multi-organ model with shared MUSK backbone and organ-specific heads."""
    
    def __init__(self, musk_model, organ_to_gene_dim, hidden_dim=1024, dropout=0.3):
        super().__init__()
        self.musk = musk_model
        self.organ_heads = nn.ModuleDict()
        self.organ_to_gene_dim = organ_to_gene_dim
        
        for organ, gene_dim in organ_to_gene_dim.items():
            head = OrganSpecificHead(gene_dim, hidden_dim=hidden_dim, dropout=dropout)
            self.organ_heads[organ] = head.to(dtype=torch.bfloat16)
        
        self.idx2organ = {v: k for k, v in ORGAN_TO_IDX.items()}

    def forward(self, patch, text_ids, paddings, organs):
        cur_device = patch.device
        text_ids = text_ids.to(cur_device, dtype=torch.long)
        paddings = paddings.to(cur_device, dtype=torch.long)
        
        # MUSK forward - vision branch
        vision_cls = self.musk(
            image=patch,
            with_head=False,
            out_norm=False,
            ms_aug=False,
            return_global=True
        )[0]
        
        # MUSK forward - text branch
        text_cls = self.musk(
            text_description=text_ids,
            padding_mask=paddings,
            with_head=False,
            out_norm=False,
            ms_aug=False,
            return_global=True
        )[1]

        fused = torch.cat([vision_cls, text_cls], dim=-1)  # (B, 2048)
        
        B = fused.size(0)
        organs_local = [self.idx2organ[int(x)] for x in organs.tolist()]
        
        out_dict = {}
        for org in set(organs_local):
            indices = [i for i, o in enumerate(organs_local) if o == org]
            if not indices:
                continue
            
            group_fused = fused[indices]
            head = self.organ_heads[org] if org in self.organ_heads else None
            
            if head is None:
                group_immune = torch.zeros(len(indices), device=cur_device, dtype=torch.bfloat16)
                group_gene = torch.zeros((len(indices), 1), device=cur_device, dtype=torch.bfloat16)
            else:
                group_immune, group_gene = head(group_fused)
            
            for local_idx, original_idx in enumerate(indices):
                out_dict[original_idx] = (group_immune[local_idx:local_idx+1],
                                          group_gene[local_idx:local_idx+1])
        
        immune_list, gene_list = [], []
        max_gene_dim = max(self.organ_to_gene_dim.values())
        
        for i in range(B):
            if i in out_dict:
                immune_list.append(out_dict[i][0])
                pred = out_dict[i][1]
                if pred.shape[1] < max_gene_dim:
                    pad = torch.zeros((1, max_gene_dim - pred.shape[1]), 
                                     device=pred.device, dtype=pred.dtype)
                    pred = torch.cat([pred, pad], dim=1)
                gene_list.append(pred)
            else:
                immune_list.append(torch.zeros(1, device=cur_device, dtype=torch.bfloat16))
                gene_list.append(torch.zeros((1, max_gene_dim), device=cur_device, dtype=torch.bfloat16))
        
        return torch.cat(immune_list, dim=0), torch.cat(gene_list, dim=0)

# ============================================================================
# Fine-tuning Utilities
# ============================================================================

def calculate_rvd_loss(pred, true, epsilon=1e-8):
    """Relative Variance Difference loss (same as pretrain.py)."""
    if pred.size(0) <= 1:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    
    if pred.dim() == 1:
        pred = pred.unsqueeze(1)
    if true.dim() == 1:
        true = true.unsqueeze(1)
    
    pred_var = torch.var(pred, dim=0, unbiased=True)
    true_var = torch.var(true, dim=0, unbiased=True)
    true_var_safe = true_var + epsilon
    
    squared_rel_error = ((pred_var - true_var) ** 2) / (true_var_safe ** 2)
    return torch.mean(squared_rel_error)


def freeze_all_except_organ(model, target_organ):
    """
    Freeze MUSK backbone and all organ heads EXCEPT the target organ.
    Only the target organ's MLP will be trainable.
    """
    # Freeze entire MUSK backbone
    for param in model.musk.parameters():
        param.requires_grad = False
    
    # Freeze all organ heads except target (immune head only)
    for organ_name, head in model.organ_heads.items():
        if organ_name == target_organ:
            # Train only immune signature head
            for param in head.immune_head.parameters():
                param.requires_grad = True
            for param in head.gene_head.parameters():
                param.requires_grad = False
            print(f"[TRAINABLE] Organ immune head: {organ_name}.immune_head")
        else:
            for param in head.parameters():
                param.requires_grad = False
    
    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


def load_pretrained_multihead_model(checkpoint_path, device):
    """Load pre-trained MUSK + MultiHeadModel from checkpoint."""
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create fresh MUSK model
    musk_model = load_pretrained_musk_model()
    musk_model = musk_model.to(device, dtype=torch.bfloat16)
    
    # Define organ dimensions (must match pretrained model)
    organ_to_gene_dim = {organ: 100 for organ in ORGAN_TO_IDX.keys()}
    
    # Create MultiHeadModel
    model = MultiHeadModel(musk_model, organ_to_gene_dim).to(device, dtype=torch.bfloat16)
    
    # Load state dict (handle DataParallel wrapper if present)
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    print(f"Loaded pre-trained model from: {checkpoint_path}")
    
    return model

# ============================================================================
# Training
# ============================================================================

def train_epoch(model, loader, optimizer, device, target_organ):
    model.train()
    mse = nn.MSELoss()
    total_loss = 0.0
    batch_count = 0
    
    for batch in tqdm(loader, desc=f"Fine-tuning [{target_organ}]", leave=False):
        if batch is None:
            continue
        
        batch_count += 1
        
        patches = batch["patch"].to(device, dtype=torch.bfloat16)
        text_ids = batch["text_ids"].to(device)
        paddings = batch["paddings"].to(device)
        true_immune = batch["immune_score"].to(device, dtype=torch.bfloat16)
        organs = batch["organ"].to(device)
        
        pred_immune, pred_gene = model(patches, text_ids, paddings, organs)

        loss_mse = mse(pred_immune.float().view(-1), true_immune.float().view(-1))
        loss_rvd = calculate_rvd_loss(pred_immune.view(-1), true_immune.view(-1))
        loss = (DEFAULT_CONFIG["lambda_immune_mse"] * loss_mse) + (DEFAULT_CONFIG["lambda_immune_rvd"] * loss_rvd)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / max(batch_count, 1)


def evaluate(model, loader, device, target_organ):
    model.eval()
    mse = nn.MSELoss()
    
    all_pred_immune, all_true_immune = [], []
    total_loss = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            batch_count += 1
            
            patches = batch["patch"].to(device, dtype=torch.bfloat16)
            text_ids = batch["text_ids"].to(device)
            paddings = batch["paddings"].to(device)
            true_immune = batch["immune_score"].to(device, dtype=torch.bfloat16)
            organs = batch["organ"].to(device)
            
            pred_immune, pred_gene = model(patches, text_ids, paddings, organs)

            loss_mse = mse(pred_immune.float().view(-1), true_immune.float().view(-1))
            loss_rvd = calculate_rvd_loss(pred_immune.view(-1), true_immune.view(-1))
            loss = (DEFAULT_CONFIG["lambda_immune_mse"] * loss_mse) + (DEFAULT_CONFIG["lambda_immune_rvd"] * loss_rvd)
            total_loss += loss.item()
            
            all_pred_immune.extend(pred_immune.float().cpu().numpy())
            all_true_immune.extend(true_immune.float().cpu().numpy())
    
    # Compute PCC
    pred_immune = np.array(all_pred_immune)
    true_immune = np.array(all_true_immune)
    immune_pcc = pearsonr(pred_immune, true_immune)[0] if len(pred_immune) > 1 else 0.0
    
    return {
        "loss": total_loss / max(batch_count, 1),
        "immune_pcc": immune_pcc,
    }

# ============================================================================
# Main
# ============================================================================

def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    config = DEFAULT_CONFIG.copy()
    config["lambda_immune_mse"] = args.lambda_immune_mse
    config["lambda_immune_rvd"] = args.lambda_immune_rvd
    config["num_epochs"] = args.num_epochs
    config["batch_size"] = args.batch_size
    
    # HuggingFace login
    login(token=args.hf_token)
    
    # Load tokenizer
    tokenizer = XLMRobertaTokenizer(args.tokenizer_path)
    
    print(f"Fine-tuning organ-specific MLP for: {args.organ}")
    
    # Load pre-trained MultiHeadModel
    model = load_pretrained_multihead_model(args.pretrained_checkpoint, device)
    
    # Freeze everything except target organ's MLP
    freeze_all_except_organ(model, args.organ)
    
    # Create organ-specific datasets
    train_dataset = OrganSTDataset(
        args.train_csv, args.patch_dir, args.gene_dir, args.text_dir,
        tokenizer, organ=args.organ, gene_dim=config["gene_dim"], max_len=config["max_len"]
    )
    val_dataset = OrganSTDataset(
        args.val_csv, args.patch_dir, args.gene_dir, args.text_dir,
        tokenizer, organ=args.organ, gene_dim=config["gene_dim"], max_len=config["max_len"]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                              shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"],
                            shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    # Optimizer - only for trainable parameters (target organ's MLP)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        betas=config["betas"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    # Training
    best_val_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(config["num_epochs"]):
        # Keep train/eval using the same weights.
        DEFAULT_CONFIG.update(config)
        train_loss = train_epoch(model, train_loader, optimizer, device, args.organ)
        val_metrics = evaluate(model, val_loader, device, args.organ)
        scheduler.step(val_metrics["loss"])
        
        print(
            f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"immune_pcc={val_metrics['immune_pcc']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            
            # Save the ENTIRE model
            torch.save({
                "model_state_dict": model.state_dict(),
                "finetuned_organ": args.organ,
                "val_metrics": val_metrics,
                "config": config
            }, args.save_path)
            print(f"  -> Saved fine-tuned model")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print("Early stopping triggered")
                break
    
    print(f"Fine-tuning complete for [{args.organ}]. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ASPIRE-IO Organ-Specific MLP Fine-tuning"
    )
    parser.add_argument("--pretrained_checkpoint", type=str, required=True,
                        help="Path to pre-trained MUSK + MultiHeadModel checkpoint")
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--patch_dir", type=str, required=True)
    
    # for backwards compatibility.
    parser.add_argument("--gene_dir", type=str, default="")
    parser.add_argument("--text_dir", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--hf_token", type=str, required=True)
    parser.add_argument("--organ", type=str, required=True,
                        choices=list(ORGAN_TO_IDX.keys()),
                        help="Target organ to fine-tune")
    parser.add_argument("--save_path", type=str, default="finetuned_checkpoint.pt")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lambda_immune_mse", type=float, default=DEFAULT_CONFIG["lambda_immune_mse"])
    parser.add_argument("--lambda_immune_rvd", type=float, default=DEFAULT_CONFIG["lambda_immune_rvd"])
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_CONFIG["num_epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    args = parser.parse_args()
    main(args)
