"""
ASPIRE-IO MoE (Organ Classifier) Training Script

Trains a simple MoE (Mixture of Experts) wiring network for organ classifier that routes inputs to the correct 
organ-specific MLP head based on MUSK embeddings.

The classifier learns to identify which organ a patch belongs to,
enabling intelligent routing during inference.

Supports three modality modes:
- "image": Use only vision features from MUSK
- "text": Use only text features from MUSK  
- "multimodal": Use concatenated vision + text features (default)
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance
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

# Organ to index mapping
ORGAN_TO_IDX = {
    "Bladder": 0, "Bowel": 1, "Breast": 2, "Lung": 3,
    "Lymph node": 4, "Prostate": 5, "Skin": 6
}
NUM_ORGANS = len(ORGAN_TO_IDX)

# Modality modes and their input dimensions (with ms_aug=False)
MODALITY_MODES = {
    "image": 1024,       # Vision CLS only
    "text": 1024,        # Text CLS only
    "both": 2048,        # Vision + Text concatenated (default)
    "multimodal": 2048,  # Alias for "both"
}

DEFAULT_CONFIG = {
    "batch_size": 108,
    "num_epochs": 100,
    "learning_rate": 1e-3,
    "betas": (0.9, 0.98),
    "hidden_dim": 256,
    "dropout": 0.1,
    "patience": 15,
    "max_len": 100,
    "label_smoothing": 0.0,
    "mode": "both",
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
    """Apply H&E-appropriate color augmentation."""
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(1 + np.random.uniform(-0.2, 0.2))
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1 + np.random.uniform(-0.2, 0.2))
    enhancer = ImageEnhance.Color(pil_img)
    pil_img = enhancer.enhance(1 + np.random.uniform(-0.2, 0.2))
    return pil_img

# ============================================================================
# Dataset
# ============================================================================

class MoEDataset(Dataset):
    """Dataset for MoE organ classification training."""
    
    def __init__(self, csv_file, patch_dir, gene_dir, text_dir,
                 tokenizer, max_len=100):
        self.df = pd.read_csv(csv_file)
        self.patch_dir = patch_dir
        self.gene_dir = gene_dir
        self.text_dir = text_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transforms.ToTensor()
        
        # Filter to valid organs only
        valid_organs = set(ORGAN_TO_IDX.keys())
        before = len(self.df)
        self.df = self.df[self.df["organ"].isin(valid_organs)].reset_index(drop=True)
        after = len(self.df)
        if before != after:
            print(f"[Dataset] Filtered {before - after} samples with invalid/removed organs")
        
        # Count organ distribution
        organ_counts = self.df["organ"].value_counts().to_dict()
        print(f"MoE Dataset: {len(self.df)} samples, organs: {organ_counts}")

    def __len__(self):
        return len(self.df)

    def _load_patch(self, patient_id, spot_id):
        """Load and preprocess a patch from PNG or H5 file."""
        # Try PNG first
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
        organ = row.get("organ", "unknown")
        
        # Get organ index for classification
        organ_idx = ORGAN_TO_IDX.get(organ, -1)
        if organ_idx == -1:
            return None
        
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
            "organ_idx": torch.tensor(organ_idx, dtype=torch.long),
            "organ": organ,
            "sample_id": patient_id,
            "spot_id": spot_id,
        }


def collate_fn(batch):
    batch = [s for s in batch if s is not None]
    if not batch:
        return None
    return {
        "patch": torch.stack([s["patch"] for s in batch]),
        "text_ids": torch.stack([s["text_ids"] for s in batch]),
        "paddings": torch.stack([s["paddings"] for s in batch]),
        "organ_idx": torch.stack([s["organ_idx"] for s in batch]),
        "organ": [s["organ"] for s in batch],
        "sample_id": [s["sample_id"] for s in batch],
        "spot_id": [s["spot_id"] for s in batch],
    }

# ============================================================================
# Organ Classifier (Simple 2-FC Network)
# ============================================================================

class OrganClassifier(nn.Module):
    """
    Simple MLP classifier for organ routing.
    
    Takes MUSK CLS embeddings (vision + text) and outputs organ logits.
    Architecture: fc1 -> GELU -> dropout -> fc2
    """
    
    def __init__(self, input_dim=2048, num_organs=NUM_ORGANS, 
                 hidden_dim=256, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_organs)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        logits = self.fc2(x)
        return logits


class EmbeddingExtractor(nn.Module):
    """Extract MUSK embeddings for organ classification."""
    
    def __init__(self, musk_model, mode="both"):
        super().__init__()
        self.musk = musk_model
        self.mode = mode.lower()
        
    @torch.no_grad()
    def forward(self, patch, text_ids, paddings):
        """Extract embeddings based on mode."""
        # Vision branch
        out_img = self.musk(
            image=patch, with_head=False, out_norm=False,
            ms_aug=False, return_global=True
        )
        vision_cls = out_img[0]
        
        # Text branch
        out_txt = self.musk(
            text_description=text_ids, padding_mask=paddings,
            with_head=False, out_norm=False, ms_aug=False, return_global=True
        )
        text_cls = out_txt[1]
        
        # Select based on mode
        if self.mode == "image" or self.mode == "vision":
            return vision_cls
        elif self.mode == "text":
            return text_cls
        else:  # both / multimodal
            return torch.cat([vision_cls, text_cls], dim=-1)


class MoEModel(nn.Module):
    """
    MUSK backbone + Organ Classifier for routing.
    
    This is the training wrapper that combines MUSK embeddings
    with the organ classifier.
    """
    
    def __init__(self, musk_model, mode="both", num_organs=NUM_ORGANS,
                 hidden_dim=256, dropout=0.1):
        super().__init__()
        
        self.mode = mode.lower()
        if self.mode == "multimodal":
            self.mode = "both"
            
        self.num_organs = num_organs
        self.embed_extractor = EmbeddingExtractor(musk_model, self.mode)
        
        # Determine input dimension
        in_dim = MODALITY_MODES.get(self.mode, 2048)
        
        # Organ classifier
        self.classifier = OrganClassifier(
            input_dim=in_dim,
            num_organs=num_organs,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        print(f"MoE Organ Classifier initialized: mode='{self.mode}', in_dim={in_dim}, num_organs={num_organs}")

    def forward(self, patch, text_ids, paddings):
        """Forward pass: extract embeddings -> classify organ."""
        # Extract embeddings (frozen MUSK)
        with torch.no_grad():
            embeddings = self.embed_extractor(patch, text_ids, paddings)
        
        # Classify organ
        logits = self.classifier(embeddings.to(dtype=self.classifier.fc1.weight.dtype))
        
        return logits

# ============================================================================
# Training
# ============================================================================

def train_epoch(model, loader, optimizer, device, label_smoothing=0.0):
    model.train()
    model.classifier.train()
    
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Training [{model.mode}]", leave=False)
    for batch_idx, batch in enumerate(pbar):
        if batch is None:
            continue
        
        patches = batch["patch"].to(device, dtype=torch.bfloat16)
        text_ids = batch["text_ids"].to(device)
        paddings = batch["paddings"].to(device)
        true_organ = batch["organ_idx"].to(device)
        
        logits = model(patches, text_ids, paddings)
        loss = criterion(logits.float(), true_organ)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * true_organ.size(0)
        
        # Track accuracy
        preds = logits.argmax(dim=1)
        correct += (preds == true_organ).sum().item()
        total += true_organ.size(0)
        
        # Update progress bar
        if (batch_idx + 1) % 10 == 0:
            pbar.set_postfix({
                "loss": f"{total_loss/total:.4f}",
                "acc": f"{correct/total:.4f}"
            })
    
    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            if batch is None:
                continue
            
            patches = batch["patch"].to(device, dtype=torch.bfloat16)
            text_ids = batch["text_ids"].to(device)
            paddings = batch["paddings"].to(device)
            true_organ = batch["organ_idx"].to(device)
            
            logits = model(patches, text_ids, paddings)
            loss = criterion(logits.float(), true_organ)
            
            total_loss += loss.item() * true_organ.size(0)
            
            preds = logits.argmax(dim=1)
            correct += (preds == true_organ).sum().item()
            total += true_organ.size(0)
    
    accuracy = correct / max(total, 1)
    
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": accuracy
    }

# ============================================================================
# Main
# ============================================================================

def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    config = DEFAULT_CONFIG.copy()
    config["num_epochs"] = args.num_epochs
    config["batch_size"] = args.batch_size
    config["mode"] = args.mode
    
    print("=" * 70)
    print("ASPIRE-IO ORGAN CLASSIFIER TRAINING")
    print("=" * 70)
    
    # HuggingFace login
    login(token=args.hf_token)
    
    # Load tokenizer
    tokenizer = XLMRobertaTokenizer(args.tokenizer_path)
    
    print(f"\nTraining organ classifier with mode: {args.mode}")
    print(f"Device: {device}")
    
    # Load MUSK backbone (frozen)
    print("\nLoading MUSK backbone...")
    musk_model = load_pretrained_musk_model()
    musk_model.eval()
    for p in musk_model.parameters():
        p.requires_grad = False
    musk_model = musk_model.to(device, dtype=torch.bfloat16)
    print("Froze MUSK backbone")
    
    # Create MoE organ classifier
    model = MoEModel(
        musk_model,
        mode=args.mode,
        num_organs=NUM_ORGANS,
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"]
    ).to(device, dtype=torch.bfloat16)
    
    trainable = sum(p.numel() for p in model.classifier.parameters())
    print(f"Trainable parameters (classifier): {trainable:,}")
    
    # Datasets
    print("\nLoading datasets...")
    train_dataset = MoEDataset(
        args.train_csv, args.patch_dir, args.gene_dir, args.text_dir,
        tokenizer, max_len=config["max_len"]
    )
    val_dataset = MoEDataset(
        args.val_csv, args.patch_dir, args.gene_dir, args.text_dir,
        tokenizer, max_len=config["max_len"]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                              shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"],
                            shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    # Optimizer (only train classifier)
    optimizer = optim.AdamW(
        model.classifier.parameters(),
        lr=config["learning_rate"],
        betas=config["betas"],
        weight_decay=0.01
    )
    
    # Training loop
    print(f"\nStarting training for {config['num_epochs']} epochs...")
    print("=" * 70)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(config["num_epochs"]):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, 
            label_smoothing=config["label_smoothing"]
        )
        val_metrics = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch+1:3d} [{args.mode}]: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['accuracy']:.4f}")
        
        # Save best model based on accuracy
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            patience_counter = 0
            
            # Save checkpoint
            torch.save({
                "classifier_state": model.classifier.state_dict(),
                "config": {
                    "embedding_dim": MODALITY_MODES.get(model.mode, 2048),
                    "hidden_dim": config["hidden_dim"],
                    "dropout": config["dropout"],
                    "mode": args.mode,
                },
                "organ2idx": ORGAN_TO_IDX,
                "removed_organs": list(REMOVED_ORGANS),
                "best_val_acc": best_val_acc,
            }, args.save_path)
            print(f"  -> Saved best model to {args.save_path} (accuracy={best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print("Early stopping triggered")
                break
    
    print("=" * 70)
    print(f"Training complete [{args.mode}]. Best val_accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASPIRE-IO Organ Classifier Training")
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--patch_dir", type=str, required=True)
    parser.add_argument("--gene_dir", type=str, required=True)
    parser.add_argument("--text_dir", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--hf_token", type=str, required=True)
    parser.add_argument("--mode", type=str, default="both",
                        choices=["image", "text", "both", "multimodal"],
                        help="Modality mode (default: both)")
    parser.add_argument("--save_path", type=str, default="organ_classifier.pt")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_CONFIG["num_epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    args = parser.parse_args()
    main(args)
