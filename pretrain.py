"""
ASPIRE-IO Pre-training Script

Pre-trains a vision-language model backbone with organ-specific prediction heads
for spatially resolved IO gene signature (immune score) prediction from H&E images.
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
import itertools
import math
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
    "num_epochs": 200,
    "learning_rate": 5e-5,
    "betas": (0.9, 0.98),
    "gene_dim": 100,
    "hidden_dim": 1024,
    "backbone_dim": 2048,  # 1024 vision_cls + 1024 text_cls (ms_aug=False)
    "dropout": 0.3,
    "lambda_immune_rvd": 1, #adjust as needed during training to keep the RVD loss and the mse loss in balance
    "lambda_gene_rvd": 0.01, #adjust as needed during training to keep the RVD loss and the mse loss in balance
    "freeze_layers": 4,  # keep last N layers trainable
    "accum_steps": 3,
    "max_len": 100,
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


def freeze_musk_layers(musk_model, num_trainable_layers=1):
    """Freeze all but the last N transformer layers."""
    num_layers = len(musk_model.beit3.encoder.layers)
    for i, layer in enumerate(musk_model.beit3.encoder.layers):
        if i < num_layers - num_trainable_layers:
            for param in layer.parameters():
                param.requires_grad = False
    print(f"Frozen {num_layers - num_trainable_layers}/{num_layers} transformer layers")

# ============================================================================
# Data Augmentation
# ============================================================================

def apply_he_color_augmentation(pil_img):
# augmentation for H&E images
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
# Dataset
# ============================================================================

class RoundRobinOrganSampler(torch.utils.data.Sampler):
    """Sampler that ensures balanced organ representation in each batch."""
    
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.total_spots = len(dataset.df)
        self.num_batches = math.ceil(self.total_spots / self.batch_size)
        
        self.organ_to_indices = {}
        for idx in range(len(dataset.df)):
            organ = dataset.df.iloc[idx]["organ"]
            self.organ_to_indices.setdefault(organ, []).append(idx)
        
        self.organs = sorted(self.organ_to_indices.keys())
        self.organ_cycles = {}
        for organ in self.organs:
            indices = self.organ_to_indices[organ][:]
            random.shuffle(indices)
            self.organ_cycles[organ] = itertools.cycle(indices)

    def __iter__(self):
        organ_cycle = itertools.cycle(self.organs)
        for _ in range(self.num_batches):
            batch = []
            for _ in range(self.batch_size):
                organ = next(organ_cycle)
                idx = next(self.organ_cycles[organ])
                batch.append(idx)
            yield batch

    def __len__(self):
        return self.num_batches


class STDataset(Dataset):
    """Spatial Transcriptomics Dataset for pre-training."""
    
    def __init__(self, csv_file, patch_dir, gene_dir, text_dir,
                 tokenizer, organs=None, gene_dim=100, max_len=100):
        self.df = pd.read_csv(csv_file)
        if organs:
            self.df = self.df[self.df["organ"].isin(organs)]
        
        self.patch_dir = patch_dir
        self.gene_dir = gene_dir
        self.text_dir = text_dir
        self.tokenizer = tokenizer
        self.gene_dim = gene_dim
        self.max_len = max_len
        self.transform = transforms.ToTensor()
        
        print(f"Dataset: {len(self.df)} spots, organs: {self.df['organ'].unique()}")

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
        """Load gene expression values for a spot."""
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
        """Load and tokenize clinical text."""
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
        organ = row["organ"]
        immune_score = row.get("Predicted_Score", 0.0)

        img = self._load_patch(patient_id, spot_id)
        gene = self._load_gene(patient_id, spot_id)
        text_ids, padding = self._load_text(patient_id)
        
        if img is None or gene is None or text_ids is None:
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
            "gene_array": torch.tensor(gene, dtype=torch.bfloat16),
            "organ": organ,
            "unique_id": f"{patient_id}_{spot_id}"
        }


def collate_fn(batch):
    """Collate function handling None samples."""
    batch = [s for s in batch if s is not None]
    if not batch:
        return None
    return {
        "patch": torch.stack([s["patch"] for s in batch]),
        "text_ids": torch.stack([s["text_ids"] for s in batch]),
        "paddings": torch.stack([s["paddings"] for s in batch]),
        "immune_score": torch.stack([s["immune_score"] for s in batch]),
        "gene_array": [s["gene_array"] for s in batch],
        "organ": torch.tensor([ORGAN_TO_IDX[s["organ"]] for s in batch]),
        "unique_id": [s["unique_id"] for s in batch]
    }

# ============================================================================
# Model Architecture
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
        
        #   - vision_cls: 1024 (mus_aug = False for MUSK)
        #   - text_cls: 1024
        #   - concat: 1024 + 1024 = 2048

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

        # Fuse vision and text features
        # With ms_aug=False: vision_cls=1024, text_cls=1024 -> concat=2048
        fused = torch.cat([vision_cls, text_cls], dim=-1)  # Shape: (B, 2048)
        
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
# RVD loss
# ============================================================================

def calculate_rvd_loss(pred, true, epsilon=1e-8):
    """Relative Variance Difference loss."""
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

# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(model, loader, optimizer, scheduler, device, config):
    """Train for one epoch with gradient accumulation."""
    model.train()
    mse_loss = nn.MSELoss()
    total_loss = 0.0
    batch_count = 0
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(tqdm(loader, desc="Training")):
        if batch is None:
            continue
        
        batch_count += 1
        
        patches = batch["patch"].to(device, dtype=torch.bfloat16)
        text_ids = batch["text_ids"].to(device)
        paddings = batch["paddings"].to(device)
        true_immune = batch["immune_score"].to(device, dtype=torch.bfloat16)
        true_gene_list = [g.to(device, dtype=torch.bfloat16) for g in batch["gene_array"]]
        organs = batch["organ"]
        
        pred_immune, pred_gene = model(patches, text_ids, paddings, organs)
        
        loss_total = torch.tensor(0.0, device=device, dtype=torch.float32)
        
        for org in organs.unique():
            indices = (organs == org).nonzero(as_tuple=True)[0]
            if len(indices) <= 1:
                continue
            
            organ_name = {v: k for k, v in ORGAN_TO_IDX.items()}[int(org)]
            expected_dim = config["gene_dim"]
            
            pred_immune_org = pred_immune[indices].view(-1)
            true_immune_org = true_immune[indices].view(-1)
            
            loss_immune_mse = mse_loss(pred_immune_org.float(), true_immune_org.float())
            loss_immune_rvd = calculate_rvd_loss(pred_immune_org, true_immune_org)
            
            pred_gene_org = pred_gene[indices, :expected_dim]
            true_gene_tensor = torch.stack([true_gene_list[i] for i in indices.tolist()])
            
            loss_gene_mse = mse_loss(pred_gene_org.float(), true_gene_tensor.float())
            loss_gene_rvd = calculate_rvd_loss(pred_gene_org, true_gene_tensor)
            
            loss_total += (loss_immune_mse + 
                          config["lambda_immune_rvd"] * loss_immune_rvd +
                          loss_gene_mse + 
                          config["lambda_gene_rvd"] * loss_gene_rvd)
        
        loss_batch = loss_total / config["accum_steps"]
        loss_batch.backward()
        
        if (batch_count % config["accum_steps"] == 0) or (batch_count == len(loader)):
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        
        total_loss += loss_total.item()
    
    return total_loss / max(batch_count, 1)


def validate(model, loader, device, config):
    """Validate model."""
    model.eval()
    mse_loss = nn.MSELoss()
    total_loss = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            if batch is None:
                continue
            
            batch_count += 1
            
            patches = batch["patch"].to(device, dtype=torch.bfloat16)
            text_ids = batch["text_ids"].to(device)
            paddings = batch["paddings"].to(device)
            true_immune = batch["immune_score"].to(device, dtype=torch.bfloat16)
            true_gene_list = [g.to(device, dtype=torch.bfloat16) for g in batch["gene_array"]]
            organs = batch["organ"]
            
            pred_immune, pred_gene = model(patches, text_ids, paddings, organs)
            
            loss_total = 0.0
            for org in organs.unique():
                indices = (organs == org).nonzero(as_tuple=True)[0]
                if len(indices) <= 1:
                    continue
                
                expected_dim = config["gene_dim"]
                
                pred_immune_org = pred_immune[indices].view(-1)
                true_immune_org = true_immune[indices].view(-1)
                pred_gene_org = pred_gene[indices, :expected_dim]
                true_gene_tensor = torch.stack([true_gene_list[i] for i in indices.tolist()])
                
                loss_total += mse_loss(pred_immune_org.float(), true_immune_org.float()).item()
                loss_total += mse_loss(pred_gene_org.float(), true_gene_tensor.float()).item()
            
            total_loss += loss_total
    
    return total_loss / max(batch_count, 1)

# ============================================================================
# Main
# ============================================================================

def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    config = DEFAULT_CONFIG.copy()
    config["num_epochs"] = args.num_epochs
    config["batch_size"] = args.batch_size
    
    # HuggingFace login (use your token)
    login(token=args.hf_token)
    
    # Load tokenizer
    tokenizer = XLMRobertaTokenizer(args.tokenizer_path)
    
    # Load and prepare MUSK model
    print("Loading MUSK backbone...")
    musk_model = load_pretrained_musk_model()
    freeze_musk_layers(musk_model, num_trainable_layers=config["freeze_layers"])
    musk_model = musk_model.to(device, dtype=torch.bfloat16)
    
    # Define organ dimensions
    organ_to_gene_dim = {organ: config["gene_dim"] for organ in ORGAN_TO_IDX.keys()}
    
    # Create model
    model = MultiHeadModel(musk_model, organ_to_gene_dim).to(device, dtype=torch.bfloat16)
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1 and args.multi_gpu:
        model = nn.DataParallel(model)
    
    # Datasets
    train_dataset = STDataset(
        args.train_csv, args.patch_dir, args.gene_dir, args.text_dir,
        tokenizer, gene_dim=config["gene_dim"], max_len=config["max_len"]
    )
    val_dataset = STDataset(
        args.val_csv, args.patch_dir, args.gene_dir, args.text_dir,
        tokenizer, gene_dim=config["gene_dim"], max_len=config["max_len"]
    )
    
    train_sampler = RoundRobinOrganSampler(train_dataset, config["batch_size"])
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"],
                            shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    # Optimizer and scheduler
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        betas=config["betas"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"] * len(train_loader)
    )
    
    # Training loop
    best_val_loss = float("inf")
    
    for epoch in range(config["num_epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, config)
        val_loss = validate(model, val_loader, device, config)
        
        print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Save checkpoint
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }, "checkpoint_latest.pt")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
            }, args.save_path)
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")
    
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASPIRE Pre-training")
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--patch_dir", type=str, required=True)
    parser.add_argument("--gene_dir", type=str, required=True)
    parser.add_argument("--text_dir", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--hf_token", type=str, required=True, help="HuggingFace token")
    parser.add_argument("--save_path", type=str, default="pretrain_best.pt")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_CONFIG["num_epochs"],
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"],
                        help="Batch size for training")
    args = parser.parse_args()
    main(args)
