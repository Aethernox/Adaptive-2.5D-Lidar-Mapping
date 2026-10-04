"""
Training Pipeline for Adaptive LiDAR Semantic Segmentation
Matches Section 8 & Section 26.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.semantic_kitti import SemanticKittiDataset
from models.segmentation import AdaptivePolarNet
from training.losses import DistanceWeightedLoss
from perception.preprocessing import create_polar_pseudo_image


class PolarDatasetTorch(Dataset):
    """PyTorch Dataset wrapper over SemanticKittiDataset."""
    def __init__(self, kitti_ds: SemanticKittiDataset, num_rings: int = 128, num_sectors: int = 256):
        self.kitti_ds = kitti_ds
        self.num_rings = num_rings
        self.num_sectors = num_sectors

    def __len__(self) -> int:
        return len(self.kitti_ds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        frame = self.kitti_ds[idx]
        pts = frame.points
        
        # Ground truth labels mapped to learning classes (0..19)
        raw_sem = frame.semantic_labels
        learning_labels = self.kitti_ds.raw_to_learning(raw_sem) if raw_sem is not None else np.zeros(len(pts), dtype=np.uint8)
        
        # Create polar pseudo-image
        pseudo_img, r_idx, s_idx, valid = create_polar_pseudo_image(
            pts, num_rings=self.num_rings, num_sectors=self.num_sectors
        )
        
        # Create 2D ground truth label grid (128, 256)
        target_grid = np.zeros((self.num_rings, self.num_sectors), dtype=np.int64)
        if np.any(valid):
            v_r = r_idx[valid]
            v_s = s_idx[valid]
            v_lbl = learning_labels[valid]
            # Mode / assignment per cell
            target_grid[v_r, v_s] = v_lbl

        return {
            "pseudo_image": torch.from_numpy(pseudo_img).float(),
            "target": torch.from_numpy(target_grid).long()
        }


def train_pipeline(config_path: str = "configs/training.yaml", resume_checkpoint: Optional[str] = None):
    # 1. Load configuration
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)["training"]

    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Training] Using Device: {device}")

    # 2. Datasets
    print("[Training] Initializing SemanticKITTI datasets...")
    train_kitti = SemanticKittiDataset(subset="debug", split_mode="train", max_frames=50)
    val_kitti = SemanticKittiDataset(subset="debug", split_mode="val", max_frames=20)

    train_ds = PolarDatasetTorch(train_kitti)
    val_ds = PolarDatasetTorch(val_kitti)

    batch_size = cfg.get("batch_size", 2)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 3. Model
    model = AdaptivePolarNet(
        num_classes=20,
        in_channels=8,
        encoder_channels=64,
        backbone_channels=[64, 128, 256],
        upsample_channels=[128, 128, 128]
    ).to(device)

    # 4. Optimizer & Scheduler
    lr = cfg.get("optimizer", {}).get("lr", 0.002)
    weight_decay = cfg.get("optimizer", {}).get("weight_decay", 0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    epochs = cfg.get("epochs", 5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # 5. Loss Function
    criterion = DistanceWeightedLoss(num_classes=20, gamma=2.0)

    # 6. Logging & Checkpoints
    checkpoints_dir = Path(cfg.get("checkpoints_dir", "checkpoints"))
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(cfg.get("logging", {}).get("tensorboard_dir", "logs/tensorboard"))
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    start_epoch = 1
    best_val_loss = float("inf")

    # Optional Resume
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"[Training] Resuming from checkpoint: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 1) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))

    # Scaler for Mixed Precision
    use_amp = cfg.get("mixed_precision", True) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"[Training] Starting training loop for {epochs} epochs...\n")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            images = batch["pseudo_image"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / max(len(train_loader), 1)

        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["pseudo_image"].to(device)
                targets = batch["target"].to(device)
                logits = model(images)
                loss = criterion(logits, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / max(len(val_loader), 1)
        epoch_time = time.time() - start_time

        print(f"Epoch [{epoch:2d}/{epochs:2d}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {epoch_time:.2f}s")

        # TensorBoard logging
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        # Save Latest Checkpoint
        latest_path = checkpoints_dir / "latest.pt"
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        }, latest_path)

        # Save Best Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = checkpoints_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": best_val_loss
            }, best_path)
            print(f"  --> Saved new best model to {best_path}")

    writer.close()
    print("\n[Training] Training completed successfully!\n")


def main():
    parser = argparse.ArgumentParser(description="Train AdaptivePolarNet on SemanticKITTI")
    parser.add_argument("--config", type=str, default="configs/training.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    train_pipeline(config_path=args.config, resume_checkpoint=args.resume)


if __name__ == "__main__":
    main()
