"""
Cell Segmentation Training Script

Train a UNet model for medical cell segmentation.
Usage:
    python train.py --image_dir /path/to/images --mask_dir /path/to/masks --exp_name my_exp
"""

import os
import argparse
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from tqdm import tqdm

import segmentation_models_pytorch as smp
from metrics import compute_all_metrics, MetricTracker


# ============================================================================
# Configuration
# ============================================================================

def get_args():
    parser = argparse.ArgumentParser(description='Train UNet for cell segmentation')
    
    # Data paths
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Path to training images directory')
    parser.add_argument('--mask_dir', type=str, required=True,
                        help='Path to training masks directory')
    
    # Experiment
    parser.add_argument('--exp_name', type=str, default='unet_exp',
                        help='Experiment name for output directory')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Base output directory')
    
    # Training params
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    
    # Model params
    parser.add_argument('--img_size', type=int, default=256,
                        help='Input image size (will resize to this)')
    parser.add_argument('--in_channels', type=int, default=1,
                        help='Number of input channels (1 for grayscale)')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of output classes')
    
    # Other
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Train/Val split ratio')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Dataset
# ============================================================================

class CellSegDataset(Dataset):
    """Dataset for cell segmentation with configurable image/mask directories."""
    
    def __init__(self, image_dir: str, mask_dir: str, img_size: int = 256,
                 file_list: Optional[List[str]] = None):
        """
        Args:
            image_dir: Path to images directory
            mask_dir: Path to masks directory
            img_size: Target image size (will resize to img_size x img_size)
            file_list: Optional list of filenames to use (for train/val split)
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        
        # Get all image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        all_files = [f.name for f in self.image_dir.iterdir() 
                     if f.suffix.lower() in image_extensions]
        
        if file_list is not None:
            self.files = [f for f in all_files if f in file_list]
        else:
            self.files = all_files
        
        self.files.sort()
        
        if len(self.files) == 0:
            raise ValueError(f"No images found in {image_dir}")
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = self.files[idx]
        
        # Load image
        img_path = self.image_dir / img_name
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
        
        # Load mask - try different extensions
        mask_name = Path(img_name).stem
        mask_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
            candidate = self.mask_dir / f"{mask_name}{ext}"
            if candidate.exists():
                mask_path = candidate
                break
        
        if mask_path is None:
            # Try original extension
            mask_path = self.mask_dir / img_name
        
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((self.img_size, self.img_size), Image.Resampling.BOX)
        mask = np.array(mask, dtype=np.float32) / 255.0  # 转为0-1
        mask = (mask > 0.5).astype(np.int64)  # 二值化
        mask = torch.from_numpy(mask)  # (H, W)
        
        return image, mask


# ============================================================================
# Model
# ============================================================================

def get_model(in_channels: int = 1, num_classes: int = 2) -> nn.Module:
    """
    Create UNet model with specified architecture.
    
    Using segmentation_models_pytorch with custom encoder configuration.
    Encoder channels: 32, 64, 128, 256, 512 (5 blocks)
    """
    model = smp.Unet(
        encoder_name="resnet18",  # Use ResNet18 as base encoder
        encoder_weights=None,     # Train from scratch for grayscale
        in_channels=in_channels,
        classes=num_classes,
        encoder_depth=5,
        decoder_channels=(256, 128, 64, 32, 16),
    )
    
    return model


# ============================================================================
# Training
# ============================================================================

def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: optim.Optimizer, device: torch.device) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(loader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)  # (B, 1, H, W)
        masks = masks.to(device)    # (B, H, W)
        
        optimizer.zero_grad()
        
        outputs = model(images)  # (B, C, H, W)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> Tuple[float, dict]:
    """Validate model and compute metrics."""
    model.eval()
    total_loss = 0.0
    tracker = MetricTracker()
    
    for images, masks in tqdm(loader, desc='Validating'):
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        total_loss += loss.item() * images.size(0)
        
        # Get predictions
        preds = outputs.argmax(dim=1).cpu().numpy()  # (B, H, W)
        targets = masks.cpu().numpy()
        
        # Compute metrics for each sample
        for pred, target in zip(preds, targets):
            metrics = compute_all_metrics(pred, target)
            tracker.update(metrics)
    
    avg_loss = total_loss / len(loader.dataset)
    avg_metrics = tracker.get_average()
    
    return avg_loss, avg_metrics


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int,
                    metrics: dict, save_path: str):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, save_path)


# ============================================================================
# Main
# ============================================================================

def main():
    args = get_args()
    set_seed(args.seed)
    
    # Setup output directory
    exp_dir = Path(args.output_dir) / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = exp_dir / 'train_log.txt'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Experiment: {args.exp_name}")
    logger.info(f"Output directory: {exp_dir}")
    logger.info(f"Arguments: {vars(args)}")
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Dataset
    full_dataset = CellSegDataset(args.image_dir, args.mask_dir, args.img_size)
    logger.info(f"Total samples: {len(full_dataset)}")
    
    # Train/Val split
    train_size = int(len(full_dataset) * args.train_ratio)
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    logger.info(f"Train samples: {train_size}, Val samples: {val_size}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Model
    model = get_model(args.in_channels, args.num_classes).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_dice = 0.0
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        logger.info(f"Val Loss: {val_loss:.4f}")
        logger.info(f"Val Metrics: dice={val_metrics['dice']:.4f} | "
                   f"hd95={val_metrics['hd95']:.2f} | "
                   f"jaccard={val_metrics['jaccard']:.4f} | "
                   f"sensitivity={val_metrics['sensitivity']:.4f} | "
                   f"ppv={val_metrics['ppv']:.4f}")
        
        scheduler.step()
        
        # Save best model
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            save_checkpoint(model, optimizer, epoch, val_metrics,
                          str(exp_dir / 'best_model.pth'))
            logger.info(f"New best model saved! Dice: {best_dice:.4f}")
        
        # Save periodic checkpoint
        if epoch % args.save_freq == 0:
            save_checkpoint(model, optimizer, epoch, val_metrics,
                          str(exp_dir / f'epoch_{epoch}.pth'))
            logger.info(f"Checkpoint saved: epoch_{epoch}.pth")
    
    # Save final model
    save_checkpoint(model, optimizer, args.epochs, val_metrics,
                   str(exp_dir / 'final_model.pth'))
    logger.info(f"\nTraining completed! Best Dice: {best_dice:.4f}")
    logger.info(f"Models saved in: {exp_dir}")


if __name__ == '__main__':
    main()
