"""
Cell Segmentation Evaluation Script

Evaluate trained model(s) on any dataset (supports OOD evaluation).
Usage:
    python eval.py --ckpt_dir outputs/my_exp --image_dir /path/to/test-image --mask_dir /path/to/test-mask
    
    # Evaluate specific checkpoint
    python eval.py --ckpt_path outputs/my_exp/best_model.pth --image_dir ... --mask_dir ...
    
    # With visualization output
    python eval.py --ckpt_path outputs/my_exp/best_model.pth --image_dir ... --mask_dir ... --save_vis --num_vis 20
"""

import os
import argparse
import logging
import random
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

import segmentation_models_pytorch as smp
from metrics import compute_all_metrics, MetricTracker


# ============================================================================
# Configuration
# ============================================================================

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate cell segmentation model')
    
    # Checkpoint
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Directory containing checkpoints (will evaluate all)')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Path to specific checkpoint file')
    
    # Data paths
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Path to test images directory')
    parser.add_argument('--mask_dir', type=str, required=True,
                        help='Path to test masks directory')
    
    # Model params (should match training)
    parser.add_argument('--img_size', type=int, default=256,
                        help='Input image size')
    parser.add_argument('--in_channels', type=int, default=1,
                        help='Number of input channels')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of output classes')
    
    # Other
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output JSON file for results (optional)')
    
    # Visualization
    parser.add_argument('--save_vis', action='store_true',
                        help='Save visualization images (original, GT, prediction)')
    parser.add_argument('--num_vis', type=int, default=10,
                        help='Number of random samples to visualize')
    
    args = parser.parse_args()
    
    if args.ckpt_dir is None and args.ckpt_path is None:
        parser.error("Either --ckpt_dir or --ckpt_path must be provided")
    
    return args


# ============================================================================
# Dataset
# ============================================================================

class CellSegDataset(Dataset):
    """Dataset for cell segmentation evaluation."""
    
    def __init__(self, image_dir: str, mask_dir: str, img_size: int = 256):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        self.files = sorted([f.name for f in self.image_dir.iterdir() 
                            if f.suffix.lower() in image_extensions])
        
        if len(self.files) == 0:
            raise ValueError(f"No images found in {image_dir}")
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int):
        img_name = self.files[idx]
        
        # Load image
        img_path = self.image_dir / img_name
        image = Image.open(img_path).convert('L')
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)
        
        # Load mask
        mask_name = Path(img_name).stem
        mask_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
            candidate = self.mask_dir / f"{mask_name}{ext}"
            if candidate.exists():
                mask_path = candidate
                break
        
        if mask_path is None:
            mask_path = self.mask_dir / img_name
        
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((self.img_size, self.img_size), Image.Resampling.BOX)
        mask = np.array(mask, dtype=np.float32) / 255.0  # 转为0-1
        mask = (mask > 0.5).astype(np.int64)  # 二值化
        mask = torch.from_numpy(mask)  # (H, W)

        return image, mask, img_name


# ============================================================================
# Model
# ============================================================================

def get_model(in_channels: int = 1, num_classes: int = 2) -> nn.Module:
    """Create UNet model (same architecture as training)."""
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=in_channels,
        classes=num_classes,
        encoder_depth=5,
        decoder_channels=(256, 128, 64, 32, 16),
    )
    return model


def load_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device) -> dict:
    """Load checkpoint and return metadata."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Evaluate model on dataset."""
    model.eval()
    tracker = MetricTracker()
    
    for images, masks, _ in tqdm(loader, desc='Evaluating'):
        images = images.to(device)
        
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        targets = masks.numpy()
        
        for pred, target in zip(preds, targets):
            metrics = compute_all_metrics(pred, target)
            tracker.update(metrics)
    
    return tracker.get_average()


def find_checkpoints(ckpt_dir: str) -> List[str]:
    """Find all checkpoint files in directory."""
    ckpt_dir = Path(ckpt_dir)
    ckpts = []
    
    for f in ckpt_dir.iterdir():
        if f.suffix == '.pth':
            ckpts.append(str(f))
    
    return sorted(ckpts)


# ============================================================================
# Visualization
# ============================================================================

@torch.no_grad()
def save_visualizations(model: nn.Module, dataset: CellSegDataset, 
                        device: torch.device, output_dir: Path, 
                        num_samples: int, img_size: int, logger):
    """
    Save visualization images for random samples.
    
    For each sample, saves:
    - {name}_original.png: Original grayscale image
    - {name}_gt.png: Ground truth mask (black=bg, white=cell)
    - {name}_pred.png: Predicted mask (black=bg, white=cell)
    """
    model.eval()
    
    # Random sample indices
    total = len(dataset)
    num_samples = min(num_samples, total)
    indices = random.sample(range(total), num_samples)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving {num_samples} visualizations to: {output_dir}")
    
    for idx in tqdm(indices, desc='Saving visualizations'):
        image, mask, img_name = dataset[idx]
        name_stem = Path(img_name).stem
        
        # Get prediction
        image_input = image.unsqueeze(0).to(device)  # (1, 1, H, W)
        output = model(image_input)
        pred = output.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)
        
        # Convert to images (keep at img_size, no resize back)
        # Original image: scale back to 0-255
        orig_np = (image.squeeze(0).numpy() * 255).astype(np.uint8)
        orig_img = Image.fromarray(orig_np, mode='L')
        
        # GT mask: 0=black (bg), 1=white (cell)
        gt_np = (mask.numpy() * 255).astype(np.uint8)
        gt_img = Image.fromarray(gt_np, mode='L')
        
        # Prediction: 0=black (bg), 1=white (cell)
        pred_np = (pred * 255).astype(np.uint8)
        pred_img = Image.fromarray(pred_np, mode='L')
        
        # Save images
        orig_img.save(output_dir / f"{name_stem}_original.png")
        gt_img.save(output_dir / f"{name_stem}_gt.png")
        pred_img.save(output_dir / f"{name_stem}_pred.png")
    
    logger.info(f"Visualizations saved!")


# ============================================================================
# Main
# ============================================================================

def main():
    args = get_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
    )
    logger = logging.getLogger(__name__)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Dataset
    dataset = CellSegDataset(args.image_dir, args.mask_dir, args.img_size)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    logger.info(f"Test samples: {len(dataset)}")
    logger.info(f"Image dir: {args.image_dir}")
    logger.info(f"Mask dir: {args.mask_dir}")
    
    # Get checkpoint paths
    if args.ckpt_path:
        ckpt_paths = [args.ckpt_path]
    else:
        ckpt_paths = find_checkpoints(args.ckpt_dir)
        logger.info(f"Found {len(ckpt_paths)} checkpoints in {args.ckpt_dir}")
    
    # Visualization output directory (if enabled)
    vis_output_dir = None
    if args.save_vis:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        vis_output_dir = Path("outputs") / "eval" / timestamp
    
    # Evaluate each checkpoint
    results = {}
    
    for ckpt_path in ckpt_paths:
        ckpt_name = Path(ckpt_path).stem
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating: {ckpt_name}")
        
        # Load model
        model = get_model(args.in_channels, args.num_classes).to(device)
        ckpt_info = load_checkpoint(model, ckpt_path, device)
        
        if 'epoch' in ckpt_info:
            logger.info(f"Checkpoint epoch: {ckpt_info['epoch']}")
        
        # Evaluate
        metrics = evaluate(model, loader, device)
        
        logger.info(f"Results for {ckpt_name}:")
        logger.info(f"  DICE:        {metrics['dice']:.4f}")
        logger.info(f"  HD95:        {metrics['hd95']:.2f}")
        logger.info(f"  Jaccard:     {metrics['jaccard']:.4f}")
        logger.info(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        logger.info(f"  PPV:         {metrics['ppv']:.4f}")
        
        results[ckpt_name] = metrics
        
        # Save visualizations (only for first/single checkpoint)
        if args.save_vis and vis_output_dir is not None:
            ckpt_vis_dir = vis_output_dir / ckpt_name
            save_visualizations(model, dataset, device, ckpt_vis_dir, 
                              args.num_vis, args.img_size, logger)
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"{'Checkpoint':<20} {'DICE':<10} {'HD95':<10} {'JI':<10} {'Sen':<10} {'PPV':<10}")
    logger.info("-" * 70)
    
    for name, m in results.items():
        logger.info(f"{name:<20} {m['dice']:<10.4f} {m['hd95']:<10.2f} "
                   f"{m['jaccard']:<10.4f} {m['sensitivity']:<10.4f} {m['ppv']:<10.4f}")
    
    # Save results to JSON if specified
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'image_dir': args.image_dir,
                'mask_dir': args.mask_dir,
                'results': results
            }, f, indent=2)
        
        logger.info(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
