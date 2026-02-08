"""
Test-Time Adaptation (TTA) Script

在测试时联合更新分割模型和SiT生成模型。
- 分割模型: eval模式，保留梯度
- SiT模型: eval模式，保留梯度
- Loss: MSE(预测速度, 真实速度)

Usage:
    python tta.py --config tta_config.yaml
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SiT_seg'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'train_segmentation'))

import segmentation_models_pytorch as smp
from diffusers.models import AutoencoderKL

# Import from SiT_seg
from SiT_seg.models import SiT_models
from SiT_seg.download import find_model

# Import metrics from train_segmentation
from train_segmentation.metrics import compute_all_metrics, MetricTracker


# =============================================================================
# Configuration
# =============================================================================

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(output_dir: str):
    """Setup logging to file and console."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'tta.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    return logging.getLogger(__name__)


# =============================================================================
# Dataset
# =============================================================================

class TTADataset(Dataset):
    """
    Dataset for TTA that provides:
    - RGB image for SiT (normalized to [-1, 1])
    - Grayscale image for segmentation model (normalized to [0, 1])
    - Optional GT mask for evaluation
    """
    
    def __init__(self, image_dir: str, mask_dir: str = None, image_size: int = 256):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.image_size = image_size
        
        # Find all image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        self.files = sorted([
            f.name for f in self.image_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ])
        
        if len(self.files) == 0:
            raise ValueError(f"No images found in {image_dir}")
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int):
        img_name = self.files[idx]
        img_path = self.image_dir / img_name
        
        # Load image
        image = Image.open(img_path)
        
        # === RGB image for SiT (BICUBIC resize, normalize to [-1, 1]) ===
        rgb_image = image.convert('RGB')
        rgb_image = rgb_image.resize((self.image_size, self.image_size), Image.BICUBIC)
        rgb_np = np.array(rgb_image, dtype=np.float32) / 255.0
        rgb_np = (rgb_np - 0.5) / 0.5  # Normalize to [-1, 1]
        rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1)  # (3, H, W)
        
        # === Grayscale image for segmentation (BILINEAR resize, normalize to [0, 1]) ===
        gray_image = image.convert('L')
        gray_image = gray_image.resize((self.image_size, self.image_size), Image.BILINEAR)
        gray_np = np.array(gray_image, dtype=np.float32) / 255.0
        gray_tensor = torch.from_numpy(gray_np).unsqueeze(0)  # (1, H, W)
        
        # === Optional GT mask for evaluation ===
        mask_tensor = None
        if self.mask_dir is not None:
            mask_name = Path(img_name).stem
            mask_path = None
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                candidate = self.mask_dir / f"{mask_name}{ext}"
                if candidate.exists():
                    mask_path = candidate
                    break
            
            if mask_path is None:
                mask_path = self.mask_dir / img_name
            
            if mask_path.exists():
                mask = Image.open(mask_path).convert('L')
                # Use BOX (AREA equivalent) for mask downsampling
                mask = mask.resize((self.image_size, self.image_size), Image.Resampling.BOX)
                mask_np = np.array(mask, dtype=np.float32) / 255.0
                mask_np = (mask_np > 0.5).astype(np.int64)
                mask_tensor = torch.from_numpy(mask_np)
        
        return rgb_tensor, gray_tensor, mask_tensor, img_name


def collate_fn(batch):
    """Custom collate function to handle optional masks."""
    rgb_tensors = torch.stack([item[0] for item in batch])
    gray_tensors = torch.stack([item[1] for item in batch])
    
    # Handle optional masks
    if batch[0][2] is not None:
        mask_tensors = torch.stack([item[2] for item in batch])
    else:
        mask_tensors = None
    
    names = [item[3] for item in batch]
    
    return rgb_tensors, gray_tensors, mask_tensors, names


# =============================================================================
# Models
# =============================================================================

def get_segmentation_model(in_channels: int = 1, num_classes: int = 2) -> nn.Module:
    """Create UNet segmentation model."""
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=in_channels,
        classes=num_classes,
        encoder_depth=5,
        decoder_channels=(256, 128, 64, 32, 16),
    )
    return model


def load_segmentation_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device):
    """Load segmentation model checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def get_sit_model(model_name: str, num_classes: int, image_size: int) -> nn.Module:
    """Create SiT model."""
    latent_size = image_size // 8
    model = SiT_models[model_name](
        input_size=latent_size,
        num_classes=num_classes,
        learn_sigma=False,  # We only need velocity prediction
    )
    return model


def load_sit_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device):
    """Load SiT model checkpoint."""
    state_dict = find_model(ckpt_path)
    if "model" in state_dict:
        model.load_state_dict(state_dict["model"])
    elif "ema" in state_dict:
        model.load_state_dict(state_dict["ema"])
    else:
        model.load_state_dict(state_dict)
    return model


# =============================================================================
# Linear Noise Schedule (from transport/path.py ICPlan)
# =============================================================================

def add_noise_linear(x1: torch.Tensor, t: float, noise: torch.Tensor = None):
    """
    Apply linear noise schedule.
    
    Linear path: xt = alpha_t * x1 + sigma_t * noise
    where alpha_t = t, sigma_t = 1 - t
    
    Args:
        x1: Clean latent (B, C, H, W)
        t: Timestep in (0, 1)
        noise: Optional noise tensor, if None will be sampled
    
    Returns:
        xt: Noised latent
        noise: The noise used
        ut: True velocity = d_alpha * x1 + d_sigma * noise = x1 - noise
    """
    if noise is None:
        noise = torch.randn_like(x1)
    
    alpha_t = t
    sigma_t = 1 - t
    
    xt = alpha_t * x1 + sigma_t * noise
    
    # True velocity: d_alpha = 1, d_sigma = -1
    ut = x1 - noise
    
    return xt, noise, ut


# =============================================================================
# TTA Training Loop
# =============================================================================

def run_tta(config: dict, logger: logging.Logger):
    """Main TTA training loop."""
    
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    image_size = config['image_size']
    latent_size = image_size // 8
    
    # Check if we have GT masks for metrics
    has_gt_masks = config['data'].get('mask_dir') is not None
    
    # =========================================================================
    # Load Models
    # =========================================================================
    
    logger.info("Loading segmentation model...")
    seg_model = get_segmentation_model(
        in_channels=config['segmentation']['in_channels'],
        num_classes=config['segmentation']['num_classes']
    ).to(device)
    seg_model = load_segmentation_checkpoint(
        seg_model, config['segmentation']['ckpt_path'], device
    )
    
    logger.info("Loading SiT model...")
    sit_model = get_sit_model(
        model_name=config['sit']['model'],
        num_classes=config['sit']['num_classes'],
        image_size=image_size
    ).to(device)
    sit_model = load_sit_checkpoint(
        sit_model, config['sit']['ckpt_path'], device
    )
    
    logger.info("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        f"stabilityai/sd-vae-ft-{config['sit']['vae']}"
    ).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False  # VAE is frozen
    
    # =========================================================================
    # Set models to eval mode but ENABLE gradients
    # This is crucial for TTA - we want eval behavior (no dropout, BN in eval)
    # but still compute and backprop gradients
    # =========================================================================
    seg_model.eval()
    sit_model.eval()
    
    # Enable gradients for both models (this is the key for TTA)
    for p in seg_model.parameters():
        p.requires_grad = True
    for p in sit_model.parameters():
        p.requires_grad = True
    
    logger.info(f"Segmentation model params: {sum(p.numel() for p in seg_model.parameters()):,}")
    logger.info(f"SiT model params: {sum(p.numel() for p in sit_model.parameters()):,}")
    logger.info("Both models in eval() mode with requires_grad=True for TTA")
    
    # =========================================================================
    # Setup Optimizer
    # =========================================================================
    
    tta_config = config['tta']
    base_lr = tta_config['learning_rate']
    
    # Create parameter groups with optional LR scaling
    param_groups = [
        {
            'params': seg_model.parameters(),
            'lr': base_lr * tta_config.get('seg_lr_scale', 1.0),
            'name': 'segmentation'
        },
        {
            'params': sit_model.parameters(),
            'lr': base_lr * tta_config.get('sit_lr_scale', 1.0),
            'name': 'sit'
        }
    ]
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0)
    
    # =========================================================================
    # Setup Dataset
    # =========================================================================
    
    dataset = TTADataset(
        image_dir=config['data']['image_dir'],
        mask_dir=config['data'].get('mask_dir'),
        image_size=image_size
    )
    
    loader = DataLoader(
        dataset,
        batch_size=tta_config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Batch size: {tta_config['batch_size']}")
    logger.info(f"Gradient accumulation steps: {tta_config['gradient_accumulation_steps']}")
    logger.info(f"Has GT masks for metrics: {has_gt_masks}")
    
    # =========================================================================
    # TTA Loop
    # =========================================================================
    
    timestep = tta_config['timestep']
    grad_accum_steps = tta_config['gradient_accumulation_steps']
    epochs = tta_config['epochs']
    output_dir = config['output']['dir']
    save_checkpoint_flag = config['output'].get('save_checkpoint', False)
    
    logger.info(f"Starting TTA for {epochs} epochs with timestep={timestep}")
    logger.info(f"Save checkpoint: {save_checkpoint_flag}")
    
    global_step = 0
    
    for epoch in range(epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        logger.info(f"{'='*60}")
        
        # Reset metric tracker for each epoch
        metric_tracker = MetricTracker()
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        
        # Create progress bar with dynamic postfix for metrics
        pbar = tqdm(loader, desc=f'Epoch {epoch + 1}', ncols=120)
        
        for batch_idx, (rgb_images, gray_images, gt_masks, names) in enumerate(pbar):
            rgb_images = rgb_images.to(device)  # (B, 3, H, W), [-1, 1]
            gray_images = gray_images.to(device)  # (B, 1, H, W), [0, 1]
            
            B = rgb_images.shape[0]
            
            # -----------------------------------------------------------------
            # Step 1: Segmentation forward (grayscale input)
            # Gradient flows through this!
            # -----------------------------------------------------------------
            logits = seg_model(gray_images)  # (B, num_classes, H, W)
            probs = F.softmax(logits, dim=1)  # Probability map - gradient flows!
            
            # Get predictions for metrics (detach for metric computation)
            preds = logits.argmax(dim=1).detach().cpu().numpy()  # (B, H, W)
            
            # -----------------------------------------------------------------
            # Step 2: Encode RGB to latent (no grad for VAE)
            # -----------------------------------------------------------------
            with torch.no_grad():
                latent = vae.encode(rgb_images).latent_dist.sample() * 0.18215  # x1
            
            # -----------------------------------------------------------------
            # Step 3: Add noise (linear schedule)
            # -----------------------------------------------------------------
            noise = torch.randn_like(latent)
            xt, _, ut = add_noise_linear(latent, timestep, noise)
            
            # -----------------------------------------------------------------
            # Step 4: Downsample probability map to latent size
            # IMPORTANT: This operation preserves gradients!
            # -----------------------------------------------------------------
            probs_latent = F.interpolate(probs, size=(latent_size, latent_size), mode='area')
            
            # -----------------------------------------------------------------
            # Step 5: SiT forward (predict velocity)
            # probs_latent carries gradients from seg_model
            # -----------------------------------------------------------------
            t_tensor = torch.full((B,), timestep, device=device)
            v_pred = sit_model(xt, t_tensor, mask=probs_latent)
            
            # -----------------------------------------------------------------
            # Step 6: Compute MSE loss
            # -----------------------------------------------------------------
            loss = F.mse_loss(v_pred, ut)
            loss_for_backward = loss / grad_accum_steps  # Normalize for gradient accumulation
            
            # -----------------------------------------------------------------
            # Step 7: Backward - gradients flow to BOTH models!
            # - v_pred -> sit_model params
            # - probs_latent -> probs -> logits -> seg_model params
            # -----------------------------------------------------------------
            loss_for_backward.backward()
            
            epoch_loss_sum += loss.item()
            epoch_loss_count += 1
            
            # -----------------------------------------------------------------
            # Step 8: Compute segmentation metrics (if GT available)
            # -----------------------------------------------------------------
            if has_gt_masks and gt_masks is not None:
                targets = gt_masks.numpy()  # (B, H, W)
                for pred, target in zip(preds, targets):
                    metrics = compute_all_metrics(pred, target)
                    metric_tracker.update(metrics)
                
                # Update progress bar with current average metrics
                avg_metrics = metric_tracker.get_average()
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'dice': f'{avg_metrics.get("dice", 0):.3f}',
                    'hd95': f'{avg_metrics.get("hd95", 0):.1f}',
                    'jac': f'{avg_metrics.get("jaccard", 0):.3f}',
                    'sen': f'{avg_metrics.get("sensitivity", 0):.3f}',
                    'ppv': f'{avg_metrics.get("ppv", 0):.3f}',
                })
            else:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # -----------------------------------------------------------------
            # Step 9: Optimizer step (with gradient accumulation)
            # -----------------------------------------------------------------
            if (batch_idx + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
        
        # Handle remaining gradients at end of epoch
        if (batch_idx + 1) % grad_accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # -----------------------------------------------------------------
        # Epoch Summary
        # -----------------------------------------------------------------
        avg_epoch_loss = epoch_loss_sum / epoch_loss_count if epoch_loss_count > 0 else 0
        
        logger.info(f"\nEpoch {epoch + 1} Summary:")
        logger.info(f"  Average Loss: {avg_epoch_loss:.6f}")
        
        if has_gt_masks:
            avg_metrics = metric_tracker.get_average()
            logger.info(f"  Metrics:")
            logger.info(f"    DICE:        {avg_metrics.get('dice', 0):.4f}")
            logger.info(f"    HD95:        {avg_metrics.get('hd95', 0):.2f}")
            logger.info(f"    Jaccard:     {avg_metrics.get('jaccard', 0):.4f}")
            logger.info(f"    Sensitivity: {avg_metrics.get('sensitivity', 0):.4f}")
            logger.info(f"    PPV:         {avg_metrics.get('ppv', 0):.4f}")
    
    # =========================================================================
    # Save final checkpoint (optional)
    # =========================================================================
    if save_checkpoint_flag:
        save_checkpoint(
            seg_model, sit_model, optimizer, global_step, epochs,
            output_dir, logger, is_final=True
        )
    
    logger.info("\nTTA completed!")


def save_checkpoint(seg_model, sit_model, optimizer, step, epoch, output_dir, logger, is_final=False):
    """Save model checkpoints."""
    os.makedirs(output_dir, exist_ok=True)
    
    suffix = "final" if is_final else f"step_{step:07d}"
    
    # Save segmentation model
    seg_path = os.path.join(output_dir, f"seg_model_{suffix}.pth")
    torch.save({
        'model_state_dict': seg_model.state_dict(),
        'epoch': epoch,
        'step': step
    }, seg_path)
    
    # Save SiT model
    sit_path = os.path.join(output_dir, f"sit_model_{suffix}.pt")
    torch.save({
        'model': sit_model.state_dict(),
        'epoch': epoch,
        'step': step
    }, sit_path)
    
    logger.info(f"Saved checkpoints: {seg_path}, {sit_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Test-Time Adaptation (TTA)')
    parser.add_argument('--config', type=str, default='tta_config.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config['output']['dir']
    if not output_dir:
        output_dir = f"outputs/tta/{timestamp}"
    config['output']['dir'] = output_dir
    
    logger = setup_logging(output_dir)
    
    # Log config
    logger.info("Configuration:")
    logger.info(yaml.dump(config, default_flow_style=False))
    
    # Save config to output dir
    config_save_path = os.path.join(output_dir, 'config.yaml')
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Run TTA
    run_tta(config, logger)


if __name__ == '__main__':
    main()
