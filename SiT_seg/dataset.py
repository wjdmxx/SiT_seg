# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Dataset for cell segmentation with image-mask pairs.
"""

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


class CellSegmentationDataset(Dataset):
    """
    Dataset for loading cell segmentation image-mask pairs.
    
    Args:
        image_dir: Path to directory containing cell images
        mask_dir: Path to directory containing binary masks (black=background, white=cell)
        image_size: Target size for resizing (images will be resized to image_size x image_size)
        num_classes: Number of segmentation classes (default: 2 for background + cell)
    """
    
    def __init__(self, image_dir, mask_dir, image_size, num_classes=2):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.num_classes = num_classes
        
        # Get list of image files
        self.image_files = sorted([
            f for f in os.listdir(image_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))
        ])
        
        # Verify masks exist for all images
        self.mask_files = []
        for img_file in self.image_files:
            # Try to find matching mask with same name (possibly different extension)
            img_name = os.path.splitext(img_file)[0]
            mask_found = False
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
                mask_path = os.path.join(mask_dir, img_name + ext)
                if os.path.exists(mask_path):
                    self.mask_files.append(img_name + ext)
                    mask_found = True
                    break
            if not mask_found:
                raise FileNotFoundError(f"No mask found for image: {img_file}")
        
        # Image transform: resize with BICUBIC, normalize to [-1, 1]
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        # Load mask
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale
        
        # Apply image transform
        image = self.image_transform(image)
        
        # Resize mask using AREA interpolation to avoid aliasing
        # First resize as tensor for AREA interpolation
        mask_np = np.array(mask).astype(np.float32) / 255.0  # Normalize to 0-1
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        # Use AREA interpolation for downsampling
        mask_resized = torch.nn.functional.interpolate(
            mask_tensor, 
            size=(self.image_size, self.image_size), 
            mode='area'
        ).squeeze(0)  # (1, H, W)
        
        # Convert to one-hot / probability format: (num_classes, H, W)
        # mask_resized is now in [0, 1] where 0=background, 1=cell
        cell_prob = mask_resized  # (1, H, W) probability of being cell
        background_prob = 1.0 - cell_prob  # (1, H, W) probability of being background
        
        # Stack to form (num_classes, H, W) where class 0=background, class 1=cell
        mask_prob = torch.cat([background_prob, cell_prob], dim=0)  # (2, H, W)
        
        return image, mask_prob


def load_mask_for_sampling(mask_path, image_size, num_classes=2, device='cpu'):
    """
    Load a single mask for sampling/inference.
    
    Args:
        mask_path: Path to the mask image
        image_size: Target size to resize mask to
        num_classes: Number of classes
        device: Device to load tensor to
        
    Returns:
        mask_prob: (1, num_classes, H, W) probability tensor
    """
    mask = Image.open(mask_path).convert('L')
    mask_np = np.array(mask).astype(np.float32) / 255.0
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    # AREA interpolation
    mask_resized = torch.nn.functional.interpolate(
        mask_tensor,
        size=(image_size, image_size),
        mode='area'
    )  # (1, 1, H, W)
    
    # Convert to probability format
    cell_prob = mask_resized  # (1, 1, H, W)
    background_prob = 1.0 - cell_prob
    mask_prob = torch.cat([background_prob, cell_prob], dim=1)  # (1, 2, H, W)
    
    return mask_prob.to(device)


def create_null_mask(batch_size, num_classes, height, width, device='cpu'):
    """
    Create a null mask for CFG (classifier-free guidance).
    Uses full background (class 0 = 1.0, other classes = 0.0).
    
    Args:
        batch_size: Number of samples
        num_classes: Number of segmentation classes
        height: Mask height
        width: Mask width
        device: Device to create tensor on
        
    Returns:
        null_mask: (B, num_classes, H, W) tensor with full background
    """
    null_mask = torch.zeros(batch_size, num_classes, height, width, device=device)
    null_mask[:, 0, :, :] = 1.0  # Set background probability to 1
    return null_mask
