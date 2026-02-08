# SiT for Cell Segmentation

Fork of [SiT (Scalable Interpolant Transformers)](https://github.com/willisma/SiT) modified for mask-conditioned image generation for cell segmentation.

## Key Modifications

- **Mask Conditioning**: Instead of class labels, the model is conditioned on segmentation probability maps `(B, num_classes, H, W)`
- **Per-patch Conditioning**: Each patch receives its own condition embedding based on the mask content at that spatial location
- **Binary Segmentation**: Designed for 2-class segmentation (background + cell), but supports any number of classes

## Data Format

```
data/
├── images/           # Cell images (.png, .jpg, .tif, etc.)
│   ├── img_001.png
│   ├── img_002.png
│   └── ...
└── masks/            # Binary masks (black=background, white=cell)
    ├── img_001.png   # Same filenames as images
    ├── img_002.png
    └── ...
```

- Images: RGB cell microscopy images
- Masks: Grayscale images where black (0) = background, white (255) = cell

## Installation

```bash
conda env create -f environment.yml
conda activate SiT
```

## Training

```bash
torchrun --nproc_per_node=N train.py \
    --data-path /path/to/images \
    --mask-path /path/to/masks \
    --image-size 256 \
    --num-classes 2 \
    --model SiT-B/2 \
    --epochs 100 \
    --global-batch-size 64 \
    --wandb
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-path` | required | Path to image directory |
| `--mask-path` | required | Path to mask directory |
| `--image-size` | 256 | Target image size (must be divisible by 8) |
| `--num-classes` | 2 | Number of segmentation classes |
| `--model` | SiT-B/2 | Model architecture |
| `--cfg-scale` | 4.0 | Classifier-free guidance scale |
| `--wandb` | flag | Enable wandb logging |

## Sampling

### Single GPU

```bash
python sample.py ODE \
    --ckpt /path/to/checkpoint.pt \
    --mask-path /path/to/condition_mask.png \
    --num-classes 2 \
    --image-size 256 \
    --num-samples 4 \
    --output samples.png
```

### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=N sample_ddp.py ODE \
    --ckpt /path/to/checkpoint.pt \
    --data-path /path/to/images \
    --mask-path /path/to/masks \
    --num-classes 2 \
    --sample-dir samples/
```

## WandB Configuration

Edit `wandb_utils.py` to set your credentials:

```python
WANDB_ENTITY = "your-entity-name"
WANDB_PROJECT = "SiT-Seg"
WANDB_API_KEY = "your-api-key"
```

## Model Architecture

The model uses a **MaskEmbedder** that:
1. Takes a probability map `(B, num_classes, H, W)` as input
2. Downsamples to patch resolution using average pooling
3. Computes weighted sum of learnable class embeddings per patch
4. Returns per-patch condition embeddings `(B, T, D)`

This allows the model to generate images that respect the spatial structure defined by the input mask.

## Soft Mask Support

The model accepts soft probability masks (e.g., from softmax of logits), not just hard 0/1 masks. This enables:
- Smooth transitions at boundaries
- Uncertainty-aware generation
- Differentiable sampling pipelines

## License

See [LICENSE.txt](LICENSE.txt)
