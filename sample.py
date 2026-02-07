# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained SiT model with mask conditioning.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from download import find_model
from models import SiT_models
from train_utils import parse_ode_args, parse_sde_args, parse_transport_args
from transport import create_transport, Sampler
from dataset import load_mask_for_sampling, create_null_mask
import torch.nn.functional as F
import argparse
import sys
from time import time


def main(mode, args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    latent_size = args.image_size // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        learn_sigma=args.learn_sigma,
    ).to(device)
    
    # Load checkpoint
    if args.ckpt is None:
        raise ValueError("Must provide --ckpt for mask-conditioned sampling")
    
    state_dict = find_model(args.ckpt)
    if "model" in state_dict:
        model.load_state_dict(state_dict["model"])
    else:
        model.load_state_dict(state_dict)
    model.eval()
    
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )
    sampler = Sampler(transport)
    
    if mode == "ODE":
        if args.likelihood:
            assert args.cfg_scale == 1, "Likelihood is incompatible with guidance"
            sample_fn = sampler.sample_ode_likelihood(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
            )
        else:
            sample_fn = sampler.sample_ode(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
                reverse=args.reverse
            )
            
    elif mode == "SDE":
        sample_fn = sampler.sample_sde(
            sampling_method=args.sampling_method,
            diffusion_form=args.diffusion_form,
            diffusion_norm=args.diffusion_norm,
            last_step=args.last_step,
            last_step_size=args.last_step_size,
            num_steps=args.num_sampling_steps,
        )
    

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Load conditioning mask:
    mask = load_mask_for_sampling(args.mask_path, args.image_size, args.num_classes, device)
    n = args.num_samples
    
    # Repeat mask for batch size
    if n > 1:
        mask = mask.repeat(n, 1, 1, 1)
    
    # Downsample mask to latent size
    mask_latent = F.interpolate(mask, size=(latent_size, latent_size), mode='area')
    
    # Create sampling noise:
    z = torch.randn(n, 4, latent_size, latent_size, device=device)

    # Setup classifier-free guidance:
    use_cfg = args.cfg_scale > 1.0
    if use_cfg:
        z = torch.cat([z, z], 0)
        null_mask = create_null_mask(n, args.num_classes, latent_size, latent_size, device)
        mask_cfg = torch.cat([mask_latent, null_mask], 0)
        model_kwargs = dict(mask=mask_cfg, cfg_scale=args.cfg_scale)
        model_fn = model.forward_with_cfg
    else:
        model_kwargs = dict(mask=mask_latent)
        model_fn = model.forward

    # Sample images:
    start_time = time()
    samples = sample_fn(z, model_fn, **model_kwargs)[-1]
    if use_cfg:
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample
    print(f"Sampling took {time() - start_time:.2f} seconds.")

    # Save and display images:
    save_image(samples, args.output, nrow=4, normalize=True, value_range=(-1, 1))
    print(f"Saved samples to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    if len(sys.argv) < 2:
        print("Usage: program.py <mode> [options]")
        sys.exit(1)
    
    mode = sys.argv[1]

    assert mode[:2] != "--", "Usage: program.py <mode> [options]"
    assert mode in ["ODE", "SDE"], "Invalid mode. Please choose 'ODE' or 'SDE'"
    
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-B/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=2, help="Number of segmentation classes")
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to a SiT checkpoint")
    parser.add_argument("--mask-path", type=str, required=True,
                        help="Path to conditioning mask image (black=background, white=cell)")
    parser.add_argument("--output", type=str, default="sample.png",
                        help="Output path for generated samples")
    parser.add_argument("--num-samples", type=int, default=4,
                        help="Number of samples to generate")
    parser.add_argument("--learn-sigma", action="store_true",
                        help="Whether the model was trained with learn_sigma=True")

    parse_transport_args(parser)
    if mode == "ODE":
        parse_ode_args(parser)
        # Further processing for ODE
    elif mode == "SDE":
        parse_sde_args(parser)
        # Further processing for SDE
    
    args = parser.parse_known_args()[0]
    main(mode, args)
