# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import os, math, argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import numpy as np

from train import pde_collate, PDEDataset, multiscale_noise
from diffusion import create_diffusion
from models import DiT_models


def ensure_4d(x):
    # (B,H,W) to (B,1,H,W)
    if x is None:
        return None
    if x.dim() == 3:
        return x.unsqueeze(1)
    return x


def _load_train_ckpt(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if "ema" in ckpt and isinstance(ckpt["ema"], dict):
            return ckpt["ema"]
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        return ckpt
    return ckpt


def main(args):
    """
    Run sampling for PDE/CFD fields using a checkpoint produced by train.py.
    """
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU."
    torch.set_grad_enabled(False)

    # TF32 new API 
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32" if args.tf32 else "ieee"
        torch.backends.cudnn.conv.fp32_precision = "tf32" if args.tf32 else "ieee"
    except Exception:
        pass

    # Setup DDP (torchrun provides env vars)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)


    seed = args.global_seed * world + rank
    torch.manual_seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={world}.")

    if args.ckpt is None:
        raise ValueError("Please specify a checkpoint path with --ckpt")

    # Dataset + loader (conditioning batches)
    ds = PDEDataset(args.dataset_pt)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, seed=args.global_seed)
    loader = DataLoader(
        ds,
        batch_size=args.per_proc_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=pde_collate,
    )

    # Load model 
    model = DiT_models[args.model](
        input_size=args.H,
        in_channels=args.cx + args.cy,
        learn_sigma=False,
        pos_mode=args.pos_mode,
    ).to(device)

    state_dict = _load_train_ckpt(args.ckpt)

    model.set_out_channels(cy=args.cy)       
    state_dict = _load_train_ckpt(args.ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    diffusion = create_diffusion(timestep_respacing="", learn_sigma=False)

    # Output
    os.makedirs(args.sample_dir, exist_ok=True)
    dist.barrier()

    # Sampling counts
    n = args.per_proc_batch_size
    global_batch_size = n * world
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of fields that will be sampled: {total_samples}")

    per_rank = total_samples // world
    iterations = per_rank // n

    pbar = range(iterations)
    if rank == 0:
        pbar = tqdm(pbar, "Sampling fields")

    # Keep a persistent iterator
    epoch = 0
    data_iter = iter(loader)

    # Accumulate per-rank outputs
    x_list, y_true_list, y_samp_list = [], [], []

    # Model wrapper for diffusion sampling (predicts epsilon for y-channels)
    def sample_fn(x_t, t, x_cond=None, phys=None):
        s = torch.cat([x_cond, x_t], dim=1)
        if phys is None:
            return model(s, t)
        return model(s, t, phys_params=phys)

    for _ in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            data_iter = iter(loader)
            batch = next(data_iter)

        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x_cond, y, phys = batch
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            x_cond, y = batch
            phys = None
        else:
            raise RuntimeError("pde_collate must return (x_cond, y) or (x_cond, y, phys).")

        x_cond = ensure_4d(x_cond).to(device, non_blocking=True)
        y = ensure_4d(y).to(device, non_blocking=True)
        if phys is not None:
            phys = phys.to(device, non_blocking=True)
        
        # Enforce expected spatial size
        if x_cond.shape[-2:] != (args.H, args.W) or y.shape[-2:] != (args.H, args.W):
            raise ValueError(
                f"Batch spatial size mismatch: x_cond {tuple(x_cond.shape[-2:])}, y {tuple(y.shape[-2:])}, "
                f"expected {(args.H, args.W)}"
            )

        # Start from noise in target space (Cy,H,W)
        z0 = torch.zeros((n, args.cy, args.H, args.W), device=device)
        z  = multiscale_noise(z0)   

        if phys is not None:
            phys = phys.to(device, non_blocking=True).float()
            if phys.numel() == 0:
                phys = None

        model_kwargs = {"x_cond": x_cond, "phys": phys}
        samples = diffusion.ddim_sample_loop(
            sample_fn,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=device,
            eta=0.0,
        )


        x_list.append(x_cond.detach().cpu().float())
        y_true_list.append(y.detach().cpu().float())
        y_samp_list.append(samples.detach().cpu().float())
        


    # Truncate extras (due to divisibility padding)
    x_out = torch.cat(x_list, dim=0).numpy()[:per_rank]
    y_true_out = torch.cat(y_true_list, dim=0).numpy()[:per_rank]
    y_samp_out = torch.cat(y_samp_list, dim=0).numpy()[:per_rank]

    out_path = os.path.join(args.sample_dir, f"samples_rank{rank:04d}.npz")
    np.savez_compressed(
        out_path,
        x_cond=x_out,
        y_true=y_true_out,
        y_sample=y_samp_out,
        H=args.H,
        W=args.W,
        cx=args.cx,
        cy=args.cy,
        model=args.model,
        ckpt=args.ckpt,
        seed=seed,
    )
    

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)

    # explicit spatial size
    parser.add_argument("--H", type=int, required=True)
    parser.add_argument("--W", type=int, required=True)

    parser.add_argument("--cx", type=int, required=True)
    parser.add_argument("--cy", type=int, required=True)
    parser.add_argument("--pos-mode", type=str, default="learned")

    # Dataset and checkpoint
    parser.add_argument("--dataset-pt", dest="dataset_pt", type=str, required=True,
                        help="Path to the .pt dataset used by PDEDataset")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to a checkpoint produced by train.py ")

    parser.add_argument("--num-workers", type=int, default=2)

    args = parser.parse_args()
    main(args)
