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
from collections import OrderedDict
from train import pde_collate, PDEDataset, multiscale_noise
from diffusion import create_diffusion
from models import DiT_models
from diffusion.gaussian_diffusion import _extract_into_tensor

def ensure_4d(x):
    # (B,H,W) to (B,1,H,W)
    if x is None:
        return None
    if x.dim() == 3:
        return x.unsqueeze(1)
    return x


def _strip_prefix(state_dict, prefixes=("module.", "_orig_mod.")):
    if not isinstance(state_dict, dict):
        return state_dict
    out = OrderedDict()
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out

def load_ckpt(path: str, prefer_ema: bool = True):
    """
    Returns:
      state_dict: cleaned model weights dict (no module/_orig_mod prefixes)
      train_args: ckpt['args'] if present else None
      which:      string describing what was loaded ('ema'|'model'|'state_dict'|'raw')
      ckpt:       full checkpoint dict (for debugging)
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ta = ckpt["args"]
    print("CKPT:", args.ckpt)
    print("ckpt args:", ta.model, ta.image_size, ta.cx, ta.cy, ta.pos_mode, "phys_dim", getattr(ta,"phys_dim",0))
    train_args = ckpt.get("args", None) if isinstance(ckpt, dict) else None

    which = "raw"
    state = None

    if isinstance(ckpt, dict):
        # pick weights deterministically
        if prefer_ema and isinstance(ckpt.get("ema", None), dict):
            state = ckpt["ema"]; which = "ema"
        elif isinstance(ckpt.get("model", None), dict):
            state = ckpt["model"]; which = "model"
        elif isinstance(ckpt.get("state_dict", None), dict):
            state = ckpt["state_dict"]; which = "state_dict"
        else:
            # sometimes ckpt itself is a state_dict
            # detect by checking for tensor values
            is_sd = any(torch.is_tensor(v) for v in ckpt.values()) if len(ckpt) else False
            if is_sd:
                state = ckpt; which = "raw_state_dict"
            else:
                raise ValueError(f"Checkpoint dict has no weights keys. Keys: {list(ckpt.keys())}")

    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

    state = _strip_prefix(state)
    return state, train_args, which, ckpt



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
    
    # load model
    model = DiT_models[args.model](
        input_size=args.H,
        in_channels=args.cx + args.cy,
        learn_sigma=False,
        pos_mode=args.pos_mode,
 
    ).cpu()  

    model.set_out_channels(cy=args.cy)  

    state_dict, train_args, which, ckpt = load_ckpt(args.ckpt, prefer_ema=True)
    print("loaded:", which)
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device).eval() 


    diffusion = create_diffusion(timestep_respacing=str(args.num_sampling_steps), learn_sigma=False)
    for name in ["model_mean_type", "model_var_type", "loss_type", "rescale_timesteps"]:
        if hasattr(diffusion, name):
            print(name, "=", getattr(diffusion, name))
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
        if epoch == 0 and rank == 0:
            print("x_cond mean/std:", x_cond.mean().item(), x_cond.std().item())
            print("y mean/std:", y.mean().item(), y.std().item())
        #test one step of the model to check it is working and matches training signature
        from diffusion.gaussian_diffusion import _extract_into_tensor

        if rank == 0 and (_ % 8 == 0):
            B = y.shape[0]
            t_test = torch.randint(0, diffusion.num_timesteps, (B,), device=device, dtype=torch.long)

            eps = multiscale_noise(y)
            ab  = _extract_into_tensor(diffusion.alphas_cumprod, t_test, y.shape)
            y_t = ab.sqrt() * y + (1.0 - ab).sqrt() * eps

            s = torch.cat([x_cond, y_t], dim=1)
            eps_hat = model(s, t_test, phys_params=None)

            cos = torch.nn.functional.cosine_similarity(eps_hat.flatten(1), eps.flatten(1), dim=1).mean()
            rel = (eps_hat - eps).pow(2).mean().sqrt() / (eps.pow(2).mean().sqrt() + 1e-8)
            ratio = eps_hat.std() / (eps.std() + 1e-8)

            print("ONE-STEP cos:", float(cos), "rel:", float(rel), "std_ratio:", float(ratio))

                    
          
        
        # Start from noise in target space (Cy,H,W)
        z = torch.randn((n, args.cy, args.H, args.W), device=device)   

        if phys is not None:
            phys = phys.to(device, non_blocking=True).float()
            if phys.numel() == 0:
                phys = None

        samples = diffusion.ddim_sample_loop(
            sample_fn,
            z.shape,
            noise=z,                    # multiscale init
            clip_denoised=False,
            model_kwargs={"x_cond": x_cond, "phys": phys},
            progress=False,
            device=device,
            eta = 0.3,
        )


        print("SAMP min/max/std:", float(samples.min()), float(samples.max()), float(samples.std()))
        print("TRUE min/max/std:", float(y.min()), float(y.max()), float(y.std()))


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
