                                                                                                                                           
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
from xml.parsers.expat import model

import torch

from diffusion.gaussian_diffusion import _extract_into_tensor
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import math
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os 
from models import DiT_models
from diffusion import create_diffusion


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

class PDEDataset(Dataset):
    """
    Expected per-sample dict (either inside a packed file or one file per sample):
      "x_cond": (N_nodes, Cx)
      "y":      (N_nodes, Cy)
      "coords": (N_nodes, d)
      "phys":   (P,) optional
    Packed-file format:
      obj["x_cond"]: (N_samples, N_nodes, Cx)
      obj["y"]:      (N_samples, N_nodes, Cy)
      obj["coords"]: (N_samples, N_nodes, d)
      obj["phys"]:   (N_samples, P) optional
    """
    def __init__(self, path: str):
        super().__init__()
        self.path = path

        if os.path.isdir(path):
            self.mode = "dir"
            self.items = sorted(
                os.path.join(path, f) for f in os.listdir(path) if f.endswith(".pt")
            )
            if len(self.items) == 0:
                raise FileNotFoundError(f"No .pt files found in directory: {path}")
            return

        if os.path.isfile(path) and path.endswith(".pt"):
            self.mode = "file"
            obj = torch.load(path, map_location="cpu")
            if not isinstance(obj, dict):
                raise TypeError(f"Expected a dict in {path}, got {type(obj)}")

            def pick(d, keys):
                for k in keys:
                    if k in d and d[k] is not None:
                        return d[k]
                return None

            self.x = pick(obj, ["x_cond", "x"])
            self.y = pick(obj, ["y", "u"])
            self.coords = pick(obj, ["coords", "xy", "xyz"])
            self.phys = obj.get("phys", None)

            if self.x is None or self.y is None or self.coords is None:
                raise KeyError(
                    f"Missing required tensors in {path}. Need x_cond/x, y/u, coords. Keys: {list(obj.keys())}"
                )

            if self.x.shape[0] != self.y.shape[0] or self.x.shape[0] != self.coords.shape[0]:
                raise ValueError(
                    f"N_samples mismatch: x {self.x.shape[0]}, y {self.y.shape[0]}, coords {self.coords.shape[0]}"
                )

            # fixed-node-count assumption for this version
            if self.x.dim() != 3 or self.y.dim() != 3 or self.coords.dim() != 3:
                raise ValueError(
                    f"Expected packed tensors with shape (N_samples, N_nodes, C). "
                    f"Got x {tuple(self.x.shape)}, y {tuple(self.y.shape)}, coords {tuple(self.coords.shape)}"
                )

            self.N = self.x.shape[0]
            return

        raise FileNotFoundError(f"Not a valid directory or .pt file: {path}")

    def __len__(self):
        return len(self.items) if self.mode == "dir" else self.N

    def __getitem__(self, idx):
        if self.mode == "dir":
            sample = torch.load(self.items[idx], map_location="cpu")
            x_cond = sample["x_cond"].float()      # (N, Cx)
            y      = sample["y"].float()           # (N, Cy)
            coords = sample["coords"].float()      # (N, d)
            phys   = sample.get("phys", None)
            phys = phys.float() if phys is not None else torch.empty(0)
            return x_cond, y, phys, coords

        # mode == "file"
        x_cond = self.x[idx].float()              # (N, Cx)
        y      = self.y[idx].float()              # (N, Cy)
        coords = self.coords[idx].float()         # (N, d)
        phys = None
        if self.phys is not None:
            phys = self.phys[idx].float()
        else:
            phys = torch.empty(0)
        return x_cond, y, phys, coords

@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    """
    Step the Exponential Moving Average model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training (destroys process group)
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout only on rank 0.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger




@torch.no_grad()
def multiscale_noise_nodes(
    y: torch.Tensor,          # (B, N, C) only used for shape/device/dtype
    coords: torch.Tensor,     # (B, N, d) node coordinates
    k: int = 16,              # kNN degree
    sigmas=(0.02, 0.05, 0.10, 0.20),  # spatial scales in coordinate units
    alpha: float = 0.5,       # geometric weight per scale
    iters: int = 1,           # repeat smoothing (increases low-pass strength)
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Multiscale correlated Gaussian noise on node sets using a kNN heat-kernel filter.

    Returns:
      E: (B, N, C) approximately unit-std per channel (global)
    """
    B, N, C = y.shape
    device = y.device
    dtype = y.dtype

    coords = coords.to(device=device, dtype=dtype)

    # Pairwise squared distances (B, N, N)
    # For large N this is O(N^2). Works if N is a few thousand or less.
    d2 = torch.cdist(coords, coords, p=2) ** 2

    # kNN indices per node (exclude self by taking k+1 then dropping the first)
    knn_idx = torch.topk(d2, k=k+1, dim=-1, largest=False).indices  # (B,N,k+1)
    knn_idx = knn_idx[:, :, 1:]                                     # (B,N,k)

    # gather neighbor distances: (B,N,k)
    d2_knn = d2.gather(-1, knn_idx)

    # base full-res iid noise
    E = torch.randn((B, N, C), device=device, dtype=dtype)
    var = 1.0

    # helper: one smoothing pass with heat weights
    def smooth(x, sigma):
        # weights: exp(-d^2 / (2 sigma^2))
        w = torch.exp(-d2_knn / (2.0 * (sigma * sigma) + eps))      # (B,N,k)
        w = w / (w.sum(dim=-1, keepdim=True) + eps)                 # normalize (B,N,k)

        # neighbor values: x_neighbors (B,N,k,C)
        x_nb = x.gather(1, knn_idx.unsqueeze(-1).expand(B, N, k, C))
        # weighted sum over neighbors -> (B,N,C)
        return (w.unsqueeze(-1) * x_nb).sum(dim=2)

    # multiscale sum: progressively smoother fields
    for i, sigma in enumerate(sigmas, start=1):
        x = torch.randn((B, N, C), device=device, dtype=dtype)
        for _ in range(iters):
            x = smooth(x, sigma)
        w_i = alpha ** i
        E = E + w_i * x
        var += w_i * w_i

    # normalize to approx unit std
    E = E / (E.std(dim=(1,2), keepdim=True) + 1e-8)

    return E



@torch.no_grad()
def log_x0_recon_stats(diffusion, y, y_t, t, eps_hat, logger, prefix="x0"):
    # compute x0_hat the same way the diffusion code does for eps-pred
    ab = _extract_into_tensor(diffusion.alphas_cumprod, t, y.shape)
    x0_hat = (y_t - (1.0 - ab).sqrt() * eps_hat) / (ab.sqrt() + 1e-8)

    err = x0_hat - y

    # core calibration signals (these match your sampling complaints)
    logger.info(
        f"{prefix}: y std {y.std().item():.4f} | x0_hat std {x0_hat.std().item():.4f} | "
        f"bias {err.mean().item():.4f} | err std {err.std().item():.4f} | "
        f"err min/max {err.min().item():.3f}/{err.max().item():.3f}"
    )

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    model = DiT_models[args.model](
        coord_dim=args.coord_dim,
        in_channels=args.cx + args.cy,
        learn_sigma=False,
        pos_mode=args.pos_mode,        # "none" | "coord_mlp" | "rff"
        n_phys_params=args.phys_dim,
        # rff_scale=... if using rff
    )
    model.set_out_channels(cy=args.cy)

    
    # Parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training   
    requires_grad(ema, False)
    gpu_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(gpu_id)
    device = torch.device("cuda", gpu_id)
    model = DDP(model.to(device), device_ids=[gpu_id])
    ema = ema.to(device)
    diffusion = create_diffusion(timestep_respacing="",learn_sigma=False)  # default: 1000 steps, linear noise schedule
    for name in ["model_mean_type", "model_var_type", "loss_type", "rescale_timesteps"]:
        if hasattr(diffusion, name):
            print(name, "=", getattr(diffusion, name))
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4): 
    # *Look at adaptive lr*
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:
    dataset = PDEDataset(args.data_path)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=pde_collate, 
    )
    logger.info(f"Dataset contains {len(dataset):,} PDE samples ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x_cond, y, phys, coords in loader:
            x_cond = x_cond.to(device)     # (B, N, Cx)
            y      = y.to(device)          # (B, N, Cy)
            coords = coords.to(device)     # (B, N, d)

            phys = phys.to(device).float()
            if phys.numel() == 0:
                phys = None

            B = y.shape[0]
            t = torch.randint(0, diffusion.num_timesteps, (B,), device=device, dtype=torch.long)

            # IID Gaussian noise for node targets
            eps = multiscale_noise_nodes(y, coords)      

            # alpha_bar broadcast to (B, N, Cy)
            alpha_bar = _extract_into_tensor(diffusion.alphas_cumprod, t, y.shape)  # (B,N,Cy)

            y_t = alpha_bar.sqrt() * y + (1.0 - alpha_bar).sqrt() * eps            # (B,N,Cy)

            # concat along feature dim (last)
            s = torch.cat([x_cond, y_t], dim=-1)   # (B, N, Cx+Cy)

            # forward: predict eps on nodes
            eps_hat = model(s, t, phys_params=phys, coords=coords)  # (B,N,Cy) (or (B,N,2*Cy) if learn_sigma)

            loss_l2 = F.mse_loss(eps_hat, eps)
            loss_l1 = F.l1_loss(eps_hat, eps)
            loss = loss_l2 + loss_l1

            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module, decay=0.999)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()
            if epoch == 0 and train_steps == 0 and rank == 0:
                print("x_cond mean/std:", x_cond.mean().item(), x_cond.std().item())
                print("y mean/std:", y.mean().item(), y.std().item())
            if train_steps % args.log_every == 0:
                # force evaluation at small t where x0 is well-conditioned
                t_small = torch.randint(0, 50, (y.shape[0],), device=y.device)  # adjust 50
                eps_eval = multiscale_noise_nodes(y, coords)
                ab = _extract_into_tensor(diffusion.alphas_cumprod, t_small, y.shape)
                y_t_small = ab.sqrt() * y + (1.0 - ab).sqrt() * eps_eval

                s_small = torch.cat([x_cond, y_t_small], dim=-1)
                eps_hat_small = model(s_small, t_small, phys_params=phys, coords=coords)
                log_x0_recon_stats(diffusion, y, y_t_small, t_small, eps_hat_small, logger, prefix="x0@t<50")

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL")
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--cx", type=int, required=True, help="conditioning channels Cx")
    parser.add_argument("--cy", type=int, required=True, help="target channels Cy")
    parser.add_argument("--phys-dim", type=int, default=0, help="number of global physical parameters (P)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--coord-dim", type=int, required=True)  # 2 or 3
    parser.add_argument("--pos-mode", type=str, choices=["none", "coord_mlp", "rff"], default="none")
    args = parser.parse_args()
    main(args)
