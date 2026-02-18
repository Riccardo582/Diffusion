                                                                                                                                           
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
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
    def __init__(self, path):
        super().__init__()
        self.path = path

        # Directory of .pt files
        if os.path.isdir(path):
            self.mode = "dir"
            self.items = sorted(
                os.path.join(path, f) for f in os.listdir(path) if f.endswith(".pt")
            )
            if len(self.items) == 0:
                raise FileNotFoundError(f"No .pt files found in directory: {path}")
            return

        # Single packed .pt file
        if os.path.isfile(path) and path.endswith(".pt"):
            self.mode = "file"
            obj = torch.load(path, map_location="cpu")

            if not isinstance(obj, dict):
                raise TypeError(f"Expected a dict in {path}, got {type(obj)}")

            def pick(d, keys):
                for k in keys:
                    v = d.get(k, None)
                    if v is not None:
                        return v
                return None
            self.x = pick(obj,["x_cond","x","a"])
            self.y = pick(obj,["y","u"])
            self.phys = obj.get("phys", None)

            if self.x is None or self.y is None:
                raise KeyError(f"Could not find x/y tensors in {path}. Keys: {list(obj.keys())}")

            if self.x.shape[0] != self.y.shape[0]:
                raise ValueError(f"x and y must have same N. Got {self.x.shape[0]} vs {self.y.shape[0]}")

            self.N = self.x.shape[0]
            return

        raise FileNotFoundError(f"Not a valid directory or .pt file: {path}")

    def __len__(self):
        if self.mode == "dir":
            return len(self.items)
        return self.N

    def __getitem__(self, idx):
        if self.mode == "dir":
            sample = torch.load(self.items[idx], map_location="cpu")
            x_cond = sample["x_cond"].float()
            y = sample["y"].float()
            phys = sample.get("phys", None)
            if phys is not None:
                phys = phys.float()
            else:
                phys = torch.empty(0)
            return x_cond, y, phys

        # mode == "file"
        x_cond = self.x[idx].float()
        y = self.y[idx].float()
        phys = None
        if self.phys is not None:
            phys = self.phys[idx].float()
        return x_cond, y, phys


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



def multiscale_noise(y, k=4, alpha=0.5):
    """
    Multiscale Gaussian noise 
    y: (B,C,H,W) or (B,H,W) used only for shape/device
    returns: (B,C,H,W)
    """
    if y.dim() == 3:
        y = y.unsqueeze(1)
    B, C, H, W = y.shape
    device = y.device

    # base full-res noise (i=0)
    E = torch.randn(B, C, H, W, device=device)
    var = 1.0  

    h_i, w_i = H, W
    for i in range(1, k):
        h_i = max(2, H // (2**i))
        w_i = max(2, W // (2**i))
        if h_i < 2 or w_i < 2:
            break

        eps = torch.randn(B, C, h_i, w_i, device=device)
        eps = F.interpolate(eps, size=(H, W), mode="bilinear", align_corners=False)

        w = alpha ** i
        E = E + w * eps
        var += w * w

    # analytic normalization to unit per-pixel std (approximately)
    E = E / math.sqrt(var)
    return E


def gather_alpha_bar(diffusion,t,device):
    """
    Gather alpha_bar values for a batch of timesteps.
    """
    ab = diffusion.alphas_cumprod  

    # Convert once per call if needed and move to device
    if isinstance(ab, np.ndarray):
        ab = torch.from_numpy(ab).float().to(device)
    elif isinstance(ab, torch.Tensor):
        ab = ab.to(device)
    else:
        ab = torch.tensor(ab, dtype=torch.float32, device=device)

    t = t.long().to(device)        # indices must be int
    out = ab.gather(0, t)          # (B,)
    return out.view(-1, 1, 1, 1)   # (B,1,1,1)

def pde_collate(batch):
    """Collate function that can handle missing phys parameters"""
    # batch: list of (x_cond, y, phys)
    x_list, y_list, phys_list = zip(*batch)
    x = torch.stack(x_list, dim=0)
    y = torch.stack(y_list, dim=0)

    # If phys is missing, return an empty tensor 
    if all(p is None for p in phys_list):
        phys = torch.empty(len(batch), 0)
    else:
        # If some are None and some not, raise inconsistency error
        if any(p is None for p in phys_list):
            raise ValueError("Inconsistent phys: some samples have phys, others are None")
        phys = torch.stack(phys_list, dim=0)

    return x, y, phys


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
        input_size=args.image_size,
        in_channels=args.cx + args.cy,
        learn_sigma=False,
        pos_mode=args.pos_mode,
        n_phys_params=args.phys_dim,
    )
    model.set_out_channels(cy=args.cy)

    
    # Parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    device = torch.device("cuda", device)
    model = DDP(model.to(device), device_ids=[device])
    diffusion = create_diffusion(timestep_respacing="",learn_sigma=False)  # default: 1000 steps, linear noise schedule
    for name in ["model_mean_type", "model_var_type", "loss_type", "rescale_timesteps"]:
        if hasattr(diffusion, name):
            print(name, "=", getattr(diffusion, name))
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4):
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
        for x_cond, y, phys in loader:
            # If no phys params, unsqueeze for consistent input shape
            if x_cond.dim() == 3:
                x_cond = x_cond.unsqueeze(1)
            if y.dim() == 3:
                y = y.unsqueeze(1)
            x_cond = x_cond.to(device)
            y      = y.to(device)
            phys = phys.to(device).float()
            if phys.numel() == 0:
                phys = None
                if train_steps == 0 and rank == 0:
                    print("x_cond shape:", x_cond.shape)
                    print("y shape:", y.shape)

            
            B = y.shape[0]
            # sample diffusion timesteps
            t = torch.randint(0, diffusion.num_timesteps, (B,), device=device, dtype=torch.long)

            # multi-scale training noise 
            eps = multiscale_noise(y)  # (B, Cy, H, W)

            # build y_t = sqrt(alpha_bar_t) y + sqrt(1-alpha_bar_t) eps
            
            alpha_bar = _extract_into_tensor(diffusion.alphas_cumprod, t, y.shape)

            y_t = alpha_bar.sqrt() * y + (1.0 - alpha_bar).sqrt() * eps

            # concat conditioning and noisy target along channels
            s = torch.cat([x_cond, y_t], dim=1)                         # (B, Cx+Cy, H, W)

            if rank == 0 and train_steps == 0:
                m = model.module if hasattr(model, "module") else model
                print("model type:", type(model))
                print("inner type:", type(m))
                print("inner forward:", m.forward)

            # forward pass: model predicts ε_hat for the target channels
            eps_hat = model(s, t, phys_params=phys)                     # (B, Cy, H, W) if learn_sigma=False
                                                                    # or (B, 2*Cy, ...) if learn_sigma=True
            # Debugging metrics
            with torch.no_grad():
                # cosine similarity over batch
                cos = torch.nn.functional.cosine_similarity(
                eps_hat.flatten(1), eps.flatten(1), dim=1
                ).mean()
                # relative RMSE
                rel = (eps_hat - eps).pow(2).mean().sqrt() / (eps.pow(2).mean().sqrt() + 1e-8)
                # std ratio (should approach 1)
                ratio = eps_hat.std() / (eps.std() + 1e-8)
                if rank == 0 and train_steps % args.log_every == 0:
                    print("cos:", cos.item(), "rel:", rel.item(), "std_ratio:", ratio.item())

            # if learn_sigma=True and the model outputs [eps_hat, other], need to split here.
            # For pure SimDiffPDE error prediction, set learn_sigma=False in model creation.

            # Multi loss: L2 + L1 on epsilon
            loss_l2 = F.mse_loss(eps_hat, eps)
            loss_l1 = F.l1_loss(eps_hat, eps)
            loss = loss_l2 + loss_l1

            # optimize
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module,decay=0.999)

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
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, required=True, help="grid size H=W")
    parser.add_argument("--cx", type=int, required=True, help="conditioning channels Cx")
    parser.add_argument("--cy", type=int, required=True, help="target channels Cy")
    parser.add_argument("--pos-mode", type=str, choices=["grid_sincos", "none"], default="grid_sincos")
    parser.add_argument("--phys-dim", type=int, default=0, help="number of global physical parameters (P)")
    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()
    main(args)
