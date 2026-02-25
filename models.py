# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp


def modulate(x, shift, scale):
    # x: (B, N, C), shift/scale: (B, C)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Physical Params              #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        t: (B,) can be fractional.
        returns: (B, dim)
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class PhysicalParameterEmbedder(nn.Module):
    def __init__(self, n_scalars: int, hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_scalars, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.proj(s)


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaLN-Zero conditioning.
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0.0,
        )

        # produces shift/scale/gate for MSA + shift/scale/gate for MLP
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C), c: (B, C)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    Final layer outputs per-node channels directly: (B, N, C_out).
    """
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C), c: (B, C)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)  # (B, N, out_channels)


class DiT(nn.Module):
    """
    DiT for node/token inputs (point clouds / unstructured meshes).

    Inputs:
      x      : (B, N, in_channels)
      coords : (B, N, coord_dim) required when pos_mode in {"rff","coord_mlp"}
      t      : (B,)
      phys_params : (B, P) optional

    Output:
      (B, N, C_out) where C_out = cy (or 2*cy if learn_sigma)
    """
    def __init__(
        self,
        coord_dim: int,
        in_channels: int,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = False,
        n_phys_params: int = 0,
        pos_mode: str = "none",     # "none" | "coord_mlp" | "rff"
        rff_scale: float = 1.0,     # only used for pos_mode="rff"
    ):
        super().__init__()
        self.coord_dim = coord_dim
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.learn_sigma = learn_sigma
        self.out_channels = None  # set via set_out_channels(cy)
        self.pos_mode = pos_mode

        # token embedding
        self.x_embedder = nn.Linear(in_channels, hidden_size)

        # diffusion timestep embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # optional global physical conditioning
        self.n_phys_params = n_phys_params
        self.phys_embedder = PhysicalParameterEmbedder(n_phys_params, hidden_size) if n_phys_params > 0 else None

        # coordinate positional encoding
        if self.pos_mode == "coord_mlp":
            self.pos_encoder = nn.Sequential(
                nn.Linear(coord_dim, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
            self.register_buffer("rff_B", torch.empty(0), persistent=False)

        elif self.pos_mode == "rff":
            assert hidden_size % 2 == 0, "hidden_size must be even for RFF sin/cos concat"
            B = torch.randn(coord_dim, hidden_size // 2) * rff_scale
            self.register_buffer("rff_B", B, persistent=False)
            self.pos_encoder = None

        else:
            self.pos_encoder = None
            self.register_buffer("rff_B", torch.empty(0), persistent=False)

        # transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        # final layer is created after out_channels known
        self.final_layer = None

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for blk in self.blocks:
            nn.init.constant_(blk.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(blk.adaLN_modulation[-1].bias, 0)

    def set_out_channels(self, cy: int):
        self.out_channels = (2 * cy) if self.learn_sigma else cy
        self.final_layer = FinalLayer(self.hidden_size, self.out_channels)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.normal_(self.final_layer.linear.weight, std=0.02)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def coords_pos_embed(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (B, N, coord_dim) -> (B, N, hidden_size)
        # assumes rff_B: (coord_dim, hidden_size/2)
        proj = (2 * math.pi) * (coords @ self.rff_B)   # (B, N, hidden/2)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, phys_params: torch.Tensor = None, coords: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, N, in_channels)
        t: (B,)
        phys_params: (B, P) optional
        coords: (B, N, coord_dim) required for pos_mode="coord_mlp" or "rff"
        """
        if self.final_layer is None or self.out_channels is None:
            raise RuntimeError("Call set_out_channels(cy) before forward().")

        B, N, Cin = x.shape
        if Cin != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got {Cin}")

        tokens = self.x_embedder(x)  # (B, N, hidden)

        if self.pos_mode in ("coord_mlp", "rff"):
            if coords is None:
                raise ValueError(f"coords required when pos_mode='{self.pos_mode}'")
            if coords.shape[:2] != (B, N) or coords.shape[2] != self.coord_dim:
                raise ValueError(f"coords must be (B,N,{self.coord_dim}), got {tuple(coords.shape)}")
            coords = coords.to(dtype=tokens.dtype, device=tokens.device)
            if self.pos_mode == "coord_mlp":
                tokens = tokens + self.pos_encoder(coords)
            else:
                tokens = tokens + self.coords_pos_embed(coords)

        t_cond = self.t_embedder(t)  # (B, hidden)
        if self.phys_embedder is not None and phys_params is not None:
            cond = t_cond + self.phys_embedder(phys_params)
        else:
            cond = t_cond

        for blk in self.blocks:
            tokens = blk(tokens, cond)

        out = self.final_layer(tokens, cond)  # (B, N, C_out)
        return out


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL(**kwargs): return DiT(depth=28, hidden_size=1152, num_heads=16, **kwargs)
def DiT_L (**kwargs): return DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)
def DiT_B (**kwargs): return DiT(depth=12, hidden_size=768,  num_heads=12, **kwargs)
def DiT_S (**kwargs): return DiT(depth=12, hidden_size=384,  num_heads=6,  **kwargs)

DiT_models = {
    'DiT-XL': DiT_XL,
    'DiT-L':  DiT_L,
    'DiT-B':  DiT_B,
    'DiT-S':  DiT_S,
}


