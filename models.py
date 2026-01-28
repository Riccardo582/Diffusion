# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class PhysicalParameterEmbedder(nn.Module):
    def __init__(self, n_scalars: int, hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_scalars, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
    # 
    def forward(self,s:torch.Tensor):
        return self.proj(s)


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=1, # Set later to Cx + Cy
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=False,
        # Number of physical parameters to condition on (ex. T for non-steady problems)
        n_phys_params=0,
        pos_mode = "grid_sincos" # grid_sincos for regular grids, "none" for irregular grids
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = None     # will be set via set_out_channels(cy, learn_sigma
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_mode = pos_mode
        

        # Embedding layers:
        # Patchify concatenated input
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        # Embed Diffusion timestep
        self.t_embedder = TimestepEmbedder(hidden_size)
        # Physical parameter embedder
        self.n_phys_params = n_phys_params
        self.phys_embedder = (
            PhysicalParameterEmbedder(n_phys_params, hidden_size)
                if n_phys_params > 0 else None
            )
        # Patchify
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, hidden_size), requires_grad=False)
        self._pos_inited = False

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        
        # out_channels set later

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Positional embedding is lazily initialized in _maybe_init_pos()

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)


        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        # (Done after final_layer is created via set_out_channels)
 
    def set_out_channels(self, cy: int):
        """Call once after constructing the model to define target channels.
        Make sure to call set:out_channels(cy) after initializing the DiT model."""
        self.out_channels = (2 * cy) if self.learn_sigma else cy
        self.final_layer = FinalLayer(self.x_embedder.embed_dim, self.patch_size, self.out_channels)
        # Zero-init final heads (AdaLN-Zero architecture)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    @torch.no_grad()
    def _maybe_init_pos(self):
        """Fill fixed 2D sin-cos positions once, using rectangular (Gh,Gw) from PatchEmbed."""
        if self._pos_inited or self.pos_mode != "grid_sincos":
            return
        # timm PatchEmbed computes grid_size at build time if input_size was given
        Gh, Gw = self.x_embedder.grid_size
        pe = get_2d_sincos_rect_pos_embed(self.x_embedder.embed_dim, Gh, Gw)  # (Gh*Gw, C)
        self.pos_embed.data = torch.from_numpy(pe).float().unsqueeze(0)       # (1,N,C)
        self._pos_inited = True

    def unpatchify(self, x):
        """
        x: (N, T, patch_size^2 * C)
        out: (N, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        Gh, Gw = self.x_embedder.grid_size # number of patches along height and width
        h,w = Gh, Gw

        x = x.reshape(x.shape[0], h, w, p, p, c) # (N, Gh, Gw, p, p, C)
        x = torch.einsum('nhwpqc->nchpwq', x)   # (N, C, Gh, p, Gw, p)
        unpatched = x.reshape(shape=(x.shape[0], c, h * p, w * p)) 
        return unpatched

def forward(self, s, t, phys_params=None, coords=None):
    """
    s:           (B, Cx+Cy[, +coord_dim], H, W)
    t:           (B,)
    phys_params: (B, P)  optional global physical parameters
    coords:      unused here (coords-as-channels expected already in `s`)
    """
    # Patchify + optional positional encoding
    self._maybe_init_pos()
    tokens = self.x_embedder(s)              # (B, N, C)
    if self.pos_mode == "grid_sincos":
        tokens = tokens + self.pos_embed     # (1, N, C) broadcast over batch

    # Build conditioning vector from diffusion time + physical paramsS
    t_cond = self.t_embedder(t)              # (B, C)
    if self.phys_embedder is not None and phys_params is not None:
        p_cond = self.phys_embedder(phys_params)  # (B, C)
        cond = t_cond + p_cond
    else:
        cond = t_cond

    # Pass through DiT blocks with AdaLN-Zero conditioning
    for blk in self.blocks:
        tokens = blk(tokens, cond)           # (B, N, C)

    # Final AdaLN + linear → patch outputs, then unpatchify
    out_patches = self.final_layer(tokens, cond)  # (B, N, p^2 * C_out)
    out = self.unpatchify(out_patches)           # (B, C_out, H, W)
    return out



#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_rect_pos_embed(embed_dim, grid_height,grid_width):
    """
    Rectangular grid 2D sine-cosine positional embedding
    grid_height: grid height 
    grid_width: grid width   
    """
    grid_h = np.arange(grid_height, dtype=np.float32)
    grid_w = np.arange(grid_width, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0).reshape([2, -1])  # 2, H*W

    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim//2, grid[1])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim//2, grid[0])  # (H*W, D/2)

    return np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}
