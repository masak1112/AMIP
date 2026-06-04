"""ClimaDiT backbone, vendored copy with the patch_size fix.

The upstream ``interpolant_pdes/modules/models/ClimaDiT.py`` declares
``patch_size`` as a config knob but then hardcodes ``patch_size=1`` inside
``self.patch_embed`` and ``self.pe2patch`` while building ``self.unpatchify_layer``
with the real ``patch_size``. Any ``patch_size>1`` therefore breaks the
``grid_emb + sphere_pe`` add (shape mismatch). The local
``/glade/work/weidong/code/FB/GenCast4FB/GenPred/ClimaDiT.py`` already fixed
this; we vendor the fix here so we don't have to depend on the GenPred tree.

Imports use ``modules.layers.*`` from the upstream ``interpolant_pdes`` repo,
which the training script adds to ``sys.path``.

Config layout:
    config['dit']        — model hyper-parameters (in_dim/out_dim/dim/...)
    config['data']       — nlat / nlon
This matches the GenPred convention; the upstream's nested
``config['model']['dit']`` shape is deliberately *not* used.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from modules.layers.embedding import TimestepEmbedder
from modules.layers.spherical_harmonics import SphericalHarmonicsPE
from modules.layers.basics import (
    MLP, LayerNorm,
    bias_dropout_add_scale_fused_train,
    bias_dropout_add_scale_fused_inference,
    modulate_fused,
)
from modules.layers.factorized_attention import FADiTBlockS2
from modules.layers.unpatchify import Unpatchify
from modules.layers.patchify import PatchEmbed
from modules.layers.cross_attention import CrossAttentionBlock


class DiTBlock(nn.Module):
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = MLP(dim, expansion_ratio=mlp_ratio)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        return bias_dropout_add_scale_fused_inference

    def forward(self, x, scalar_cond):
        batch_size = x.shape[0]
        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        (shift_msa, scale_msa, gate_msa,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(
            scalar_cond)[:, None].chunk(6, dim=2)

        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b h three s d',
                        three=3, h=self.n_heads)
        qk, v = qkv[:, :, :2], qkv[:, :, 2]
        q, k = qk[:, :, 0], qk[:, :, 1]
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
        x = rearrange(x, 'b h s d -> b s (h d)', b=batch_size)

        x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa,
                                  x_skip, self.dropout)
        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
            None, gate_mlp, x, self.dropout,
        )
        return x


class ClimaDiT(nn.Module):
    """ClimaDiT backbone — wide DiT with cross-attended grid conditioning.

    Differences vs. the upstream copy:
      * ``patch_embed`` and ``pe2patch`` now use ``self.patch_size`` instead of
        the hardcoded ``patch_size=1``, so any ``patch_size>=1`` works.
      * Top-level config is ``config['dit']`` and ``config['data']`` (no nested
        ``config['model']`` wrapper).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        modelconfig = config['dit']

        self.in_dim = modelconfig['in_dim']
        self.out_dim = modelconfig['out_dim']
        self.dim = modelconfig['dim']
        self.cond_dim = modelconfig['cond_dim']
        self.num_heads = modelconfig['num_heads']
        self.num_fa_blocks = modelconfig['num_fa_blocks']
        self.num_sa_blocks = modelconfig['num_sa_blocks']
        self.num_ca_blocks = modelconfig['num_ca_blocks']
        self.num_cond = modelconfig['num_cond']
        self.patch_size = modelconfig['patch_size']
        self.num_constants = modelconfig['num_constants']
        self.nlat = config['data']['nlat']
        self.nlon = config['data']['nlon']

        self.with_poles = False
        self.grid_x = self.nlat // self.patch_size
        self.grid_y = self.nlon // self.patch_size

        # grid (boundary + constants) embedding -> patched
        self.constant_embedder = nn.Sequential(
            Rearrange('b ny nx c -> b c ny nx'),
            nn.Conv2d(self.num_constants, self.dim,
                      kernel_size=self.patch_size,
                      stride=self.patch_size, padding=0),
            nn.SiLU(),
            nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0),
            Rearrange('b c ny nx -> b ny nx c'),
        )

        # state input embedding -> patched
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=self.in_dim,
            hidden_size=self.dim,
            flatten=False,
        )

        # spherical-harmonic positional embedding -> patched
        self.pe_embed = SphericalHarmonicsPE(
            modelconfig['l_max'], self.dim, self.dim, use_mlp=True,
        )
        self.pe2patch = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=self.dim,
            hidden_size=self.dim,
            flatten=False,
        )

        if self.num_cond > 0:
            self.cond_map = TimestepEmbedder(self.cond_dim,
                                             num_conds=self.num_cond)

        fa_blocks = []
        for _ in range(self.num_fa_blocks):
            fa_blocks.append(FADiTBlockS2(
                self.dim,
                self.dim // self.num_heads,
                self.num_heads,
                modelconfig['proj_bottleneck_dim'],
                self.dim,
                self.cond_dim,
                kernel_expansion_ratio=modelconfig['kernel_expansion_ratio'],
                use_softmax=True,
                depth_dropout=modelconfig['depth_dropout'],
            ))

        sa_blocks = [
            DiTBlock(self.dim, self.num_heads, self.cond_dim,
                     dropout=modelconfig['depth_dropout'])
            for _ in range(self.num_sa_blocks)
        ]
        ca_blocks = [
            CrossAttentionBlock(self.num_heads, self.dim)
            for _ in range(self.num_ca_blocks)
        ]

        self.fa_blocks = nn.ModuleList(fa_blocks)
        self.sa_blocks = nn.ModuleList(sa_blocks)
        self.ca_blocks = nn.ModuleList(ca_blocks)

        self.scale_by_sigma = modelconfig['scale_by_sigma']
        if self.scale_by_sigma:
            self.sigma_map = TimestepEmbedder(self.cond_dim)

        self.unpatchify_layer = Unpatchify(
            grid_size=(self.grid_x, self.grid_y),
            patch_size=(self.patch_size, self.patch_size),
            in_dim=self.dim,
            out_dim=self.out_dim,
            cond_dim=self.cond_dim,
        )

        self.init_params()

    @torch.no_grad()
    def get_grid(self, nlat, nlon, device='cpu'):
        if self.with_poles:
            lat = torch.linspace(-math.pi / 2, math.pi / 2, nlat).to(device)
        else:
            lat_end = (nlat - 1) * (2 * math.pi / nlon) / 2
            lat = torch.linspace(-lat_end, lat_end, nlat).to(device)
        lon = torch.linspace(0, 2 * math.pi - (2 * math.pi / nlon),
                             nlon).to(device)
        latlon = torch.stack(torch.meshgrid(lat, lon, indexing='ij'), dim=-1)
        return latlon, lat, lon

    def forward(self, u, sigma_t, scalar_params, grid_params):
        # u:             [B, ny, nx, in_dim]
        # sigma_t:       [B] or [B, 1]
        # scalar_params: [B, num_cond]
        # grid_params:   [B, ny, nx, num_constants]
        batch_size = u.size(0)
        nlat, nlon = u.size(1), u.size(2)
        nlat_grid = nlat // self.patch_size
        nlon_grid = nlon // self.patch_size
        _, lat, lon = self.get_grid(nlat, nlon, u.device)
        _, lat_grid, lon_grid = self.get_grid(nlat_grid, nlon_grid, u.device)

        lat_grid_diff = lat_grid.unsqueeze(0) - lat_grid.unsqueeze(1)
        lon_grid_diff = lon_grid.unsqueeze(0) - lon_grid.unsqueeze(1)

        u = self.patch_embed(u)                  # [B, ny/p, nx/p, dim]
        grid_emb = self.constant_embedder(grid_params)
        sphere_pe = self.pe_embed(
            lat + math.pi / 2, lon - math.pi,
        ).expand(batch_size, -1, -1, -1)         # [B, ny, nx, dim]
        sphere_pe = self.pe2patch(sphere_pe)     # [B, ny/p, nx/p, dim]

        u = u + sphere_pe
        grid_emb = grid_emb + sphere_pe

        if self.scale_by_sigma:
            if sigma_t.dim() == 1:
                sigma_t = sigma_t.unsqueeze(-1)
            c_t = F.silu(self.sigma_map(sigma_t))
            c = c_t
        else:
            c_t = None
            c = 0

        if self.num_cond > 0:
            c = c + self.cond_map(scalar_params)

        for l in range(self.num_fa_blocks):
            u = self.fa_blocks[l](
                u, lat_grid, lat_grid_diff, lon_grid_diff, c,
            )

        u = rearrange(u, 'b ny nx c -> b (ny nx) c')
        grid_emb = rearrange(grid_emb, 'b ny nx c -> b (ny nx) c')

        for l in range(self.num_sa_blocks):
            u = self.sa_blocks[l](u, c)
            if l < self.num_ca_blocks:
                u = self.ca_blocks[l](u, grid_emb)

        if self.scale_by_sigma:
            u = self.unpatchify_layer(u, c_t)
        else:
            u = self.unpatchify_layer(u)
        return u

    def init_params(self):
        nn.init.constant_(self.constant_embedder[-2].weight, 0)
