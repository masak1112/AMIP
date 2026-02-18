import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from modules.layers.positional_encoding import TimestepEmbedder
from modules.layers.spherical_harmonics import SphericalHarmonicsPE
from modules.layers.unpatchify import SubPixelConvICNR_2D, Unpatchify
from modules.layers.patchify import PatchEmbed
from modules.layers.cross_attention import CrossAttentionBlock
from modules.layers.arches_layers import (
    CondBasicLayer,
    DCDownSample,
    DCUpSample,
    Mlp,
    TimestepEmbedder as ArchesTimestepEmbedder,
)


class DiTBlock(nn.Module):
    """
    Vanilla self-attention transformer block with AdaLN-Zero timestep conditioning.

    Input/output shape: [b, n, dim] where n = (nlat//p) * (nlon//p).
    """

    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        dim_head = dim // num_heads

        # Self-attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim)
        self.scale = dim_head ** -0.5

        # Feedforward
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

        # AdaLN-Zero modulation: 6 * dim for (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )

        # Zero-init the modulation output
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, t_emb):
        """
        Args:
            x: [b, n, dim]
            t_emb: [b, dim] timestep embedding
        """
        # AdaLN modulation parameters
        mod = self.adaLN_modulation(t_emb).unsqueeze(1)  # [b, 1, 6*dim]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)

        # Self-attention with AdaLN
        h = self.norm1(x)
        h = h * (1 + scale_msa) + shift_msa

        b, n, c = h.shape
        qkv = self.qkv(h).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, b, heads, n, dim_head]
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1) # [b, heads, n, n]
        h = (attn @ v).transpose(1, 2).reshape(b, n, c) # [b, n, dim]
        h = self.attn_out(h) # [b, n, dim]

        x = x + gate_msa * h

        # FFN with AdaLN
        h = self.norm2(x)
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        x = x + gate_mlp * h

        return x


class SIDiT(nn.Module):
    """
    Patchified Diffusion Transformer for stochastic interpolant velocity prediction.

    Architecture:
    - PatchEmbed for main input:
        - use_history=True:  (2*in_channels) — I_t concat with downsampled current state
        - use_history=False: (in_channels)   — I_t only, with cond via cross-attention
    - Separate conditioning encoder for cross-attention context
    - Spherical harmonic positional encoding
    - N blocks of: DiTBlock (vanilla self-attn with AdaLN) + CrossAttentionBlock
    - Unpatchify: dim -> out_channels @ nlat x nlon
    - Zero-initialized output projection
    """

    def __init__(self,
                 in_channels=249,
                 out_channels=249,
                 dim=384,
                 num_heads=8,
                 num_blocks=8,
                 patch_size=4,
                 nlat=180,
                 nlon=360,
                 dropout=0.0,
                 unpatch="vanilla",
                 use_history=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.patch_size = patch_size
        self.nlat = nlat
        self.nlon = nlon
        self.dropout = dropout
        self.use_history = use_history

        # Pad spatial dims to be divisible by patch_size
        self.nlat_pad = math.ceil(nlat / patch_size) * patch_size
        self.nlon_pad = math.ceil(nlon / patch_size) * patch_size
        self.pad_lat = self.nlat_pad - nlat
        self.pad_lon = self.nlon_pad - nlon

        self.grid_x = self.nlat_pad // patch_size
        self.grid_y = self.nlon_pad // patch_size
        self.with_poles = False

        # Patch embedding for main input
        # use_history: [I_t; cond] channel-wise concat (2*c), cross-attn attends to history
        # no history:  I_t only (c), cross-attn attends to cond
        main_in_chans = 2 * in_channels if use_history else in_channels
        self.patch_embed_main = PatchEmbed(
            patch_size=patch_size,
            in_chans=main_in_chans,
            hidden_size=dim,
            flatten=False)

        # Patch embedding for cross-attention context (history or cond)
        self.patch_embed_cond = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_channels,
            hidden_size=dim,
            flatten=False)

        # Spherical harmonic positional encoding
        l_max = 20
        self.pe_embed = SphericalHarmonicsPE(l_max, dim, dim, use_mlp=True)
        self.pe2patch = PatchEmbed(
            patch_size=patch_size,
            in_chans=dim,
            hidden_size=dim,
            flatten=False)

        # Timestep embedding
        self.t_embedder = TimestepEmbedder(dim)

        # Transformer blocks: vanilla self-attention + cross-attention
        sa_blocks = []
        ca_blocks = []
        for _ in range(num_blocks):
            sa_blocks.append(DiTBlock(dim, num_heads, mlp_ratio=4, dropout=dropout))
            ca_blocks.append(CrossAttentionBlock(num_heads, dim))

        self.sa_blocks = nn.ModuleList(sa_blocks)
        self.ca_blocks = nn.ModuleList(ca_blocks)

        # Unpatchify
        if unpatch == "subpixel":
            self.unpatchify_layer = SubPixelConvICNR_2D(
                img_size=(self.nlat_pad, self.nlon_pad),
                patch_size=(patch_size, patch_size),
                in_chans=dim,
                out_chans=dim,
                cond_dim=dim,
                num_lat=self.nlat_pad)
        elif unpatch == "vanilla":
            self.unpatchify_layer = Unpatchify(
                grid_size=(self.grid_x, self.grid_y),
                patch_size=(patch_size, patch_size),
                in_dim=dim,
                out_dim=dim,
                cond_dim=dim)
        else:
            raise ValueError(f"unpatch type '{unpatch}' not supported")

        # Output projection (zero-initialized for stable training start)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_channels))

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-init output projection for stable training
        nn.init.constant_(self.out_proj[-1].weight, 0)
        nn.init.constant_(self.out_proj[-1].bias, 0)

        # Re-zero-init AdaLN modulation outputs (apply overwrites them)
        for block in self.sa_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    @torch.no_grad()
    def get_grid(self, nlat, nlon, device):
        if self.with_poles:
            lat = torch.linspace(-math.pi / 2, math.pi / 2, nlat).to(device)
        else:
            lat_end = (nlat - 1) * (2 * math.pi / nlon) / 2
            lat = torch.linspace(-lat_end, lat_end, nlat).to(device)
        lon = torch.linspace(0, 2 * math.pi - (2 * math.pi / nlon), nlon).to(device)
        return lat, lon

    def forward(self, x_noised, t, cond, history=None):
        """
        Args:
            x_noised: [b, c, nlat, nlon] — interpolant I_t (channel-first from assemble_input)
            t: [b, 1] — timestep
            cond: [b, c, nlat, nlon] — downsampled current state (channel-first).
                  When use_history=True: concatenated channel-wise with x_noised.
                  When use_history=False: used as cross-attention context.
            history: [b, c, nlat, nlon] — high-res prior state/history (channel-first),
                     used as cross-attention context when use_history=True. Ignored otherwise.

        Returns:
            [b, c, nlat, nlon] — predicted velocity (channel-first)
        """
        batch_size = x_noised.shape[0]
        nlat, nlon = self.nlat, self.nlon

        if self.use_history:
            # History mode: concat cond channel-wise with I_t, cross-attend to history
            x_input = torch.cat([x_noised, cond], dim=1)  # [b, 2c, nlat, nlon]
            ca_context = history if history is not None else cond
        else:
            # Standard mode: I_t only as input, cross-attend to cond
            x_input = x_noised
            ca_context = cond

        # Pad spatial dims to be divisible by patch_size
        if self.pad_lat > 0 or self.pad_lon > 0:
            # F.pad order: (left, right, top, bottom) for last two dims
            x_input = F.pad(x_input, (0, self.pad_lon, 0, self.pad_lat), mode='reflect')
            ca_context = F.pad(ca_context, (0, self.pad_lon, 0, self.pad_lat), mode='reflect')

        # Get grid coordinates for positional encoding at padded resolution
        lat, lon = self.get_grid(self.nlat_pad, self.nlon_pad, x_input.device)

        # Convert channel-first to channel-last for PatchEmbed: [b, c, h, w] -> [b, h, w, c]
        x_nhwc = x_input.permute(0, 2, 3, 1)
        c_nhwc = ca_context.permute(0, 2, 3, 1)

        # Patchify: [b, h, w, c] -> [b, h//p, w//p, dim]
        x = self.patch_embed_main(x_nhwc)
        c = self.patch_embed_cond(c_nhwc)

        # Positional encoding
        sphere_pe = self.pe_embed(lat + math.pi / 2, lon - math.pi)
        sphere_pe = sphere_pe.expand(batch_size, -1, -1, -1)  # [b, nlat_pad, nlon_pad, dim]
        sphere_pe = self.pe2patch(sphere_pe)  # [b, nlat_pad//p, nlon_pad//p, dim]

        x = x + sphere_pe
        c = c + sphere_pe

        # Flatten spatial dims for sequence processing: [b, h//p, w//p, dim] -> [b, n, dim]
        x = rearrange(x, 'b ny nx c -> b (ny nx) c')
        c = rearrange(c, 'b ny nx c -> b (ny nx) c')

        # Timestep embedding
        if t is not None and len(t.shape) == 1:
            t = t[:, None]
        t_emb = self.t_embedder(t)  # [b, dim]

        # Transformer blocks: vanilla self-attention + cross-attention with history
        for sa_block, ca_block in zip(self.sa_blocks, self.ca_blocks):
            x = sa_block(x, t_emb)        # self-attention with AdaLN
            x = ca_block(x, c)            # cross-attention with history context

        # Unpatchify: [b, n, dim] -> [b, nlat, nlon, dim]
        x = self.unpatchify_layer(x, t_emb)

        # Output projection
        x = self.out_proj(x)  # [b, nlat, nlon, out_channels]

        # Convert back to channel-first: [b, h, w, c] -> [b, c, h, w]
        x = x.permute(0, 3, 1, 2)

        # Crop back to original spatial dims
        if self.pad_lat > 0 or self.pad_lon > 0:
            x = x[:, :, :nlat, :nlon]

        return x


class ArchesSiT(nn.Module):
    """
    Arches-backbone Stochastic Interpolant Transformer.

    Replaces SIDiT's vanilla self-attention + cross-attention with the Arches
    hierarchical window-attention backbone (CondBasicLayer + DCDownSample/DCUpSample).

    Conditioning (cond) is channel-concatenated with x_noised as the patch-embed input.
    Timestep t is embedded into a global AdaLN conditioning vector for all layers.

    Interface is identical to SIDiT:
        __init__(in_channels, out_channels, dim, num_heads, num_blocks, patch_size,
                 nlat, nlon, dropout, unpatch, use_history)
        forward(x_noised, t, cond, history=None) -> [b, out_channels, nlat, nlon]
    """

    def __init__(self,
                 in_channels=249,
                 out_channels=249,
                 dim=384,
                 num_heads=8,
                 num_blocks=8,
                 patch_size=4,
                 nlat=180,
                 nlon=360,
                 dropout=0.0,
                 unpatch="vanilla",
                 use_history=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.patch_size = patch_size
        self.nlat = nlat
        self.nlon = nlon
        self.use_history = use_history

        # Pad spatial dims to be divisible by patch_size
        self.nlat_pad = math.ceil(nlat / patch_size) * patch_size
        self.nlon_pad = math.ceil(nlon / patch_size) * patch_size
        self.pad_lat = self.nlat_pad - nlat
        self.pad_lon = self.nlon_pad - nlon

        # Patch-space grid (treat zdim=1: no pressure levels in flat 2D input)
        self.zdim = 1
        self.zlat = self.nlat_pad // patch_size
        self.zlon = self.nlon_pad // patch_size
        self.layer1_shape = (self.zlat, self.zlon)
        self.layer2_shape = (self.zlat // 2, self.zlon // 2)

        # Input channels: concat x_noised + cond, optionally also history
        in_chans = 3 * in_channels if use_history else 2 * in_channels

        # Patch embedding: (b, in_chans, nlat_pad, nlon_pad) -> (b, dim, zlat, zlon)
        self.patch_embed = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)

        # Timestep embedding -> global cond_emb for AdaLN (uses arches embedder)
        self.t_embedder = ArchesTimestepEmbedder(dim)

        # Depth distribution: [1, 3, 3, 1] scaled by factor (num_blocks=8 -> factor=1)
        factor = max(1, num_blocks // 8)
        depth1, depth2, depth3, depth4 = factor, 3 * factor, 3 * factor, factor

        # window_size=(1, 6, 10): EarthSpecificBlock pads internally for non-divisible grids
        window_size = (1, 6, 10)

        layer_args = dict(
            cond_dim=dim,
            window_size=window_size,
            act_layer=nn.GELU,
            drop=dropout,
            mlp_layer=Mlp,
            mlp_ratio=4.0,
        )

        self.layer1 = CondBasicLayer(
            dim=dim,
            input_resolution=(self.zdim, *self.layer1_shape),
            depth=depth1,
            num_heads=num_heads,
            drop_path=[0.0] * depth1,
            **layer_args,
        )
        self.downsample = DCDownSample(
            in_dim=dim,
            out_dim=dim * 2,
            input_resolution=(self.zdim, *self.layer1_shape),
            output_resolution=(self.zdim, *self.layer2_shape),
        )
        self.layer2 = CondBasicLayer(
            dim=dim * 2,
            input_resolution=(self.zdim, *self.layer2_shape),
            depth=depth2,
            num_heads=num_heads,
            drop_path=[0.0] * depth2,
            **layer_args,
        )
        self.layer3 = CondBasicLayer(
            dim=dim * 2,
            input_resolution=(self.zdim, *self.layer2_shape),
            depth=depth3,
            num_heads=num_heads,
            drop_path=[0.0] * depth3,
            **layer_args,
        )
        self.upsample = DCUpSample(
            dim * 2, dim,
            (self.zdim, *self.layer2_shape),
            (self.zdim, *self.layer1_shape),
        )
        # Skip connection doubles the channel dim for layer4
        skip_dim = 2 * dim
        self.layer4 = CondBasicLayer(
            dim=skip_dim,
            input_resolution=(self.zdim, *self.layer1_shape),
            depth=depth4,
            num_heads=num_heads,
            drop_path=[0.0] * depth4,
            **layer_args,
        )

        # Project skip-connected features back to dim before unpatchify
        self.skip_proj = nn.Linear(skip_dim, dim)

        # Unpatchify: [b, n, dim] -> [b, nlat_pad, nlon_pad, dim]
        if unpatch == "subpixel":
            self.unpatchify_layer = SubPixelConvICNR_2D(
                img_size=(self.nlat_pad, self.nlon_pad),
                patch_size=(patch_size, patch_size),
                in_chans=dim,
                out_chans=dim,
                cond_dim=dim,
                num_lat=self.nlat_pad,
            )
        elif unpatch == "vanilla":
            self.unpatchify_layer = Unpatchify(
                grid_size=(self.zlat, self.zlon),
                patch_size=(patch_size, patch_size),
                in_dim=dim,
                out_dim=dim,
                cond_dim=dim,
            )
        else:
            raise ValueError(f"unpatch type '{unpatch}' not supported")

        # Output projection (zero-initialized for stable training start)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_channels),
        )

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-init output projection for stable training
        nn.init.constant_(self.out_proj[-1].weight, 0)
        nn.init.constant_(self.out_proj[-1].bias, 0)

    def forward(self, x_noised, t, cond, history=None):
        """
        Args:
            x_noised: [b, c, nlat, nlon] -- interpolant I_t
            t:        [b, 1] or [b]      -- timestep
            cond:     [b, c, nlat, nlon] -- conditioning context (always concat'd)
            history:  [b, c, nlat, nlon] -- extra history (concat'd when use_history=True)

        Returns:
            [b, out_channels, nlat, nlon]
        """
        batch_size = x_noised.shape[0]
        nlat, nlon = self.nlat, self.nlon

        # Build input tensor
        if self.use_history and history is not None:
            x_input = torch.cat([x_noised, cond, history], dim=1)
        else:
            x_input = torch.cat([x_noised, cond], dim=1)

        # Pad spatial dims
        if self.pad_lat > 0 or self.pad_lon > 0:
            x_input = F.pad(x_input, (0, self.pad_lon, 0, self.pad_lat), mode='reflect')

        # Patch embed: [b, in_chans, nlat_pad, nlon_pad] -> [b, dim, zlat, zlon]
        x = self.patch_embed(x_input)

        # Add zdim=1 and flatten to token sequence: [b, zlat*zlon, dim]
        B, C, Lat, Lon = x.shape
        x = x.unsqueeze(2)                      # [b, dim, 1, zlat, zlon]
        x = x.reshape(B, C, -1).transpose(1, 2) # [b, n, dim]

        # Timestep -> global AdaLN conditioning; arches embedder expects [b] (1-D)
        if t is not None and t.dim() > 1:
            t = t.squeeze(-1)
        cond_emb = self.t_embedder(t)  # [b, dim]

        # Hierarchical Arches backbone (U-Net with window attention)
        x = self.layer1(x, cond_emb)

        skip = x
        x = self.downsample(x)

        x = self.layer2(x, cond_emb)
        x = self.layer3(x, cond_emb)

        x = self.upsample(x)
        x = torch.cat([x, skip], dim=-1)  # [b, n, 2*dim]
        x = self.layer4(x, cond_emb)

        # Project skip-connected features to dim
        x = self.skip_proj(x)  # [b, n, dim]

        # Unpatchify: [b, n, dim] -> [b, nlat_pad, nlon_pad, dim]
        x = self.unpatchify_layer(x, cond_emb)

        # Output projection
        x = self.out_proj(x)  # [b, nlat_pad, nlon_pad, out_channels]

        # Channel-first and crop to original dims
        x = x.permute(0, 3, 1, 2)  # [b, out_channels, nlat_pad, nlon_pad]
        if self.pad_lat > 0 or self.pad_lon > 0:
            x = x[:, :, :nlat, :nlon]

        return x
