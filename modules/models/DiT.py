import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from modules.layers.conv import SphereConv2d

from modules.layers.positional_encoding import (
    TimestepEmbedder,
    RotaryEmbedding,
    apply_2d_rotary_pos_emb,
)
from modules.layers.unpatchify import SubPixelConvICNR_2D, Unpatchify, sphere_pad, PatchInterpolate2D
from modules.layers.patchify import PatchEmbed
from modules.layers.embedding import CalendarEmbedding

class PatchBoundaryRefiner(nn.Module):
    """Lightweight conv smoother applied only to x_0 estimates, not velocities."""
    def __init__(self, channels, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            SphereConv2d(channels, hidden, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.GELU(),
            SphereConv2d(hidden, hidden, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.GELU(),
            SphereConv2d(hidden, hidden, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.GELU(),
            SphereConv2d(hidden, hidden, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        # Zero-init final layer so it starts as identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


class DiTBlock(nn.Module):
    """
    Vanilla self-attention transformer block with AdaLN-Zero timestep conditioning
    and 2D Rotary Position Embedding (RoPE).

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

    def forward(self, x, t_emb, rope_cos_lat, rope_sin_lat, rope_cos_lon, rope_sin_lon):
        """
        Args:
            x: [b, n, dim]
            t_emb: [b, dim] timestep embedding
            rope_cos_lat: [1, n, dim_head//2] cosine freqs for latitude
            rope_sin_lat: [1, n, dim_head//2] sine freqs for latitude
            rope_cos_lon: [1, n, dim_head//2] cosine freqs for longitude
            rope_sin_lon: [1, n, dim_head//2] sine freqs for longitude
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

        # Apply 2D RoPE to q and k
        q = apply_2d_rotary_pos_emb(q, rope_cos_lat, rope_sin_lat, rope_cos_lon, rope_sin_lon)
        k = apply_2d_rotary_pos_emb(k, rope_cos_lat, rope_sin_lat, rope_cos_lon, rope_sin_lon)

        h = F.scaled_dot_product_attention(q, k, v)
        h = rearrange(h, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")
        h = self.attn_out(h) # [b, n, dim]

        x = x + gate_msa * h

        # FFN with AdaLN
        h = self.norm2(x)
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        x = x + gate_mlp * h

        return x


class DiT(nn.Module):
    """
    Patchified Diffusion Transformer for stochastic interpolant velocity prediction.

    Architecture:
    - PatchEmbed for main input
    - 2D Rotary Position Embedding (RoPE) on lat/lon patch grid
    - N blocks of DiTBlock (self-attn with AdaLN + RoPE)
    - Unpatchify: dim -> out_channels @ nlat x nlon
    - Zero-initialized output projection
    """

    def __init__(self,
                 in_channels=249,
                 out_channels=249,
                 dim=384,
                 num_heads=8,
                 num_blocks=8,
                 patch_size=2,
                 nlat=180,
                 nlon=360,
                 dropout=0.0,
                 unpatch="subpixel",
                 scalar_dim=1,
                 c_grid_dim=0,
                 c_grid_embed_dim=4,
                 c_grid_downsample=4):
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
        self.c_grid_dim = c_grid_dim

        # Pad spatial dims to be divisible by patch_size              # test case if nlat,nlon = 45, 90
        self.nlat_pad = math.ceil(nlat / patch_size) * patch_size     # 45/2 = 23 * 2 = 46
        self.nlon_pad = math.ceil(nlon / patch_size) * patch_size     # 90/2 = 45 * 2 = 90
        self.pad_lat = self.nlat_pad - nlat                           # 46 - 45 = 1
        self.pad_lon = self.nlon_pad - nlon                           # 90 - 90 = 0

        # Polar padding for latitude (split top/bottom), circular for longitude (split left/right)
        self.pad_lat_top = math.ceil(self.pad_lat / 2)                # 1/2 = 0.5 -> 1
        self.pad_lat_bottom = self.pad_lat - self.pad_lat_top         # 1 - 1 = 0
        self.pad_lon_left = math.ceil(self.pad_lon / 2)               # 0/2 = 0
        self.pad_lon_right = self.pad_lon - self.pad_lon_left         # 0 - 0 = 0

        self.grid_x = self.nlat_pad // patch_size
        self.grid_y = self.nlon_pad // patch_size
        self.with_poles = False

        # c_grid downsampling: (c_grid_dim, 180, 360) -> (c_grid_embed_dim, 45, 90)
        if c_grid_downsample > 0:
            self.c_grid_embed = nn.Conv2d(c_grid_dim, c_grid_embed_dim,
                                          kernel_size=c_grid_downsample,
                                          stride=c_grid_downsample)
            patch_in_channels = in_channels + c_grid_embed_dim
        else:
            self.c_grid_embed = None
            patch_in_channels = in_channels

        self.patch_embed_main = PatchEmbed(
            patch_size=patch_size,
            in_chans=patch_in_channels,
            hidden_size=dim,
            flatten=False)

        # 2D RoPE: one RotaryEmbedding per spatial axis
        # Each axis gets half the head dimension
        dim_head = dim // num_heads
        self.rope_lat = RotaryEmbedding(dim_head // 2)
        self.rope_lon = RotaryEmbedding(dim_head // 2)

        # Timestep embedding
        self.t_embedder = TimestepEmbedder(dim, num_conds = scalar_dim)

        # Transformer blocks
        sa_blocks = []
        for _ in range(num_blocks):
            sa_blocks.append(DiTBlock(dim, num_heads, mlp_ratio=4, dropout=dropout))

        self.sa_blocks = nn.ModuleList(sa_blocks)

        # Unpatchify
        self.unpatch = unpatch

        if unpatch == "subpixel":
            self.unpatchify_layer = SubPixelConvICNR_2D(
                grid_size = (self.grid_x, self.grid_y),
                patch_size=(patch_size, patch_size),
                in_chans=dim,
                out_chans=out_channels)
        elif unpatch == "interpolate":
            self.unpatchify_layer = PatchInterpolate2D(grid_size=(self.grid_x, self.grid_y),
                                                       patch_size=(patch_size, patch_size),
                                                       in_chans=dim,
                                                       out_chans=out_channels,
                                                       hidden_dim=dim,)
        elif unpatch == "vanilla":
            self.unpatchify_layer = Unpatchify(
                grid_size=(self.grid_x, self.grid_y),
                patch_size=(patch_size, patch_size),
                in_dim=dim,
                out_dim=out_channels,
                cond_dim=dim)
        else:
            raise ValueError(f"unpatch type '{unpatch}' not supported")

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Re-apply ICNR init for subpixel conv (self.apply(_basic_init) overwrites it)
        if self.unpatch == "subpixel":
            from modules.layers.unpatchify import ICNR
            weight = ICNR(self.unpatchify_layer.conv.weight,
                          initializer=nn.init.kaiming_normal_,
                          upscale_factor=self.patch_size)
            self.unpatchify_layer.conv.weight.data.copy_(weight)

        # Re-zero-init AdaLN modulation outputs (apply overwrites them)
        for block in self.sa_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # at the end of _init_weights, after the ModulatedNorm loop:                                                                                                                                                                                                                    
        if isinstance(self.unpatchify_layer, PatchInterpolate2D):
            nn.init.constant_(self.unpatchify_layer.adaLN_shift_scale[-1].weight, 0)                                                                                                                                                                                                           
            nn.init.constant_(self.unpatchify_layer.adaLN_shift_scale[-1].bias, 0)

    @torch.no_grad()
    def get_grid(self, nlat, nlon, device):
        if self.with_poles:
            lat = torch.linspace(-math.pi / 2, math.pi / 2, nlat).to(device)
        else:
            lat_end = (nlat - 1) * (2 * math.pi / nlon) / 2
            lat = torch.linspace(-lat_end, lat_end, nlat).to(device)
        lon = torch.linspace(0, 2 * math.pi - (2 * math.pi / nlon), nlon).to(device)
        return lat, lon

    @torch.no_grad()
    def compute_rope_freqs(self, device):
        """Compute 2D RoPE cos/sin frequencies for the patch grid.

        Uses physical lat/lon coordinates at patch centers so the model
        encodes actual geographic position rather than integer indices.

        Returns cached buffers after first call.
        """
        if hasattr(self, '_rope_cos_lat') and self._rope_cos_lat.device == device:
            return (self._rope_cos_lat, self._rope_sin_lat,
                    self._rope_cos_lon, self._rope_sin_lon)

        # Get physical coordinates at padded resolution
        lat, lon = self.get_grid(self.nlat_pad, self.nlon_pad, device)

        # Average pool to patch centers: [nlat_pad] -> [grid_x], [nlon_pad] -> [grid_y]
        lat_patches = lat.reshape(self.grid_x, self.patch_size).mean(dim=1)  # [grid_x]
        lon_patches = lon.reshape(self.grid_y, self.patch_size).mean(dim=1)  # [grid_y]

        # Create 2D grid of patch positions and flatten to sequence
        # lat_grid[i,j] = lat of patch (i,j), lon_grid[i,j] = lon of patch (i,j)
        lat_grid, lon_grid = torch.meshgrid(lat_patches, lon_patches, indexing='ij')  # [grid_x, grid_y]
        lat_seq = lat_grid.reshape(-1)  # [n]
        lon_seq = lon_grid.reshape(-1)  # [n]

        # Compute RoPE frequencies: [n, dim_head//2] -> cos/sin each [1, n, dim_head//2]
        freqs_lat = self.rope_lat(lat_seq.unsqueeze(0))  # [1, n, dim_head//2]
        freqs_lon = self.rope_lon(lon_seq.unsqueeze(0))  # [1, n, dim_head//2]

        self._rope_cos_lat = freqs_lat.cos()
        self._rope_sin_lat = freqs_lat.sin()
        self._rope_cos_lon = freqs_lon.cos()
        self._rope_sin_lon = freqs_lon.sin()

        return (self._rope_cos_lat, self._rope_sin_lat,
                self._rope_cos_lon, self._rope_sin_lon)

    def forward(self, x_noised, cond, t, c_grid):
        """
        Args:
            x_noised: [b, c, nlat, nlon] — interpolant I_t (channel-first from assemble_input)
            cond: [b, c, nlat, nlon] — conditional information (current state + forcings)
            t: [b, n_scalar] — timestep + scalar conditions

        Returns:
            [b, c, nlat, nlon] — predicted velocity (channel-first)
        """
        batch_size = x_noised.shape[0]
        nlat, nlon = self.nlat, self.nlon

        if self.c_grid_embed is not None and c_grid is not None:
            c_grid_emb = self.c_grid_embed(c_grid)  # [b, c_grid_embed_dim, nlat, nlon] (latent res)
            x_input = torch.cat([x_noised, cond, c_grid_emb], dim=1)
        elif c_grid is not None:
            x_input = torch.cat([x_noised, cond, c_grid], dim=1)
        else:
            x_input = torch.cat([x_noised, cond], dim=1)

        # Pad spatial dims to be divisible by patch_size
        # Circular padding in longitude, polar padding in latitude
        x_input = sphere_pad(x_input, padding=(self.pad_lon_left, self.pad_lon_right, self.pad_lat_top, self.pad_lat_bottom))

        # Convert channel-first to channel-last for PatchEmbed: [b, c, h, w] -> [b, h, w, c]
        x_nhwc = x_input.permute(0, 2, 3, 1)

        # Patchify: [b, h, w, c] -> [b, h//p, w//p, dim]
        x = self.patch_embed_main(x_nhwc)

        # Flatten spatial dims for sequence processing: [b, h//p, w//p, dim] -> [b, n, dim]
        x = rearrange(x, 'b ny nx c -> b (ny nx) c')

        # Compute 2D RoPE frequencies for patch grid
        rope_cos_lat, rope_sin_lat, rope_cos_lon, rope_sin_lon = self.compute_rope_freqs(x.device)

        # Timestep embedding
        if len(t.shape) == 1:
            t = t[:, None]

        t_emb = self.t_embedder(t)  # [b, dim]

        # Transformer blocks with RoPE
        for sa_block in self.sa_blocks:
            x = sa_block(x, t_emb, rope_cos_lat, rope_sin_lat, rope_cos_lon, rope_sin_lon)

        x = self.unpatchify_layer(x, t_emb)

        # Convert back to channel-first: [b, h, w, c] -> [b, c, h, w]
        x = x.permute(0, 3, 1, 2)
        
        # Crop back to original spatial dims
        if self.pad_lat > 0 or self.pad_lon > 0:
            x = x[:, :, self.pad_lat_top:self.pad_lat_top + nlat, self.pad_lon_left:self.pad_lon_left + nlon]

        return x
    

class cDiT(nn.Module):
    """
    Architecture:
    - PatchEmbed for main input
    - 2D Rotary Position Embedding (RoPE) on lat/lon patch grid
    - N blocks of DiTBlock (self-attn with AdaLN + RoPE)
    - Unpatchify: dim -> out_channels @ nlat x nlon
    - Zero-initialized output projection
    """

    def __init__(self,
                 in_channels=249,
                 out_channels=249,
                 dim=384,
                 num_heads=8,
                 num_blocks=8,
                 patch_size=2,
                 nlat=180,
                 nlon=360,
                 dropout=0.0,
                 #grid_in_dim = 1,
                 #cond_dim=4,
                 unpatch = "vanilla"):
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
        self.unpatchify = unpatch

        self.grid_x = self.nlat // patch_size
        self.grid_y = self.nlon // patch_size
        self.with_poles = False

        self.patch_embed_main = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_channels,
            hidden_size=dim,
            flatten=False)
        
        #self.embed_grid = SphereConv2d(in_channels = grid_in_dim, out_channels = cond_dim, kernel_size=(3,3))
        #self.embed_calendar = CalendarEmbedding(nlon=nlon, nlat=nlat, embed_channels=cond_dim)

        # 2D RoPE: one RotaryEmbedding per spatial axis
        # Each axis gets half the head dimension
        dim_head = dim // num_heads
        self.rope_lat = RotaryEmbedding(dim_head // 2)
        self.rope_lon = RotaryEmbedding(dim_head // 2)

        # Timestep embedding
        self.t_embedder = TimestepEmbedder(dim, num_conds = 1)

        # Transformer blocks
        sa_blocks = []
        for _ in range(num_blocks):
            sa_blocks.append(DiTBlock(dim, num_heads, mlp_ratio=4, dropout=dropout))

        self.sa_blocks = nn.ModuleList(sa_blocks)

        if self.unpatchify == "interpolate":
            self.unpatchify_layer = PatchInterpolate2D(grid_size=(self.grid_x, self.grid_y),
                                                       patch_size=(patch_size, patch_size),
                                                       in_chans=dim,
                                                       out_chans=out_channels,
                                                       hidden_dim=dim // 2,)
        else:
            self.unpatchify_layer = Unpatchify(
                grid_size=(self.grid_x, self.grid_y),
                patch_size=(patch_size, patch_size),
                in_dim=dim,
                out_dim=out_channels,
                cond_dim=dim)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

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

    @torch.no_grad()
    def compute_rope_freqs(self, device):
        """Compute 2D RoPE cos/sin frequencies for the patch grid.

        Uses physical lat/lon coordinates at patch centers so the model
        encodes actual geographic position rather than integer indices.

        Returns cached buffers after first call.
        """
        if hasattr(self, '_rope_cos_lat') and self._rope_cos_lat.device == device:
            return (self._rope_cos_lat, self._rope_sin_lat,
                    self._rope_cos_lon, self._rope_sin_lon)

        # Get physical coordinates at padded resolution
        lat, lon = self.get_grid(self.nlat, self.nlon, device)

        # Average pool to patch centers: [nlat_pad] -> [grid_x], [nlon_pad] -> [grid_y]
        lat_patches = lat.reshape(self.grid_x, self.patch_size).mean(dim=1)  # [grid_x]
        lon_patches = lon.reshape(self.grid_y, self.patch_size).mean(dim=1)  # [grid_y]

        # Create 2D grid of patch positions and flatten to sequence
        # lat_grid[i,j] = lat of patch (i,j), lon_grid[i,j] = lon of patch (i,j)
        lat_grid, lon_grid = torch.meshgrid(lat_patches, lon_patches, indexing='ij')  # [grid_x, grid_y]
        lat_seq = lat_grid.reshape(-1)  # [n]
        lon_seq = lon_grid.reshape(-1)  # [n]

        # Compute RoPE frequencies: [n, dim_head//2] -> cos/sin each [1, n, dim_head//2]
        freqs_lat = self.rope_lat(lat_seq.unsqueeze(0))  # [1, n, dim_head//2]
        freqs_lon = self.rope_lon(lon_seq.unsqueeze(0))  # [1, n, dim_head//2]

        self._rope_cos_lat = freqs_lat.cos()
        self._rope_sin_lat = freqs_lat.sin()
        self._rope_cos_lon = freqs_lon.cos()
        self._rope_sin_lon = freqs_lon.sin()

        return (self._rope_cos_lat, self._rope_sin_lat,
                self._rope_cos_lon, self._rope_sin_lon)

    def forward(self, x_noised, t, cond,): # grid_cond, scalar_cond):
        """
        Args:
            x_noised: [b, c, nlat, nlon] — interpolant I_t (channel-first from assemble_input)
            cond: [b, c, zlat, zlon] — conditional information. Is the output of the VAE encoder
            grid_cond: [b c nlat nlon] — grid-scale conditional information (SST)
            scalar_cond: [b c] - scalar conditions (e.g. calendar, co2, etc.)
            t: [b, 1] — timestep + scalar conditions

        Returns:
            [b, c, nlat, nlon] — predicted velocity (channel-first)
        """

        #grid_embed = self.embed_grid(grid_cond) # [b, cond_dim, nlat, nlon]
        #scalar_embed = self.embed_calendar(scalar_cond) # [b, cond_dim, nlat, nlon]
        cond = F.interpolate(cond, size=(self.nlat, self.nlon), mode='bilinear', align_corners=False)  # [b, c, nlat, nlon]

        #x_input = torch.cat([x_noised, cond, grid_embed, scalar_embed], dim=1)  # [b, dim, nlat, nlon]
        x_input = torch.cat([x_noised, cond], dim=1)  # [b, dim, nlat, nlon]

        # Convert channel-first to channel-last for PatchEmbed: [b, c, h, w] -> [b, h, w, c]
        x_nhwc = x_input.permute(0, 2, 3, 1)

        # Patchify: [b, h, w, c] -> [b, h//p, w//p, dim]
        x = self.patch_embed_main(x_nhwc)

        # Flatten spatial dims for sequence processing: [b, h//p, w//p, dim] -> [b, n, dim]
        x = rearrange(x, 'b ny nx c -> b (ny nx) c')

        # Compute 2D RoPE frequencies for patch grid
        rope_cos_lat, rope_sin_lat, rope_cos_lon, rope_sin_lon = self.compute_rope_freqs(x.device)

        # Timestep embedding
        if len(t.shape) == 1:
            t = t[:, None]

        t_emb = self.t_embedder(t)  # [b, dim]

        # Transformer blocks with RoPE
        for sa_block in self.sa_blocks:
            x = sa_block(x, t_emb, rope_cos_lat, rope_sin_lat, rope_cos_lon, rope_sin_lon)

        # Unpatchify: [b, n, dim] -> [b, nlat, nlon, dim]
        x = self.unpatchify_layer(x, t_emb)

        # Convert back to channel-first: [b, h, w, c] -> [b, c, h, w]
        x = x.permute(0, 3, 1, 2)

        return x
