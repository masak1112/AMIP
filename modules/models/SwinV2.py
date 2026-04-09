import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

# ----------------------------------------------------------------------------
# Utility Functions


def window_partition(x: torch.Tensor, window_size: tuple[int, int]):
    """(B, H, W, C) -> (num_windows*B, window_size, window_size, C)"""
    B, H, W, C = x.shape
    x = x.view(
        B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C
    )
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size[0], window_size[1], C)
    )
    return windows


def window_reverse(
    windows: torch.Tensor, window_size: tuple[int, int], img_size: tuple[int, int]
):
    """(num_windows * B, window_size[0], window_size[1], C) -> (B, H, W, C)"""
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(
        -1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x


def get_shift_window_mask(
    grid_size: tuple[int, int],
    window_size: tuple[int, int],
    shift_size: tuple[int, int],
):
    """Compute attention mask for shifted windows on a lat/lon grid.

    Longitude is periodic, so shifted boundary windows along that axis contain
    genuinely adjacent tokens and need no masking.  Only latitude boundaries
    require masking after the cyclic shift.

    Returns:
        attn_mask: (n_windows, 1, win_h*win_w, win_h*win_w) or None if no
        latitude shift is needed.
    """
    H, W = grid_size
    wh, ww = window_size
    sh, _sw = shift_size

    if sh == 0:
        return None

    img_mask = torch.zeros((1, H, W, 1))

    lat_slices = (slice(0, -wh), slice(-wh, -sh), slice(-sh, None))

    cnt = 0
    for lat in lat_slices:
        img_mask[:, lat, :, :] = cnt
        cnt += 1

    mask_windows = window_partition(img_mask, window_size)  # (n_win, wh, ww, 1)
    mask_windows = mask_windows.view(-1, wh * ww)  # (n_win, wh*ww)

    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
        attn_mask == 0, 0.0
    )

    return attn_mask.unsqueeze(1)  # (n_win, 1, wh*ww, wh*ww)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000):
    """Sinusoidal timestep embeddings."""
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=t.dtype) / half
    ).to(device=t.device)
    args = t[:, None].to(t.dtype) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    embedding = (
        embedding.reshape(embedding.shape[0], 2, -1).flip(1).reshape(*embedding.shape)
    )  # flip sin/cos as done with edm

    return embedding


# ----------------------------------------------------------------------------
# Swin Modules


class LatentEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.l1 = nn.Linear(dim, dim, bias=True)
        self.l2 = nn.Linear(dim, dim, bias=True)

    def forward(self, emb):
        return F.silu(self.l2(F.silu(self.l1(emb))))


class ModulatedNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps)
        self.modulation = nn.Linear(dim, dim * 2, bias=True)

    def forward(self, x, t):
        x = self.norm(x)  # b, n, d
        scale, shift = self.modulation(t).chunk(2, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FeedForward(nn.Module):
    """SwiGLU FeedForward"""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.norm = ModulatedNorm(dim)
        self.w1 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x, t):
        gate, up_proj = self.w1(x).chunk(2, dim=-1)
        x = self.w2(F.silu(gate) * up_proj)
        x = self.norm(x, t)  # new: post-norm
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads, head_dim, flash=True):
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads
        self.flash = flash
        self.norm = ModulatedNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.wo = nn.Linear(inner_dim, dim, bias=False)

        self.scale = nn.Parameter(torch.log(10 * torch.ones(1, heads, 1, 1)))

    def forward(self, x, t, mask=None, jvp: bool = False):
        qkv = self.to_qkv(x)
        qkv = rearrange(qkv, "b n (h d) -> b h n d", h=self.heads)
        q, k, v = qkv.chunk(3, dim=-1)

        q = (
            F.normalize(q, dim=-1)
            * torch.clamp(self.scale, max=math.log(1.0 / 0.01)).exp()
        )
        k = F.normalize(k, dim=-1)

        if self.flash and not jvp:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
        else:
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn + mask
            attn = attn.softmax(dim=-1)
            x = attn @ v

        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.wo(x)
        x = self.norm(x, t)  # new: post-norm
        return x


class SwinTransformer(nn.Module):
    def __init__(
        self,
        depth,
        dim,
        heads,
        window_size,
        grid_size,
        shift_size,
        flash,
    ):
        super().__init__()

        self.window_size = window_size
        self.grid_size = grid_size
        self.shift_size = shift_size

        assert grid_size[0] % window_size[0] == 0 and grid_size[1] % window_size[1] == 0, (
            f"grid_size {grid_size} must be divisible by window_size {window_size}"
        )

        head_dim = dim // heads
        mlp_dim = int(8 / 3.0 * dim)

        self.layers = nn.Sequential(
            *[
                nn.ModuleList(
                    [
                        Attention(dim, heads, head_dim, flash),
                        FeedForward(dim, mlp_dim),
                    ]
                )
                for _ in range(depth)
            ]
        )

        attn_mask = get_shift_window_mask(grid_size, window_size, shift_size)
        if attn_mask is not None:
            self.register_buffer("attn_mask", attn_mask)
        else:
            self.attn_mask = None

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, jvp: bool = False
    ) -> torch.Tensor:
        sh, sw = self.shift_size
        do_shift: bool = any(self.shift_size)

        # expand t to match the number of windows
        repeat_factor = (self.grid_size[0] // self.window_size[0]) * (
            self.grid_size[1] // self.window_size[1]
        )
        t_expanded = t.repeat_interleave(repeat_factor, dim=0)  # num_windows * b, d

        for i, (attn, ff) in enumerate(self.layers):  # type:ignore  ??
            xp = x

            x = x.view(-1, self.grid_size[0], self.grid_size[1], x.shape[-1])
            B, h, w, d = x.shape

            use_shift = do_shift and i % 2 != 0

            # cyclic shift
            if use_shift:
                x = torch.roll(x, shifts=(-sh, -sw), dims=(1, 2))

            # partition windows
            x = window_partition(x, self.window_size)
            x = x.view(-1, self.window_size[0] * self.window_size[1], d)

            # attention mask for shifted layers (latitude boundaries only)
            mask = None
            if use_shift and self.attn_mask is not None:
                mask = self.attn_mask.repeat(B, 1, 1, 1)

            x = attn(x, t_expanded, mask=mask, jvp=jvp)  # num_windows * b, n, d

            # merge windows
            x = x.view(-1, self.window_size[0], self.window_size[1], d)
            x = window_reverse(x, self.window_size, (h, w))

            # reverse cyclic shift
            if use_shift:
                x = torch.roll(x, shifts=(sh, sw), dims=(1, 2))
            x = x.view(-1, h * w, d)

            x = xp + x
            x = x + ff(x, t)

        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, dim):
        super().__init__()
        self.patch_size = p1, p2 = patch_size
        self.emb = nn.Linear(in_channels * p1 * p2, dim)

    def forward(self, x):
        x = rearrange(
            x,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
        )
        return self.emb(x)


class OutputHead(nn.Module):
    def __init__(self, dim, out_channels, patch_size, grid_size):
        super().__init__()
        p1, p2 = patch_size
        gh, gw = grid_size

        self.head = nn.Sequential(
            nn.Linear(dim, out_channels * p1 * p2, bias=False),  # b, n, c*p1*p2
            Rearrange(
                "b (h w) (c p1 p2) -> b c (h p1) (w p2)", p1=p1, p2=p2, h=gh, w=gw
            ),
        )

    def forward(self, x):
        return self.head(x)


# ----------------------------------------------------------------------------
# Swin Transformer Class


class SwinV2(nn.Module):
    def __init__(
        self,
        img_resolution,
        in_channels: int,
        out_channels: int,
        window_size,
        shift_size,
        patch_size,
        depth: int = 6,
        head_depth: int = 2,
        dim: int = 512,
        heads: int = 12,
        flash: bool = True,
    ):
        super().__init__()

        image_height, image_width = img_resolution
        patch_height, patch_width = patch_size
        grid_size = gh, gw = (image_height // patch_height, image_width // patch_width)

        self.pos_embed = nn.Parameter(torch.randn(1, gh * gw, dim) * 0.02)
        self.patch_embed = PatchEmbedding(in_channels, patch_size, dim)
        self.t_embed = LatentEmbedding(dim)
        self.h_embed = LatentEmbedding(dim)

        self.transformer = SwinTransformer(
            depth,
            dim,
            heads,
            window_size,
            grid_size,
            shift_size,
            flash=flash,
        )

        self.u_transformer = SwinTransformer(
            head_depth,
            dim,
            heads,
            window_size,
            grid_size,
            shift_size,
            flash=flash,
        )

        self.v_transformer = SwinTransformer(
            head_depth,
            dim,
            heads,
            window_size,
            grid_size,
            shift_size,
            flash=flash,
        )

        self.u_head = OutputHead(dim, out_channels, patch_size, grid_size)
        self.v_head = OutputHead(dim, out_channels, patch_size, grid_size)

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if "modulation" in name or "head" in name:  # start with layer norm
                    nn.init.zeros_(m.weight)
                else:
                    nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
        cond: torch.Tensor,
        jvp: bool = True, # defaults to true. Set to True for training, False for inference to save memory.
    ):  
        x = torch.cat([x, cond], dim = 1) 

        x = self.patch_embed(x)  # b, n, d
        x = x + self.pos_embed  # new: nersc swinv2

        if t.dim() == 0 or (t.dim() == 1 and t.size(0) == 1):
            t = t.repeat(x.size(0))

        t = self.t_embed(timestep_embedding(t, x.size(2)))  # b, d
        h = self.h_embed(timestep_embedding(h, x.size(2)))

        t = t + h 

        x = self.transformer(x, t, jvp)  # b, n, d
        
        u = self.u_transformer(x, t, jvp)  # b, n, d
        u = self.u_head(u)  # b, c, h, w

        if jvp:
            v = self.v_transformer(x, t, jvp)
            v = self.v_head(v)
        else:
            v = None  # instantenous velocity not used in inference.

        return u, v