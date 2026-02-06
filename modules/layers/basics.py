
# flake8: noqa
# Copied from https://github.com/ai2cm/modulus/commit/22df4a9427f5f12ff6ac891083220e7f2f54d229
# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.checkpoint import checkpoint
from torch_harmonics import *
from modules.layers.contractions import *
import typing
from einops import rearrange

class ComplexReLU(nn.Module):
    """
    Complex-valued variants of the ReLU activation function
    """

    def __init__(self, negative_slope=0.0, mode="real", bias_shape=None, scale=1.0):
        super(ComplexReLU, self).__init__()

        # store parameters
        self.mode = mode
        if self.mode in ["modulus", "halfplane"]:
            if bias_shape is not None:
                self.bias = nn.Parameter(
                    scale * torch.ones(bias_shape, dtype=torch.float32)
                )
            else:
                self.bias = nn.Parameter(scale * torch.ones((1), dtype=torch.float32))
        else:
            self.bias = 0

        self.negative_slope = negative_slope
        self.act = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.mode == "cartesian":
            zr = torch.view_as_real(z)
            za = self.act(zr)
            out = torch.view_as_complex(za)

        elif self.mode == "modulus":
            zabs = torch.sqrt(torch.square(z.real) + torch.square(z.imag))
            out = torch.where(zabs + self.bias > 0, (zabs + self.bias) * z / zabs, 0.0)
            # out = self.act(zabs - self.bias) * torch.exp(1.j * z.angle())

        elif self.mode == "halfplane":
            # bias is an angle parameter in this case
            modified_angle = torch.angle(z) - self.bias
            condition = torch.logical_and(
                (0.0 <= modified_angle), (modified_angle < torch.pi / 2.0)
            )
            out = torch.where(condition, z, self.negative_slope * z)

        elif self.mode == "real":
            zr = torch.view_as_real(z)
            outr = zr.clone()
            outr[..., 0] = self.act(zr[..., 0])
            out = torch.view_as_complex(outr)

        else:
            raise NotImplementedError

        return out


class ComplexActivation(nn.Module):
    """
    A module implementing complex-valued activation functions.
    The module supports different modes of operation, depending on how
    the complex numbers are treated for the activation function:
    - "cartesian": the activation function is applied separately to the
       real and imaginary parts of the complex input.
    - "modulus": the activation function is applied to the modulus of
       the complex input, after adding a learnable bias.
    - any other mode: the complex input is returned as-is (identity operation).
    """

    def __init__(self, activation, mode="cartesian", bias_shape=None):
        super(ComplexActivation, self).__init__()

        # store parameters
        self.mode = mode
        if self.mode == "modulus":
            if bias_shape is not None:
                self.bias = nn.Parameter(torch.zeros(bias_shape, dtype=torch.float32))
            else:
                self.bias = nn.Parameter(torch.zeros((1), dtype=torch.float32))
        else:
            bias = torch.zeros((1), dtype=torch.float32)
            self.register_buffer("bias", bias)

        # real valued activation
        self.act = activation

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.mode == "cartesian":
            zr = torch.view_as_real(z)
            za = self.act(zr)
            out = torch.view_as_complex(za)
        elif self.mode == "modulus":
            zabs = torch.sqrt(torch.square(z.real) + torch.square(z.imag))
            out = self.act(zabs + self.bias) * torch.exp(1.0j * z.angle())
        else:
            # identity
            out = z

        return out


@torch.jit.script
def drop_path(
    x: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:  # pragma: no cover
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).
    This is the same as the DropConnect impl for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in
    a separate paper. See discussion:
        https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956
    We've opted for changing the layer and argument names to 'drop path' rather than
    mix DropConnect as a layer name and use 'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2d ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual
    blocks).
    """

    def __init__(self, drop_prob=None):  # pragma: no cover
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):  # pragma: no cover
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    Divides the input image into patches and embeds them into a specified dimension
    using a convolutional layer.
    """

    def __init__(
        self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768
    ):  # pragma: no cover
        super(PatchEmbed, self).__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):  # pragma: no cover
        # gather input
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # new: B, C, H*W
        x = self.proj(x).flatten(2)
        return x


class MLP(nn.Module):
    """
    Basic CNN with support for gradient checkpointing
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        output_bias=True,
        drop_rate=0.0,
        checkpointing=0,
    ):  # pragma: no cover
        super(MLP, self).__init__()
        self.checkpointing = checkpointing
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        fc1 = nn.Conv2d(in_features, hidden_features, 1, bias=True)
        act = act_layer()
        fc2 = nn.Conv2d(hidden_features, out_features, 1, bias=output_bias)
        if drop_rate > 0.0:
            drop = nn.Dropout(drop_rate)
            self.fwd = nn.Sequential(fc1, act, drop, fc2, drop)
        else:
            self.fwd = nn.Sequential(fc1, act, fc2)

        # by default, all weights are shared

    @torch.jit.ignore
    def checkpoint_forward(self, x):  # pragma: no cover
        """Forward method with support for gradient checkpointing"""
        return checkpoint(self.fwd, x)

    def forward(self, x):  # pragma: no cover
        if self.checkpointing >= 2:
            return self.checkpoint_forward(x)
        else:
            return self.fwd(x)

class GroupNorm(nn.Module):
    # group norm with channel at the last dimension
    def __init__(self, num_groups, num_channels,
                 domain_wise=False,
                 eps=1e-8, affine=True):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.domain_wise = domain_wise
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(num_channels), requires_grad=True)

    def forward(self, x):
        # b h w c
        b, c = x.shape[0], x.shape[-1]
        if self.domain_wise:
            x = rearrange(x, 'b ... (g c) -> b g (... c)', g=self.num_groups)
        else:
            x = rearrange(x, 'b ... (g c) -> b ... g c', g=self.num_groups)

        x = (x - x.mean(dim=-1, keepdim=True)) / (x.var(dim=-1, keepdim=True) + self.eps).sqrt()
        if self.domain_wise:
            x = rearrange(x, 'b g (... c) -> b ... (g c)',
                          g=self.num_groups)
        else:
            x = rearrange(x, 'b ... g c -> b ... (g c)',
                          g=self.num_groups)
        if self.affine:
            x = x * self.weight + self.bias
        return x


class RMSNorm(torch.nn.Module):
    def __init__(self,
                 dim: int,
                 eps: float = 1e-6,
                 channel_first: bool = False):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.channel_first = channel_first

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        if not self.channel_first:
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        else:
            return x * torch.rsqrt(x.permute(0, 2, 3, 1).pow(2).mean(-1).unsqueeze(1) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        if self.channel_first:
            return output * self.weight.view(1, -1, 1, 1)
        else:
            return output * self.weight


def get_activation(activation: str, **kwargs):
    if activation == "gelu":
        activation_fn = nn.GELU()
    elif activation == "relu":
        activation_fn = nn.ReLU()
    elif activation == "tanh":
        activation_fn = nn.Tanh()
    elif activation == "silu":
        activation_fn = nn.SiLU()
    else:
        raise ValueError("Please specify different activation function")
    return activation_fn


# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add_scale(
        x: torch.Tensor,
        bias: typing.Optional[torch.Tensor],
        scale: torch.Tensor,
        residual: typing.Optional[torch.Tensor],
        prob: float,
        training: bool) -> torch.Tensor:
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)

    if residual is not None:
        out = residual + out
    return out


def get_bias_dropout_add_scale(training):
    def _bias_dropout_add(x, bias, scale, residual, prob):
        return bias_dropout_add_scale(
            x, bias, scale, residual, prob, training)

    return _bias_dropout_add


# function overload
def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
        x: torch.Tensor,
        bias: typing.Optional[torch.Tensor],
        scale: torch.Tensor,
        residual: typing.Optional[torch.Tensor],
        prob: float) -> torch.Tensor:
    return bias_dropout_add_scale(
        x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
        x: torch.Tensor,
        bias: typing.Optional[torch.Tensor],
        scale: torch.Tensor,
        residual: typing.Optional[torch.Tensor],
        prob: float) -> torch.Tensor:
    return bias_dropout_add_scale(
        x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
    return modulate(x, shift, scale)


def unpatchify(x, patch_size, channels, h=None, w=None, t=1):
    """
    x: (N, S, patch_size**2 * C), s is the number of patches
    out: (N, H, W, C), H = W = sqrt(S)*patch_size
    """
    b = x.shape[0]
    p = patch_size
    c = channels
    s = x.shape[1]

    if h is not None or w is not None:
        if h is None:
            h = s // w
        elif w is None:
            w = s // h
    else: # square image
        h = w = int((s//t) ** 0.5)
    assert h * w == s//t
    if t == 1:
        out = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                        h=h, w=w, p1=p, p2=p, c=c)
        return out
    else:
        out = rearrange(x, 'b (t h w) (p1 p2 p3 c) -> b (t p1) c (h p2) (w p3)',
                        h=h, w=w, p1=p, p2=p, p3=p, c=c, t=t)
        return out


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    # basically a layer norm with out bias
    def __init__(self, dim, force_fp32=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim
        self.force_fp32 = force_fp32

    def forward(self, x):

        with torch.cuda.amp.autocast(enabled=False):
            if self.force_fp32:
                x = x.float()
            x = F.layer_norm(x, [self.dim])
        return x * self.weight[None, None, :]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out),
        x.view(-1, dim_in),
        W.T,
        alpha=residual_scale).view(*x.shape[:-1], dim_out)


class RealFFT2(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None):  # pragma: no cover
        super(RealFFT2, self).__init__()

        # use local FFT here
        self.fft_handle = torch.fft.rfft2

        self.nlat = nlat
        self.nlon = nlon
        self.lmax = lmax or self.nlat
        self.mmax = mmax or self.nlon // 2 + 1

        self.truncate = True
        if (self.lmax == self.nlat) and (self.mmax == (self.nlon // 2 + 1)):
            self.truncate = False

        # self.num_batches = 1
        assert self.lmax % 2 == 0

    def forward(self, x):  # pragma: no cover
        y = self.fft_handle(x, (self.nlat, self.nlon), (-2, -1), "ortho")

        if self.truncate:
            y = torch.cat(
                (
                    y[..., : math.ceil(self.lmax / 2), : self.mmax],
                    y[..., -math.floor(self.lmax / 2) :, : self.mmax],
                ),
                dim=-2,
            )

        return y


class InverseRealFFT2(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None):  # pragma: no cover
        super(InverseRealFFT2, self).__init__()

        # use local FFT here
        self.ifft_handle = torch.fft.irfft2

        self.nlat = nlat
        self.nlon = nlon
        self.lmax = lmax or self.nlat
        self.mmax = mmax or self.nlon // 2 + 1

    def forward(self, x):  # pragma: no cover
        out = self.ifft_handle(x, (self.nlat, self.nlon), (-2, -1), "ortho")

        return out

class SpectralAttention2d(nn.Module):
    """
    2d Spectral Attention layer
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        sparsity_threshold=0.0,
        hidden_size_factor=2,
        use_complex_network=True,
        use_complex_kernels=False,
        complex_activation="real",
        bias=False,
        spectral_layers=1,
        drop_rate=0.0,
    ):  # pragma: no cover
        super(SpectralAttention2d, self).__init__()

        self.embed_dim = embed_dim
        self.sparsity_threshold = sparsity_threshold
        self.hidden_size = int(hidden_size_factor * self.embed_dim)
        self.scale = 0.02
        self.spectral_layers = spectral_layers
        self.mul_add_handle = (
            None if use_complex_kernels else compl_muladd2d_fwd
        )
        self.mul_handle = None if use_complex_kernels else compl_mul2d_fwd

        self.modes_lat = forward_transform.lmax
        self.modes_lon = forward_transform.mmax

        # only storing the forward handle to be able to call it
        self.forward_transform = forward_transform.forward
        self.inverse_transform = inverse_transform.forward

        assert inverse_transform.lmax == self.modes_lat
        assert inverse_transform.mmax == self.modes_lon

        # weights
        w = [self.scale * torch.randn(self.embed_dim, self.hidden_size, 2)]
        # w = [self.scale * torch.randn(self.embed_dim + 2*self.embed_freqs, self.hidden_size, 2)]
        # w = [self.scale * torch.randn(self.embed_dim + 4*self.embed_freqs, self.hidden_size, 2)]
        for l in range(1, self.spectral_layers):
            w.append(self.scale * torch.randn(self.hidden_size, self.hidden_size, 2))
        self.w = nn.ParameterList(w)

        if bias:
            self.b = nn.ParameterList(
                [
                    self.scale * torch.randn(self.hidden_size, 1, 2)
                    for _ in range(self.spectral_layers)
                ]
            )

        self.wout = nn.Parameter(
            self.scale * torch.randn(self.hidden_size, self.embed_dim, 2)
        )

        self.drop = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

        self.activation = ComplexReLU(
            mode=complex_activation, bias_shape=(self.hidden_size, 1, 1)
        )

    def forward_mlp(self, xr):  # pragma: no cover
        """forward method for the MLP part of the network"""
        for l in range(self.spectral_layers):
            if hasattr(self, "b"):
                xr = self.mul_add_handle(
                    xr, self.w[l].to(xr.dtype), self.b[l].to(xr.dtype)
                )
            else:
                xr = self.mul_handle(xr, self.w[l].to(xr.dtype))
            xr = torch.view_as_complex(xr)
            xr = self.activation(xr)
            xr = self.drop(xr)
            xr = torch.view_as_real(xr)

        xr = self.mul_handle(xr, self.wout)

        return xr

    def forward(self, x):  # pragma: no cover
        dtype = x.dtype
        # x = x.to(torch.float32)

        # FWD transform
        with amp.autocast(enabled=False):
            x = x.to(torch.float32)
            x = x.contiguous()
            x = self.forward_transform(x)
            x = torch.view_as_real(x)

        # MLP
        x = self.forward_mlp(x)

        # BWD transform
        with amp.autocast(enabled=False):
            x = torch.view_as_complex(x)
            x = x.contiguous()
            x = self.inverse_transform(x)
            x = x.to(dtype)

        return x


class SpectralAttentionS2(nn.Module):
    """
    geometrical Spectral Attention layer
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        sparsity_threshold=0.0,
        hidden_size_factor=2,
        use_complex_network=True,
        use_complex_kernels=False,
        complex_activation="real",
        bias=False,
        spectral_layers=1,
        drop_rate=0.0,
    ):  # pragma: no cover
        super(SpectralAttentionS2, self).__init__()

        self.embed_dim = embed_dim
        self.sparsity_threshold = sparsity_threshold
        self.hidden_size = int(hidden_size_factor * self.embed_dim)
        self.scale = 0.02
        # self.mul_add_handle = compl_muladd1d_fwd_c if use_complex_kernels else compl_muladd1d_fwd
        self.mul_add_handle = (
            None if use_complex_kernels else compl_muladd2d_fwd
        )
        # self.mul_handle = compl_mul1d_fwd_c if use_complex_kernels else compl_mul1d_fwd
        self.mul_handle = None if use_complex_kernels else compl_mul2d_fwd
        self.spectral_layers = spectral_layers

        self.modes_lat = forward_transform.lmax
        self.modes_lon = forward_transform.mmax

        # only storing the forward handle to be able to call it
        self.forward_transform = forward_transform.forward
        self.inverse_transform = inverse_transform.forward

        assert inverse_transform.lmax == self.modes_lat
        assert inverse_transform.mmax == self.modes_lon

        # weights
        w = [self.scale * torch.randn(self.embed_dim, self.hidden_size, 2)]
        # w = [self.scale * torch.randn(self.embed_dim + 4*self.embed_freqs, self.hidden_size, 2)]
        for l in range(1, self.spectral_layers):
            w.append(self.scale * torch.randn(self.hidden_size, self.hidden_size, 2))
        self.w = nn.ParameterList(w)

        if bias:
            self.b = nn.ParameterList(
                [
                    self.scale * torch.randn(2 * self.hidden_size, 1, 1, 2)
                    for _ in range(self.spectral_layers)
                ]
            )

        self.wout = nn.Parameter(
            self.scale * torch.randn(self.hidden_size, self.embed_dim, 2)
        )

        self.drop = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

        self.activation = ComplexReLU(
            mode=complex_activation, bias_shape=(self.hidden_size, 1, 1)
        )

    def forward_mlp(self, xr):  # pragma: no cover
        """forward method for the MLP part of the network"""
        for l in range(self.spectral_layers):
            if hasattr(self, "b"):
                xr = self.mul_add_handle(
                    xr, self.w[l].to(xr.dtype), self.b[l].to(xr.dtype)
                )
            else:
                xr = self.mul_handle(xr, self.w[l].to(xr.dtype))
            xr = torch.view_as_complex(xr)
            xr = self.activation(xr)
            xr = self.drop(xr)
            xr = torch.view_as_real(xr)

        # final MLP
        xr = self.mul_handle(xr, self.wout)

        return xr

    def forward(self, x):  # pragma: no cover
        dtype = x.dtype
        # x = x.to(torch.float32)

        # FWD transform
        with amp.autocast(enabled=False):
            x = x.to(torch.float32)
            x = x.contiguous()
            x = self.forward_transform(x)
            x = torch.view_as_real(x)

        # MLP
        x = self.forward_mlp(x)

        # BWD transform
        with amp.autocast(enabled=False):
            x = torch.view_as_complex(x)
            x = x.contiguous()
            x = self.inverse_transform(x)
            x = x.to(dtype)

        return x
