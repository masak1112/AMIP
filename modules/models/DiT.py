import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa N812
import torch.utils.checkpoint as gradient_checkpoint
from einops import rearrange
from timm.layers.mlp import SwiGLU

from modules.layers.arches_layers import (
    CondBasicLayer,
    DCDownSample,
    LinVert,
    Mlp,
    DCUpSample,
    ICNR_init,
    TimestepEmbedder
)


class WeatherEncodeDecodeLayer(nn.Module):
    """
    gathers layers for the encoder and decoder
    """

    def __init__(
        self,
        img_size=(26, 180, 360),
        emb_dim=192,
        out_emb_dim=2 * 192,  
        patch_size=(2, 2, 2),
        surface_ch=6,
        level_ch=8,
        forcing_ch=3,
        invariant_ch=2,
        diagnostic_ch=9,
        encode_noise=True
    ) -> None:
        super().__init__()
        
        self.img_size = img_size
        self.emb_dim = emb_dim
        self.patch_size = patch_size
        self.surface_ch = surface_ch
        self.level_ch = level_ch
        self.forcing_ch = forcing_ch
        self.invariant_ch = invariant_ch
        self.diagnostic_ch = diagnostic_ch
        self.encode_noise = encode_noise

        surface_ch_in = surface_ch + forcing_ch + invariant_ch
        level_ch_in = level_ch 

        if self.encode_noise:
            surface_ch_in += diagnostic_ch + surface_ch
            level_ch_in += level_ch

        self.level_proj = nn.Conv3d(
            level_ch_in, emb_dim, kernel_size=patch_size, stride=patch_size
        )
        self.surface_proj = nn.Conv2d(
            surface_ch_in, emb_dim, kernel_size=patch_size[1:], stride=patch_size[1:]
        )

        self.surface_deconv = nn.Conv2d(
            out_emb_dim,
            (surface_ch + diagnostic_ch) * patch_size[-1] ** 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=0,
        )
        self.level_deconv = nn.Conv2d(
            out_emb_dim // 2,
            level_ch * patch_size[-1] ** 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=0,
        )
        self.pixelshuffle = nn.PixelShuffle(patch_size[-1])
        ICNR_init(
            self.surface_deconv.weight,
            initializer=nn.init.kaiming_normal_,
            upscale_factor=patch_size[-1],
        )
        ICNR_init(
            self.level_deconv.weight,
            initializer=nn.init.kaiming_normal_,
            upscale_factor=patch_size[-1],
        )

    def encode(self, surface, multilevel, forcing, invariants,
               surface_noised=None, multi_noised=None, diag_noised=None):
        """
        surface: B, nlat, nlon, surface_ch
        multilevel: B, nlevel, nlat, nlon, level_ch
        forcing: B, nlat, nlon, forcing_ch
        invariants: B, nlat, nlon, invariant_ch
        """

        surface = rearrange(surface, "b nlat nlon c -> b c nlat nlon")
        multilevel = rearrange(
            multilevel, "b nlevel nlat nlon c -> b c nlevel nlat nlon"
        )
        forcing = rearrange(forcing, "b nlat nlon c -> b c nlat nlon")
        invariants = rearrange(invariants, "b nlat nlon c -> b c nlat nlon")
        
        surface = torch.cat([surface, forcing, invariants], dim=1) # b (surface_ch + forcing_ch + invariant_ch) nlat nlon
        
        if self.encode_noise:
            surface_noised = rearrange(surface_noised, "b nlat nlon c -> b c nlat nlon")
            diag_noised = rearrange(diag_noised, "b nlat nlon c -> b c nlat nlon")
            multi_noised = rearrange(multi_noised, "b nlevel nlat nlon c -> b c nlevel nlat nlon")

            surface = torch.cat([surface, surface_noised, diag_noised], dim=1) # b (surface_ch + forcing_ch + invariant_ch + surface_noised_ch + diagnostic_ch) nlat nlon
            multilevel = torch.cat([multilevel, multi_noised], dim=1) # b (level_ch + multi_noised_ch) nlevel nlat nlon

        # patchify
        surface = self.surface_proj(surface) # b emb_dim zlat zlon
        level = self.level_proj(multilevel) # b emb_dim zlevel zlat zlon

        x = torch.concat([surface.unsqueeze(2), level], dim=2) # b emb_dim (1 + zlevel) zlat zlon
        return x

    def decode(self, x):
        # x: b, emb_dim, zlevel+1, zlat, zlon

        surface, level = x[:, :, 0], x[:, :, 1:]

        output_surface = self.surface_deconv(surface) # b, surface_ch * r^2, zlat, zlon
        output_surface = self.pixelshuffle(output_surface) # b, surface_ch, lat, lon
        output_surface, output_diagnostic = output_surface[:, :self.surface_ch], output_surface[:, self.surface_ch:]  

        # b c/2 2 zlevel zlat zlon -> b c/2 2*zlevel zlat zlon
        level = level.reshape(level.shape[0], level.shape[1] // 2, 2, *level.shape[2:]).flatten(2, 3)
        level = level.movedim(-3, 1).flatten(0, 1) # b*2*zlevel, c/2, zlat, zlon

        output_level = self.level_deconv(level) # b*2*zlevel, level_ch * r^2, zlat, zlon
        output_level = self.pixelshuffle(output_level) # b*2*zlevel, level_ch, lat, lon
        output_level = output_level.reshape(-1, self.img_size[0], *output_level.shape[1:]).movedim(
            1, -3
        ) # b, level_ch, nlevel, lat, lon

        output_surface = rearrange(output_surface, "b c nlat nlon -> b nlat nlon c")
        output_level = rearrange(output_level, "b c nlevel nlat nlon -> b nlevel nlat nlon c")
        output_diagnostic = rearrange(output_diagnostic, "b c nlat nlon -> b nlat nlon c")

        return output_surface, output_level, output_diagnostic


class ArchesDiT(nn.Module):
    def __init__(
        self,
        encode_decode_params: dict,
        tensor_size=(14, 90, 180),
        emb_dim=192,
        cond_dim=256,  # dim of the conditioning
        num_heads=(6, 12, 12, 6),
        window_size=(1, 6, 10),
        droppath_coeff=0.0,
        depth_multiplier=2,
        dropout=0.0,
        mlp_ratio=4.0,
        use_skip=True,
        first_interaction_layer="linear",
        gradient_checkpointing=False,
        mlp_layer="swiglu",
        **kwargs,
    ):
        super().__init__()
        self.use_skip = use_skip
        self.gradient_checkpointing = gradient_checkpointing
        self.first_interaction_layer = first_interaction_layer

        self.encode_decode = WeatherEncodeDecodeLayer(**encode_decode_params)
        self.zdim = tensor_size[0]
        
        drop_path = np.linspace(
            0, droppath_coeff / depth_multiplier, self.zdim * depth_multiplier
        ).tolist()

        self.layer1_shape = tensor_size[1:]

        self.layer2_shape = (self.layer1_shape[0] // 2, self.layer1_shape[1] // 2)

        if first_interaction_layer == "linear":
            self.interaction_layer = LinVert(in_features=emb_dim,
                                             n_cols = self.zdim)

        layer_args = dict(
            cond_dim=cond_dim,
            window_size=window_size,
            act_layer=nn.GELU,
            drop=dropout,
            mlp_layer=Mlp,
            mlp_ratio=mlp_ratio,
        )

        if mlp_layer == "swiglu":
            layer_args["mlp_ratio"] = mlp_ratio * 2 / 3
            layer_args["mlp_layer"] = SwiGLU

        self.layer1 = CondBasicLayer(
            dim=emb_dim,
            input_resolution=(self.zdim, *self.layer1_shape),
            depth=2 * depth_multiplier,
            num_heads=num_heads[0],
            drop_path=drop_path[: 2 * depth_multiplier],
            **layer_args,
            **kwargs,
        )
        self.downsample = DCDownSample(
            in_dim=emb_dim,
            out_dim=emb_dim * 2,
            input_resolution=(self.zdim, *self.layer1_shape),
            output_resolution=(self.zdim, *self.layer2_shape),
        )
        self.layer2 = CondBasicLayer(
            dim=emb_dim * 2,
            input_resolution=(self.zdim, *self.layer2_shape),
            depth=6 * depth_multiplier,
            num_heads=num_heads[1],
            drop_path=drop_path[2 * depth_multiplier :],
            **layer_args,
            **kwargs,
        )
        self.layer3 = CondBasicLayer(
            dim=emb_dim * 2,
            input_resolution=(self.zdim, *self.layer2_shape),
            depth=6 * depth_multiplier,
            num_heads=num_heads[2],
            drop_path=drop_path[2 * depth_multiplier :],
            **layer_args,
            **kwargs,
        )
        self.upsample = DCUpSample(
            emb_dim * 2, emb_dim, (self.zdim, *self.layer2_shape), (self.zdim, *self.layer1_shape)
        )
        out_dim = emb_dim if not self.use_skip else 2 * emb_dim
        self.layer4 = CondBasicLayer(
            dim=out_dim,
            input_resolution=(self.zdim, *self.layer1_shape),
            depth=2 * depth_multiplier,
            num_heads=num_heads[3],
            drop_path=drop_path[: 2 * depth_multiplier],
            **layer_args,
            **kwargs,
        )

        self.cond_embedders = nn.ModuleList([
            TimestepEmbedder(cond_dim),
            TimestepEmbedder(cond_dim),
            TimestepEmbedder(cond_dim),
        ])

    def forward(self, surface, multi, forcing, invariant, cond_emb, 
                surface_noised=None, multi_noised=None, diag_noised=None):
        
        # cond_emb in shape (b, 3)
        cond_emb = [emb(cond_emb[:, i]) for i, emb in enumerate(self.cond_embedders)]
        cond_emb = torch.stack(cond_emb, dim=0) # 3, b, cond_dim
        cond_emb = torch.sum(cond_emb, dim=0) # b, cond_dim
        
        x = self.encode_decode.encode(surface, multi, forcing, invariant,
                                      surface_noised, multi_noised, diag_noised) 

        B, C, Pl, Lat, Lon = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2) # B, N, C

        if self.first_interaction_layer:
            x = self.interaction_layer(x)

        x = self.layer1(x, cond_emb)

        skip = x
        x = self.downsample(x)

        x = self.layer2(x, cond_emb)

        if self.gradient_checkpointing:
            x = gradient_checkpoint.checkpoint(self.layer3, x, cond_emb, use_reentrant=False)
        else:
            x = self.layer3(x, cond_emb)

        x = self.upsample(x)
        if self.use_skip and skip is not None:
            x = torch.concat([x, skip], dim=-1)
        x = self.layer4(x, cond_emb)

        output = x
        output = output.transpose(1, 2).reshape(output.shape[0], -1, self.zdim, *self.layer1_shape)

        output_surface, output_level, output_diagnostic = self.encode_decode.decode(output)

        return output_surface, output_level, output_diagnostic