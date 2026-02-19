import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from modules.layers.arches_layers import (
    CondBasicLayer,
    DCDownSample,
    DCUpSample,
    Mlp,
    TimestepEmbedder
)

from modules.models.Arches_DiT import WeatherEncodeDecodeLayer

class ArchesSiT(nn.Module):
    def __init__(
        self,
        encode_decode_params: dict,
        tensor_size=(28, 180, 360),
        emb_dim=256,
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

        # Apply He Initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Applies He (Kaiming) initialization to Conv2d and Linear layers.
        Initializes normalization layers (LayerNorm, BatchNorm) with scale 1 and bias 0.
        """
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

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
