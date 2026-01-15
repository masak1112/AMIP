import torch
import torch.nn as nn
from einops import rearrange

from modules.layers.dc_layers import SphereConv2d, LayerNorm2d, \
    PixelShuffleUpSampleLayer, PixelUnshuffleDownSampleLayer, ChannelAveragingDownSampleLayer, ChannelDuplicatingUpSampleLayer

class DCDownBlock2d(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        factor: int = 2,
    ) -> None:
        super().__init__()

        self.conv_block = PixelUnshuffleDownSampleLayer(
            in_channels=in_dim, out_channels=out_dim, kernel_size=3, factor=factor
        )
        self.shortcut_block = ChannelAveragingDownSampleLayer(
            in_channels=in_dim, out_channels=out_dim, factor=factor
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.conv_block(x) + self.shortcut_block(x)
    
class DCDownBlockLevels(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        factor: int = 2,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim 
        self.factor = factor

        self.conv = nn.Conv3d(in_dim, out_dim // factor, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(out_dim // factor)

        self.group_size = in_dim * factor // out_dim

    def forward(self, x):
        # x in shape b c nlevel nlat nlon
        
        residual = x 
        x = self.conv(x) # b c//factor nlevel nlat nlon
        x = self.act(x)
        x = self.norm(rearrange(x, 'b c nlevel nlat nlon -> b nlevel nlat nlon c'))
        x = rearrange(x, 'b nlevel nlat nlon c -> b c nlevel nlat nlon')
        # pixel unshuffle in level dimension
        x = rearrange(x, 'b c (f l) nlat nlon -> b (c f) l nlat nlon', f=self.factor)

        # downsample in level dimension
        residual = rearrange(residual, 'b c (f l) nlat nlon -> b (c f) l nlat nlon', f=self.factor)
        b, c, nlevel, nlat, nlon = residual.shape
        residual = residual.view(b, self.out_dim, self.group_size, nlevel, nlat, nlon)
        residual = residual.mean(dim=2) # b out_dim nlevel nlat nlon
        return x + residual



class DCUpBlock2d(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        factor: int = 2,
    ) -> None:
        super().__init__()
        self.conv_block = PixelShuffleUpSampleLayer(
            in_channels=in_dim, out_channels=out_dim, kernel_size=3, factor=factor
        )
        self.shortcut_block = ChannelDuplicatingUpSampleLayer(
            in_channels=in_dim, out_channels=out_dim, factor=factor)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.conv_block(x) + self.shortcut_block(x)

class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.nonlinearity = nn.GELU()
        self.conv1 = SphereConv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = SphereConv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm = LayerNorm2d(out_channels)

    def forward(self, x) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)
        x = self.norm(x)

        return x + residual
    
class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        hidden_channels = (128, 256, 512),
        blocks_per_layer = (2, 2, 2),
        out_shortcut: bool = True,
    ):
        super().__init__()

        num_layers = len(hidden_channels)

        self.in_2D = SphereConv2d(
            in_channels,
            hidden_channels[0],
            kernel_size=3,
            padding=1
        )
        
        self.in_3D = nn.Conv3d(
            in_channels,
            hidden_channels[0],
            kernel_size=3,
            padding=1,
        )

        self.down_layers = nn.ModuleList()
        for i, (out_channel, num_blocks) in enumerate(
            zip(hidden_channels, blocks_per_layer)
        ):
            for _ in range(num_blocks):
                block = ResBlock(
                    in_channels=out_channel,
                    out_channels=out_channel,
                )
                self.down_layers.append(block)

            if i < num_layers - 1: # no downsample on last layer
                downsample_block = DCDownBlock2d(
                    in_channels=out_channel,
                    out_channels=hidden_channels[i + 1],
                )
                self.down_layers.append(downsample_block)

        self.conv_out = SphereConv2d(hidden_channels[-1], 
                                     latent_channels, 
                                     kernel_size=3,
                                     padding=1)

        self.out_shortcut = out_shortcut
        if out_shortcut:
            self.out_shortcut_average_group_size = (
                hidden_channels[-1] // latent_channels
            )

    def forward(self, surface, multilevel, diagnostic) -> torch.Tensor:
        # surface in shape b nlat nlon c 
        # multilevel in shape b nlevel nlat nlon c
        # diagnostic in shape b nlat nlon c


        x = self.conv_in(x)
        for idx, down_block in enumerate(self.down_layers):
            hidden_states = down_block(hidden_states, temb)

        if self.out_shortcut:
            x = hidden_states.unflatten(1, (-1, self.out_shortcut_average_group_size))
            x = x.mean(dim=2)
            hidden_states = self.conv_out(hidden_states) + x
        else:
            hidden_states = self.conv_out(hidden_states)

        return hidden_states