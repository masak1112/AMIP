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
        hidden_channels = (256, 256, 512, 512),
        blocks_per_layer = (2, 2, 2, 2),
        out_shortcut: bool = True,
    ):
        super().__init__()

        num_layers = len(hidden_channels)
        latent_channels = in_channels

        self.conv_in = SphereConv2d(
            in_channels,
            hidden_channels[0],
            kernel_size=3,
            padding=1
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
        n_surface = surface.shape[-1]
        n_diagnostic = diagnostic.shape[-1]
        n_levels = multilevel.shape[1]

        surface = rearrange(surface, 'b nlat nlon c -> b c nlat nlon')
        diagnostic = rearrange(diagnostic, 'b nlat nlon c -> b c nlat nlon')
        # flatten levels to channels. This is because we are purely compressing in lat/lon dimensions
        multilevel = rearrange(multilevel, 'b nlevel nlat nlon c -> b (c nlevel) nlat nlon')

        x = torch.cat([surface, diagnostic, multilevel], dim=1) # b c nlat nlon

        x = self.conv_in(x) # b hidden_dim nlat nlon

        for down_block in self.down_layers:
            x = down_block(x) 

        if self.out_shortcut:
            residual = x.unflatten(1, (-1, self.out_shortcut_average_group_size))
            residual = residual.mean(dim=2)
            x = self.conv_out(x) + residual
        else:
            x = self.conv_out(x) # b latent_dim zlat zlon

        z_surface = x[:, :n_surface, :, :] # b n_surface zlat zlon
        z_diagnostic = x[:, n_surface:n_surface + n_diagnostic, :, :] # b n_diagnostic zlat zlon
        z_multilevel = x[:, n_surface + n_diagnostic:, :, :] # b (n_multilevel * nlevel) zlat zlon

        z_multilevel = rearrange(z_multilevel, 'b (c nlevel) zlat zlon -> b nlevel zlat zlon c', nlevel=n_levels)
        z_surface = rearrange(z_surface, 'b c zlat zlon -> b zlat zlon c')
        z_diagnostic = rearrange(z_diagnostic, 'b c zlat zlon -> b zlat zlon c')

        return z_surface, z_diagnostic, z_multilevel
    
class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels = (256, 256, 512, 512),
        blocks_per_layer = (2, 2, 2, 2),
        out_shortcut: bool = True,
    ):
        super().__init__()

        num_layers = len(hidden_channels)
        latent_channels = in_channels

        self.conv_in = SphereConv2d(
            in_channels,
            hidden_channels[0],
            kernel_size=3,
            padding=1
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
        n_surface = surface.shape[-1]
        n_diagnostic = diagnostic.shape[-1]
        n_levels = multilevel.shape[1]

        surface = rearrange(surface, 'b nlat nlon c -> b c nlat nlon')
        diagnostic = rearrange(diagnostic, 'b nlat nlon c -> b c nlat nlon')
        # flatten levels to channels. This is because we are purely compressing in lat/lon dimensions
        multilevel = rearrange(multilevel, 'b nlevel nlat nlon c -> b (c nlevel) nlat nlon')

        x = torch.cat([surface, diagnostic, multilevel], dim=1) # b c nlat nlon

        x = self.conv_in(x) # b hidden_dim nlat nlon

        for down_block in self.down_layers:
            x = down_block(x) 

        if self.out_shortcut:
            residual = x.unflatten(1, (-1, self.out_shortcut_average_group_size))
            residual = residual.mean(dim=2)
            x = self.conv_out(x) + residual
        else:
            x = self.conv_out(x) # b latent_dim zlat zlon

        z_surface = x[:, :n_surface, :, :] # b n_surface zlat zlon
        z_diagnostic = x[:, n_surface:n_surface + n_diagnostic, :, :] # b n_diagnostic zlat zlon
        z_multilevel = x[:, n_surface + n_diagnostic:, :, :] # b (n_multilevel * nlevel) zlat zlon

        z_multilevel = rearrange(z_multilevel, 'b (c nlevel) zlat zlon -> b nlevel zlat zlon c', nlevel=n_levels)
        z_surface = rearrange(z_surface, 'b c zlat zlon -> b zlat zlon c')
        z_diagnostic = rearrange(z_diagnostic, 'b c zlat zlon -> b zlat zlon c')

        return z_surface, z_diagnostic, z_multilevel