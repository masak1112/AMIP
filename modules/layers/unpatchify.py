import torch
from torch import nn
from einops import rearrange
from modules.layers.old.fa_basics import modulate_fused

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, 
                 hidden_size,
                 cond_dim,
                 patch_size, 
                 out_channels,
                 modulate_2d=False,
                 hpx=False):
        super().__init__()
        self.cond_dim = cond_dim
        if cond_dim is None:
            self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True))
        else:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            if hpx:
                self.linear = nn.Linear(hidden_size, patch_size**3 * out_channels, bias=True)
            else:
                self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(cond_dim, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )
            self.modulate_2d = modulate_2d
        
        self.init_params()

    def forward(self, x, c=None):
        if c is None:
            x = self.output_layer(x)
            return x
        else:
            z = self.adaLN_modulation(c) # b, 2*hidden_size
            if self.modulate_2d:
                z = z.unsqueeze(1).unsqueeze(1) # b, 1, 1, 2*hidden_size
            else:
                z = z.unsqueeze(1) # b, 1, 2*hidden_size
            shift, scale = z.chunk(2, dim=-1) # b, 1, hidden_size or b, 1, 1, hidden_size
            x = modulate_fused(self.norm_final(x), shift, scale)
            x = self.linear(x)
            return x
        
    def init_params(self):
        if self.cond_dim is not None:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.adaLN_modulation[0].weight, 0)
            nn.init.constant_(self.adaLN_modulation[0].bias, 0)
            nn.init.constant_(self.linear.weight, 0)
            nn.init.constant_(self.linear.bias, 0)

        else:
            nn.init.constant_(self.output_layer[1].weight, 0)
            nn.init.constant_(self.output_layer[1].bias, 0)


class Unpatchify(nn.Module):
    """
    Unpatchify a tensor.

    Args:
        img_size (tuple[int]): Lat, Lon
        patch_size (tuple[int]): Lat, Lon
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, grid_size, patch_size, in_dim, out_dim, cond_dim=None):
        super().__init__()
        self.grid_x, self.grid_y = grid_size
        self.patch_size = patch_size
        self.out_dim = out_dim
        self.in_dim = in_dim

        self.out_layer = FinalLayer(hidden_size=in_dim,
                                    cond_dim=cond_dim,
                                    patch_size=patch_size[0],
                                    out_channels=out_dim,)
    
    def forward(self, x, cond=None):
        # x in shape [b, nlat//p * nlon//p, dim]
        x = self.out_layer(x, cond) # [batch_size, nlat//p * nlon//p, patch_size * patch_size * out_dim]
        c = self.out_dim
        h, w = self.grid_x, self.grid_y

        assert h * w == x.shape[1]
        ph, pw = self.patch_size
        x = x.reshape(shape=(x.shape[0], h, w, ph, pw, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * ph, w * pw)) # [b, c, nlat, nlon]
        imgs = imgs.permute(0, 2, 3, 1) # [b, nlat, nlon, c]

        return imgs
    

class UnpatchifyHPX(nn.Module):
    """
    Unpatchify a tensor.

    Args:
        img_size (tuple[int]): Lat, Lon
        patch_size (tuple[int]): Lat, Lon
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, grid_side, grid_face, patch_size, in_dim, out_dim, cond_dim=None):
        super().__init__()
        self.grid_side = grid_side 
        self.grid_face = grid_face 
        self.patch_size = patch_size
        self.out_dim = out_dim
        self.in_dim = in_dim

        self.out_layer = FinalLayer(hidden_size=in_dim,
                                    cond_dim=cond_dim,
                                    patch_size=patch_size,
                                    out_channels=out_dim,
                                    hpx=True)
    
    def forward(self, x, cond=None):
        # x in shape [b, nface//p * nside//p * nside//p, dim]
        x = self.out_layer(x, cond) # [batch_size, nface//p * nside//p * nside//p, patch_size**3 * out_dim]
        c = self.out_dim
        h = w = self.grid_side
        f = self.grid_face

        assert f * h * w == x.shape[1]
        pf = ph = pw = self.patch_size
        x = x.reshape(shape=(x.shape[0], f, h, w, pf, ph, pw, c))
        x = torch.einsum('nfhwpqrc->ncfphqwr', x)
        imgs = x.reshape(shape=(x.shape[0], c, f * pf, h * ph, w * pw)) # [b, c, f, nside, nside]
        imgs = imgs.permute(0, 2, 3, 4, 1) # [b, f, nside, nside, c]

        return imgs

# borrowed from 
#https://gist.github.com/A03ki/2305398458cb8e2155e8e81333f0a965
def ICNR(tensor, initializer, upscale_factor=2, *args, **kwargs):
    "tensor: the 2-dimensional Tensor or more"
    upscale_factor_squared = upscale_factor * upscale_factor
    assert tensor.shape[0] % upscale_factor_squared == 0, \
        ("The size of the first dimension: "
         f"tensor.shape[0] = {tensor.shape[0]}"
         " is not divisible by square of upscale_factor: "
         f"upscale_factor = {upscale_factor}")
    sub_kernel = torch.empty(tensor.shape[0] // upscale_factor_squared,
                             *tensor.shape[1:])
    sub_kernel = initializer(sub_kernel, *args, **kwargs)
    return sub_kernel.repeat_interleave(upscale_factor_squared, dim=0)

class SubPixelConvICNR_2D(nn.Module):
    """
    Patch Embedding Recovery to 2D Image.

    Args:
        img_size (tuple[int]): Lat, Lon
        patch_size (tuple[int]): Lat, Lon
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, img_size, 
                 patch_size, 
                 in_chans, 
                 out_chans, 
                 cond_dim=None,
                 num_lat = 64, 
                 polar_pad = True, 
                 grid_has_poles = False):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        assert patch_size[0] == patch_size[1], 'mismatch'

        if polar_pad:
            self.pad_poles = PolarPad2d((1, 1))
        else:
            self.pad_poles = nn.ZeroPad2d((0, 0, 1, 1))
        self.pad_circular = nn.CircularPad2d((1, 1, 0, 0))

        self.conv = nn.Conv2d(in_chans, out_chans*patch_size[0]**2, kernel_size=3, stride=1, padding=0, bias=0)

        self.pixelshuffle = nn.PixelShuffle(patch_size[0])
        weight = ICNR(self.conv.weight, 
                      initializer=nn.init.kaiming_normal_,
                      upscale_factor=patch_size[0])
        self.conv.weight.data.copy_(weight)   # initialize conv.weight
        
        '''
        self.out_conv = nn.Sequential(nn.CircularPad2d((1, 1, 0, 0)),
                                      PolarPad2d((1, 1), num_lat=num_lat, grid_has_poles=grid_has_poles),
                                      nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=0, bias=0),
                                      nn.GELU(),
                                      nn.CircularPad2d((1, 1, 0, 0)),
                                      PolarPad2d((1, 1), num_lat=num_lat, grid_has_poles=grid_has_poles),
                                      nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=0, bias=0),
                                      nn.GELU())
        '''
        
        self.out_layer = FinalLayer(hidden_size=out_chans,
                                    cond_dim=cond_dim, 
                                    patch_size=1,
                                    out_channels=out_chans,
                                    modulate_2d=True)

    def forward(self, x, cond=None):
        # x in shape [b, nlat//p * nlon//p, dim]
        x = rearrange(x, 'b (h w) c -> b c h w', h = self.img_size[0] // self.patch_size[0], w = self.img_size[1] // self.patch_size[1])

        x_padded = self.pad_poles(self.pad_circular(x))
        output = self.conv(x_padded) # [batch_size, out_chans * patch_size[0]**2, nlat//p, nlon//p]
        
        output = self.pixelshuffle(output) # [batch_size, out_chans, nlat, nlon]
        #output = self.out_conv(output) # [batch_size, out_chans, nlat, nlon]
        
        output = rearrange(output, 'b c h w -> b h w c')
        output = self.out_layer(output, cond) # [batch_size, nlat, nlon, out_chans]

        return output

class PolarPad2d(nn.Module):
    """
    Padding for convolutions on a 2D grid over the pole.

    Args:
        pad: (size of top padding, size of bottom padding)
        x: Image with shape (n_batches, n_channels, lat, lon)
    """
    def __init__(self, pad):
        super().__init__()
        self.pad_top = pad[0]
        self.pad_bottom = pad[1]

    def forward(self, x):
        # assume x in shape (b, c, nlat, nlon), where nlat, nlon are even
        num_lat = x.shape[-2]
        pad_idxs = torch.cat((torch.arange(self.pad_top), torch.arange(self.pad_top+1, num_lat+self.pad_top+1),
                                torch.arange(num_lat+self.pad_top+2, num_lat+self.pad_top+self.pad_bottom+2))).long()
        x = nn.functional.pad(x, (0, 0, 1, 1), mode = 'constant', value = 0.)
        padded_x = nn.functional.pad(x, (0, 0, self.pad_top, self.pad_bottom), mode = 'reflect')[..., pad_idxs, :]
        
        padded_x[..., :self.pad_top, :] = torch.roll(padded_x[..., :self.pad_top, :], padded_x.shape[-1] // 2, dims = -1)
        padded_x[..., -self.pad_bottom:, :] = torch.roll(padded_x[..., -self.pad_bottom:, :], padded_x.shape[-1] // 2, dims = -1)
        return padded_x


class PolarPad3d(nn.Module):
    """
    Padding for convolutions on a 3D grid over the pole.

    Args:
        pad: (size of top padding, size of bottom padding, size of level padding)
        x: Image with shape (n_batches, n_channels, lat, lon, nlevel)
    """
    def __init__(self, pad): # assume grid does not have poles
        super().__init__()
        self.pad_top = pad[0]
        self.pad_bottom = pad[1]

    def forward(self, x):
        # x in shape b c nlat nlon nlevel, assume nlat, nlon are even
        num_lat = x.shape[-3]
        pad_idxs = torch.cat((torch.arange(self.pad_top), torch.arange(self.pad_top+1, num_lat+self.pad_top+1),
                                    torch.arange(num_lat+self.pad_top+2, num_lat+self.pad_top+self.pad_bottom+2))).long()

        x = nn.functional.pad(x, (0, 0, 1, 1, 0, 0), mode = 'constant', value = 0.)
        padded_x = nn.functional.pad(x, (0, 0, self.pad_top, self.pad_bottom, 0, 0), mode = 'reflect')[..., pad_idxs, :, :]

        padded_x[..., :self.pad_top, :] = torch.roll(padded_x[..., :self.pad_top, :], padded_x.shape[-1] // 2, dims = -1)
        padded_x[..., -self.pad_bottom:, :] = torch.roll(padded_x[..., -self.pad_bottom:, :], padded_x.shape[-1] // 2, dims = -1)
        return padded_x