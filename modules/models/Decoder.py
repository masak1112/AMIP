import torch
import torch.nn as nn
import torch.nn.functional as F

class SphereConv2d(nn.Conv2d):
    """
    2D Convolution with circular padding in horizontal direction,
    and inverted reflection in vertical direction.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = [1, 1],
        padding =  [1, 1],
        dilation = [1, 1],
        groups: int = 1,
        bias: bool = True,
        padding_mode=None,
        padding_value=None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode="zeros",
            device=device,
            dtype=dtype,
        )

        # assert padding == 1, "For now, SphereConv2d only tested on padding=1 for spherical convolution."
        # self.padding = padding
        # assert self.kernel_size[0] == self.kernel_size[1] == 3, \
        # "SphereConv2d currently only tested on 3x3 kernels for spherical convolution."
        assert self.stride[0] == self.stride[1] == 1, (
            "SphereConv2d currently only tested on stride=1 for spherical convolution. "
        )

    @staticmethod
    def sphere_pad(input: torch.Tensor, padding = (1, 1)) -> torch.Tensor:
        """
        Pad a 4D tensor (batch, channels, height, width) for spherical convolution.
        Uses circular padding for longitude (width) and special pole handling for latitude (height).
        The top and bottom rows still need to be handled

        Args:
            input: Input tensor of shape (B, C, H, W)
            padding: Number of padding elements on each side (padH, padW).

        Returns:
            Padded tensor with spherical boundary conditions
        """

        half_width = input.shape[3] // 2

        top_rows = input[:, :, : padding[0], :]
        top_rows = torch.roll(top_rows, shifts=half_width, dims=3)
        top_rows = torch.flip(top_rows, dims=[2])
        bottom_rows = input[:, :, -padding[0] :, :]
        bottom_rows = torch.roll(bottom_rows, shifts=half_width, dims=3)
        bottom_rows = torch.flip(bottom_rows, dims=[2])
        input = torch.cat([top_rows, input, bottom_rows], dim=2)

        return F.pad(input, (padding[1], padding[1], 0, 0), mode="circular")

    def top_conv(self, input: torch.Tensor) -> torch.Tensor:
        """
        Apply convolution to the top slice of the input tensor.
        This is used to handle the top row of the spherical convolution after padding.
        """
        # Flip the top row weight for top slice convolution
        kernel = self.weight.clone()
        kernel[:, :, : self.padding[0], :] = torch.flip(
            kernel[:, :, : self.padding[0], :], dims=[3]
        )
        return F.conv2d(
            input, kernel, self.bias, self.stride, 0, self.dilation, self.groups
        )

    def bottom_conv(self, input: torch.Tensor) -> torch.Tensor:
        """
        Apply convolution to the bottom slice of the input tensor.
        This is used to handle the bottom row of the spherical convolution after padding.
        """
        # Flip the bottom row weight for bottom slice convolution
        kernel = self.weight.clone()
        kernel[:, :, -self.padding[0] :, :] = torch.flip(
            kernel[:, :, -self.padding[0] :, :], dims=[3]
        )
        return F.conv2d(
            input, kernel, self.bias, self.stride, 0, self.dilation, self.groups
        )

    def _conv_forward(
        self, input: torch.Tensor, weight: torch.Tensor, bias
    ):
        raise NotImplementedError(
            " SphereConv2d does not support _conv_forward method. Use forward method instead."
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input: (B, C, H, W)
        example:
        tmp = torch.arange(0, 24).view(1, 1, 3, 8)
        conv_cls = SphereConv2d(1, 1, 5, 1, 5//2)
        print(tmp)
        print(conv_cls.sphere_pad(tmp, (5//2, 5//2)))
        >>>
        tensor([[[[ 0,  1,  2,  3,  4,  5,  6,  7],
                [ 8,  9, 10, 11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20, 21, 22, 23]]]])
        tensor([[[[10, 11, 12, 13, 14, 15,  8,  9, 10, 11, 12, 13],
                [ 2,  3,  4,  5,  6,  7,  0,  1,  2,  3,  4,  5],
                [ 6,  7,  0,  1,  2,  3,  4,  5,  6,  7,  0,  1],
                [14, 15,  8,  9, 10, 11, 12, 13, 14, 15,  8,  9],
                [22, 23, 16, 17, 18, 19, 20, 21, 22, 23, 16, 17],
                [18, 19, 20, 21, 22, 23, 16, 17, 18, 19, 20, 21],
                [10, 11, 12, 13, 14, 15,  8,  9, 10, 11, 12, 13]]]])

        conv_cls.weight.data = torch.tensor([[[[0,1,0,0,0],[0,1,0,0,0],[0,0,0,0,0],[0,0,0,1,0],[0,0,0,1,0]]]], requires_grad=True, dtype=torch.float32)
        conv_cls.bias.data = torch.tensor([0.0], requires_grad=True, dtype=torch.float32)
        print(conv_cls.weight.data.shape)
        >>>
        tensor([[[[0., 1., 0., 0., 0.],
                [0., 1., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 1., 0.],
                [0., 0., 0., 1., 0.]]]])

        conv_cls(tmp.float())
        >>>
        tensor([[[[44., 48., 52., 40., 44., 48., 52., 40.],
                [48., 44., 48., 44., 48., 44., 48., 44.],
                [52., 40., 44., 48., 52., 40., 44., 48.]]]], grad_fn=<CatBackward0>)
        """
        input = self.sphere_pad(input, padding=self.padding)
        top_slice = input[:, :, : self.kernel_size[0], :]
        mid_slice = input[:, :, self.stride[0] : -self.stride[0], :]
        bottom_slice = input[:, :, -self.kernel_size[0] :, :]
        top_slice = self.top_conv(top_slice)
        # print("top slice", top_slice, top_slice.shape)
        mid_slice = F.conv2d(
            mid_slice,
            self.weight,
            self.bias,
            self.stride,
            0,
            self.dilation,
            self.groups,
        )
        # print("mid slice", mid_slice, mid_slice.shape)
        bottom_slice = self.bottom_conv(bottom_slice)
        # print("bottom slice", bottom_slice, bottom_slice.shape)
        return torch.cat([top_slice, mid_slice, bottom_slice], dim=2)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = SphereConv2d(in_channels, out_channels, kernel_size=(3, 3), padding = (1, 1))

        self.norm2 = Normalize(out_channels)
        self.conv2 = SphereConv2d(out_channels, out_channels, kernel_size=(3, 3), padding = (1, 1))

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = SphereConv2d(in_channels, out_channels, kernel_size=(3, 3), padding = (1, 1))
            else:
                self.nin_shortcut = SphereConv2d(in_channels, out_channels, kernel_size=1, stride=(1, 1), padding=(0, 0))

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class DCUpsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        interpolate: bool = False,
        shortcut: bool = True,
        interpolation_mode: str = "nearest",
    ) -> None:
        super().__init__()

        self.interpolate = interpolate
        self.interpolation_mode = interpolation_mode
        self.shortcut = shortcut
        self.factor = 2
        self.repeats = out_channels * self.factor**2 // in_channels

        out_ratio = self.factor**2

        if not interpolate:
            out_channels = out_channels * out_ratio

        self.conv = SphereConv2d(
            in_channels, out_channels, kernel_size=(3, 3), padding = (1, 1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.interpolate:
            x = F.interpolate(
                hidden_states, scale_factor=self.factor, mode=self.interpolation_mode
            )
            x = self.conv(x)
        else:
            x = self.conv(hidden_states)
            x = F.pixel_shuffle(x, self.factor)

        if self.shortcut:
            y = hidden_states.repeat_interleave(self.repeats, dim=1)
            y = F.pixel_shuffle(y, self.factor)
            hidden_states = x + y
        else:
            hidden_states = x

        return hidden_states

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def Normalize(in_channels, num_groups=16, type="group"):
    if type == "layer":
        return torch.nn.LayerNorm(in_channels, eps=1e-6)
    elif type == "group":
        return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    else:
        raise ValueError(f"unknown normalization type {type}")

class featscale2(nn.Module):
    def __init__(self, patch_size,channels):
        super(featscale2, self).__init__()
        self.patch_size = patch_size
        self.lambda1 = nn.Parameter(torch.ones(1,channels,1,1,1))
        self.lambda2 = nn.Parameter(torch.ones(1,channels,1,1,1))

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # Reshape into patches of shape (batch_size, channels, num_patches, patch_size, patch_size)
        X_patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        num_patches = (height//self.patch_size)* (width//self.patch_size)
        X_patches = X_patches.reshape(batch_size, channels, num_patches, self.patch_size, self.patch_size)
        X_mean_patch = X_patches.mean(dim=2)
        X_mean_expanded = X_mean_patch.unsqueeze(2).expand(-1, -1, num_patches, -1, -1)
        
        #Generate X_d and X_h
        X_d = X_mean_expanded
        X_h = X_patches - X_d

        # #Combine X_d and X_h
        X = X_patches + self.lambda1*X_d + self.lambda2*X_h
        X = X.reshape(batch_size, channels, height//self.patch_size, width//self.patch_size, self.patch_size, self.patch_size)
        X = X.permute(0,1,2,4,3,5).reshape(batch_size, channels, height, width)
        return X

    
class DecoderCNN(nn.Module):
    def __init__(self,
                 out_channels, # output channel dim
                 hidden_channels, # width of network
                 z_channels, # input latent dim
                 ch_mult=(1,2,4), 
                 num_res_blocks = 4,
                 num_out_blocks=0,
                 resolution = (180, 360), 
                 ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.nsurface = 6
        self.ndiagnostic = 9
        self.nlevels = 26
        init_patch_size=1

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = self.hidden_channels*ch_mult[self.num_resolutions-1]

        curr_res = resolution[-1] // 2**(self.num_resolutions-1)

        # z to block_in
        self.conv_in = SphereConv2d(in_channels=z_channels,
                                    out_channels=block_in,
                                    kernel_size=(3, 3), padding = (1, 1))

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in)
        self.fs_bottleneck = featscale2(patch_size=init_patch_size, channels=block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in)
        self.fs_bottleneck2 = featscale2(patch_size=init_patch_size, channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        self.feat_up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = self.hidden_channels*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                       out_channels=block_out))
                block_in = block_out

            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = DCUpsample(block_in, 
                                            block_in)
                curr_res = curr_res * 2
                init_patch_size = 2*init_patch_size
            self.up.insert(0, up) # prepend to get consistent order
            self.feat_up.insert(0, featscale2(patch_size=init_patch_size, channels=block_in))

        self.num_out_blocks = num_out_blocks
        if num_out_blocks > 0:
            self.out_blocks = nn.ModuleList()
            self.out_scale = nn.ModuleList()
            for i in range(num_out_blocks):
                self.out_blocks.append(ResnetBlock(in_channels=block_in,
                                       out_channels=block_out))
                self.out_scale.append(featscale2(patch_size=init_patch_size, channels=block_in))

        self.norm_out = Normalize(block_in) 
        self.conv_out = SphereConv2d(in_channels=z_channels,
                                    out_channels=block_in,
                                    kernel_size=(3, 3), padding = (1, 1))
        
        # Apply He Initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Applies He (Kaiming) initialization to Conv2d and Linear layers.
        Initializes normalization layers (LayerNorm, BatchNorm) with scale 1 and bias 0.
        """
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear, SphereConv2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, z) -> torch.Tensor:

        # x in shape b c nlat nlon

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.fs_bottleneck(h)
        h = self.mid.block_2(h)
        h = self.fs_bottleneck2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
            h = self.feat_up[i_level](h)

        if self.num_out_blocks > 0:
            for i in range(len(self.out_blocks)):
                h = self.out_blocks[i](h)
                h = self.out_scale[i](h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h) # b c h w

        return h
