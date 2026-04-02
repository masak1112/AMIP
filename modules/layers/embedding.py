# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import math


class PositionalEmbedding(torch.nn.Module):
    """Timestep embedding used in the DDPM++ and ADM architectures.


    f = (1/M)^(i / N)
    [cos(f_i x), sin(f_i x)] for i =0,...,N - 1

    sup wavelength = sup  2 pi / f = 2 pi 1 / inf f = 2 pi / (1 / M) = 2 pi M

    """

    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class FourierEmbedding(torch.nn.Module):
    """Timestep embedding used in the NCSN++ architecture."""

    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * math.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class FrequencyEmbedding(torch.nn.Module):
    """Periodic Embedding.

    Useful for inputs defined on the circle [0, 2pi)
    """

    def __init__(self, num_channels):
        super().__init__()
        self.register_buffer(
            "freqs", torch.arange(1, num_channels + 1), persistent=False
        )

    def forward(self, x):
        freqs = self.freqs[None, :, None, None]
        x = x[:, None, :, :]
        x = x * (2 * math.pi * freqs).to(x.dtype)
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class CalendarEmbedding(torch.nn.Module):
    """Time embedding assuming 365.25 day years

    Args:
        day_of_year: (n, t)
        second_of_day: (n, t)
    Returns:
        (n, embed_channels * 4, t, x)

    """

    def __init__(self, nlon, nlat, embed_channels: int):
        super().__init__()

        lon = (torch.arange(nlon, dtype=torch.float32) + 0.5) / nlon * 360

        self.nlat = nlat

        self.register_buffer("lon", lon, persistent=False)
        self.embed_channels = embed_channels
        self.embed_second = FrequencyEmbedding(embed_channels)
        self.embed_day = FrequencyEmbedding(embed_channels)
        self.embed_co2 = FrequencyEmbedding(embed_channels)
        self.out_channels = embed_channels * 6
        self.out_proj = torch.nn.Linear(self.out_channels, self.embed_channels)

    def forward(self, calendar):
        
        second_of_day = calendar[:, :1] # n 1
        day_of_year = calendar[:, 1:2] # n 1
        co2 = calendar[:, 2:] # n 1

        if second_of_day.shape != day_of_year.shape:
             # n, 1
            raise ValueError()
        
        local_time = (second_of_day.unsqueeze(2) + self.lon * 86400 // 360) % 86400 # n 1 nlon

        a = self.embed_second(local_time / 86400)
        doy = day_of_year.unsqueeze(2)
        b = self.embed_day((doy / 365.25) % 1)
        c = self.embed_co2(co2.unsqueeze(2))
        a, b, c = torch.broadcast_tensors(a, b, c)

        out = torch.concat([a, b, c], dim=1).squeeze(2) # n, embed_channels * 4, nlon

        out = self.out_proj(out.transpose(1, 2)).transpose(1, 2)

        # repeat to n c nlat nlon 
        out = out.unsqueeze(2).expand(-1, -1, self.nlat, -1)

        return out 