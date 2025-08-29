import time
from collections import OrderedDict
import numpy as np
import torch
import torchsparse
import torch.nn as nn
import torchsparse.nn as spnn
import MinkowskiEngine as ME
from mamba_ssm import Mamba

__all__ = ['MinkUNet']


class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True)
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, stride=stride, transpose=True),
            spnn.BatchNorm(outc),
            spnn.ReLU(True)
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation, stride=1),
            spnn.BatchNorm(outc)
        )
        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1, stride=stride),
                spnn.BatchNorm(outc)
            )
        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class GlobalChannelAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(GlobalChannelAttention, self).__init__()
        self.global_avg_pool = spnn.GlobalAveragePooling()

        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.conv_q = nn.Conv1d(1, 1, kernel_size, padding=(kernel_size - 1) // 2)
        self.conv_k = nn.Conv1d(1, 1, kernel_size, padding=(kernel_size - 1) // 2)
        self.conv_v = nn.Conv1d(1, 1, kernel_size, padding=(kernel_size - 1) // 2)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs: torchsparse.SparseTensor):
        feats = inputs.F          # (N, C)
        coords = inputs.C         # (N, 4)
        stride = inputs.s
        batch_index = coords[:, -1]  # (N,)

        # 2025-06-08 Jinzheng Guang
        # Step 1: Get global pooled features (B, C) torch.Size([32, 256])
        avg_pool = self.global_avg_pool(inputs)  # (B, C)

        # Step 2: Generate Q, K, V (B, 1, C)
        q = self.conv_q(avg_pool.unsqueeze(1))  # (B, 1, C) torch.Size([32, 1, 256])
        k = self.conv_k(avg_pool.unsqueeze(1))  # (B, 1, C) torch.Size([32, 1, 256])
        v = self.conv_v(avg_pool.unsqueeze(1))  # (B, 1, C) torch.Size([32, 1, 256])

        # Step 3: QK^T → attention map: (B, C, C)
        q = q.permute(0, 2, 1)  # (B, C, 1)
        attn_map = torch.bmm(q, k)  # (B, C, C) torch.Size([32, 256, 256])
        attn_map = self.softmax(attn_map)

        # Step 4: Apply attention to V
        v = v.permute(0, 2, 1)  # (B, C, 1) torch.Size([32, 256, 1])
        attn_out = torch.bmm(attn_map, v).squeeze(2)  # (B, C) torch.Size([32, 256])

        # Step 5: Map per-batch attention to per-point
        attn_expanded = attn_out[batch_index.long()]  # (N, C)

        # Step 6: Apply attention to original features
        output_feats = feats * attn_expanded

        # Step 7: Return new SparseTensor
        output_tensor = torchsparse.SparseTensor(
            coords=coords,
            feats=output_feats,
            stride=stride
        )
        output_tensor.coord_maps = inputs.coord_maps
        output_tensor.kernel_maps = inputs.kernel_maps

        return output_tensor


class ResidualPath(nn.Module):
    def __init__(self, inc, outc, ks=3):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks),
            spnn.BatchNorm(outc),
            spnn.ReLU(True)
        )
        self.se = GlobalChannelAttention(kernel_size=3)
        # 2025-02-18 Jinzheng Guang mamba
        self.mamba = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=outc, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )

    def forward(self, x):
        # 2025-02-18 Jinzheng Guang mamba
        inputs = self.net(x)
        inputs = self.se(inputs)
        # inputs = self.sp(inputs)
        
        n, c = inputs.F.shape 
        # (1, n, c) 
        y = inputs.F.clone().reshape(1, n, c)
        att = self.mamba(y) # (b,n,c)
        inputs.F = att.reshape(n, c) # (n,c)

        out = x + inputs
        return out

class MinkUNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        cr = kwargs.get('cr', 1.0)
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]
        self.run_up = kwargs.get('run_up', True)
        input_dim = kwargs.get("input_dim", 3)

        self.stem = nn.Sequential(
            spnn.Conv3d(input_dim, cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(True)
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1))

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1),
        )

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[5] + cs[3], cs[5], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[5], cs[5], ks=3, stride=1, dilation=1),
            )
        ])

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[6] + cs[2], cs[6], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[6], cs[6], ks=3, stride=1, dilation=1),
            )
        ])

        self.up3 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[7] + cs[1], cs[7], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[7], cs[7], ks=3, stride=1, dilation=1),
            )
        ])

        self.up4 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[8] + cs[0], cs[8], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[8], cs[8], ks=3, stride=1, dilation=1),
            )
        ])

        # 2023-03-31 Jinzheng Guang
        # cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        self.path = ResidualPath(cs[4], cs[4])

        self.classifier = nn.Sequential(nn.Linear(cs[8], kwargs['num_classes']))

    def forward(self, x):
        x0 = self.stem(x)  # 621,687 x 3
        x1 = self.stage1(x0)  # 621,687 x 32
        x2 = self.stage2(x1)  # 362,687 x 32
        x3 = self.stage3(x2)  # 192,434 x 64
        x4 = self.stage4(x3)  # 94,584 x 128
        x4 = self.path(x4)

        y1 = self.up1[0](x4)  # 42,187 x 256
        y1 = torchsparse.cat([y1, x3])
        y1 = self.up1[1](y1)  # 94,584 x 256

        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.up2[1](y2)  # 192,434 x 128

        y3 = self.up3[0](y2)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up3[1](y3)  # 362,687 x 96

        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up4[1](y4)  # 621,687 x 96

        out = self.classifier(y4.F)  # 621,687 x 31

        return out  # (n, 31)
