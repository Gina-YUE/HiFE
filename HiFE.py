"""
The code of the paper "DeepFake Detection Based on High-Frequency Enhancement Network for Highly Compressed Content"
Author: Jie Gao
"""
# !/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import numpy as np
from math import sqrt
import math
from PIL import Image
from torchvision import transforms
from torchvision import transforms as trans
from torch.nn import functional as F
from Model_Prepare.Models.xception import xception
from pytorch_wavelets import DWTForward, DWTInverse
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

class DCT(nn.Module):
    def __init__(self, N = 8, in_channal = 3):
        super(DCT, self).__init__()
        self.N = N  # default is 8 for JPEG
        self.fre_len = N * N  #8*8=64
        self.in_channal = in_channal  #3
        self.out_channal =  N * N * in_channal #8*8*3=192

        # 3 H W -> N*N*in_channel  H/N  W/N
        self.dct_conv = nn.Conv2d(self.in_channal, self.out_channal, N, N, bias=False, groups=self.in_channal)
        # 64 *1 * 8 * 8, from low frequency to high fre
        self.weight = torch.from_numpy(self.mk_coff(N = N)).float().unsqueeze(1)
        self.dct_conv.weight.data = torch.cat([self.weight]*self.in_channal, dim=0) # 64 1 8 8
        self.dct_conv.weight.requires_grad = False


        self.Ycbcr = nn.Conv2d(3, 3, 1, 1, bias=False)
        trans_matrix = np.array([[0.299, 0.587, 0.114],
                                 [-0.169, -0.331, 0.5],
                                 [0.5, -0.419, -0.081]])  # a simillar version, maybe be a little wrong
        trans_matrix = torch.from_numpy(trans_matrix).float().unsqueeze(
            2).unsqueeze(3)
        self.Ycbcr.weight.data = trans_matrix
        self.Ycbcr.weight.requires_grad = False

        self.reYcbcr = nn.Conv2d(3, 3, 1, 1, bias=False)
        re_matrix = np.linalg.pinv(np.array([[0.299, 0.587, 0.114],
                                             [-0.169, -0.331, 0.5],
                                             [0.5, -0.419, -0.081]]))
        re_matrix = torch.from_numpy(re_matrix).float().unsqueeze(
            2).unsqueeze(3)
        self.reYcbcr.weight.data = re_matrix

    def forward(self, x):
        '''

        :param x: B C H W, 0-1. RGB  YCbCr:  b c h w, YCBCR  DCT: B C*64 H//8 W//8 ,   Y_L..Y_H  Cb_L...Cb_H   Cr_l...Cr_H
        :return:
        '''
        # jpg = (jpg * self.std) + self.mean # 0-1
        ycbcr = self.Ycbcr(x)  # b 3 h w
        dct = self.dct_conv(ycbcr)
        return ycbcr,dct

    def reverse(self, x):
        ycbcr = F.conv_transpose2d(x, torch.cat([self.weight] * 3, 0),
                                   bias=None, stride=8, groups=3)
        rgb = self.reYcbcr(ycbcr)
        return rgb, ycbcr

    def mk_coff(self, N = 8, rearrange = True):
        dct_weight = np.zeros((N*N, N, N))
        for k in range(N*N):
            u = k // N
            v = k % N
            for i in range(N):
                for j in range(N):
                    tmp1 = self.get_1d(i, u, N=N)
                    tmp2 = self.get_1d(j, v, N=N)
                    tmp = tmp1 * tmp2
                    tmp = tmp * self.get_c(u, N=N) * self.get_c(v, N=N)

                    dct_weight[k, i, j] += tmp
        if rearrange:
            out_weight = self.get_order(dct_weight, N = N)  # from low frequency to high frequency
        return out_weight # (N*N) * N * N

    def get_1d(self, ij, uv, N=8):
        result = math.cos(math.pi * uv * (ij + 0.5) / N)
        return result

    def get_c(self, u, N=8):
        if u == 0:
            return math.sqrt(1 / N)
        else:
            return math.sqrt(2 / N)

    def get_order(self, src_weight, N = 8):
        array_size = N * N
        # order_index = np.zeros((N, N))
        i = 0
        j = 0
        rearrange_weigth = src_weight.copy() # (N*N) * N * N
        for k in range(array_size - 1):
            if (i == 0 or i == N-1) and  j % 2 == 0:
                j += 1
            elif (j == 0 or j == N-1) and i % 2 == 1:
                i += 1
            elif (i + j) % 2 == 1:
                i += 1
                j -= 1
            elif (i + j) % 2 == 0:
                i -= 1
                j += 1
            index = i * N + j
            rearrange_weigth[k+1, ...] = src_weight[index, ...]
        return rearrange_weigth

class ReDCT(nn.Module):
    def __init__(self, N = 8, in_channal = 3):
        super(ReDCT, self).__init__()

        self.N = N  # default is 8 for JPEG
        self.in_channal = in_channal * N * N
        self.out_channal = in_channal
        self.fre_len = N * N

        self.weight = torch.from_numpy(self.mk_coff(N=N)).float().unsqueeze(1)


        self.reDCT = nn.ConvTranspose2d(self.in_channal, self.out_channal, self.N,  self.N, bias = False, groups=self.out_channal)
        self.reDCT.weight.data = torch.cat([self.weight]*self.out_channal, dim=0)
        self.reDCT.weight.requires_grad = False

        self.reYcbcr = nn.Conv2d(3, 3, 1, 1, bias=False)
        re_matrix = np.linalg.pinv(np.array([[0.299, 0.587, 0.114],
                                             [-0.169, -0.331, 0.5],
                                             [0.5, -0.419, -0.081]]))
        re_matrix = torch.from_numpy(re_matrix).float().unsqueeze(
            2).unsqueeze(3)
        self.reYcbcr.weight.data = re_matrix


    def forward(self, dct):
        '''
        IDCT  from DCT domain to pixle domain
        B C*64 H//8 W//8   ->   B C H W
        '''
        ycbcr = self.reDCT(dct)
        out=self.reYcbcr(ycbcr)

        return out,ycbcr

    def mk_coff(self, N = 8, rearrange = True):
        dct_weight = np.zeros((N*N, N, N))
        for k in range(N*N):
            u = k // N
            v = k % N
            for i in range(N):
                for j in range(N):
                    tmp1 = self.get_1d(i, u, N=N)
                    tmp2 = self.get_1d(j, v, N=N)
                    tmp = tmp1 * tmp2
                    tmp = tmp * self.get_c(u, N=N) * self.get_c(v, N=N)

                    dct_weight[k, i, j] += tmp
        if rearrange:
            out_weight = self.get_order(dct_weight, N = N)  # from low frequency to high frequency
        return out_weight # (N*N) * N * N

    def get_1d(self, ij, uv, N=8):
        result = math.cos(math.pi * uv * (ij + 0.5) / N)
        return result

    def get_c(self, u, N=8):
        if u == 0:
            return math.sqrt(1 / N)
        else:
            return math.sqrt(2 / N)

    def get_order(self, src_weight, N = 8):
        array_size = N * N
        # order_index = np.zeros((N, N))
        i = 0
        j = 0
        rearrange_weigth = src_weight.copy() # (N*N) * N * N
        for k in range(array_size - 1):
            if (i == 0 or i == N-1) and  j % 2 == 0:
                j += 1
            elif (j == 0 or j == N-1) and i % 2 == 1:
                i += 1
            elif (i + j) % 2 == 1:
                i += 1
                j -= 1
            elif (i + j) % 2 == 0:
                i -= 1
                j += 1
            index = i * N + j
            rearrange_weigth[k+1, ...] = src_weight[index, ...]
        return rearrange_weigth

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class DualCrossModalAttention(nn.Module):
    """ Dual CMA attention Layer"""

    def __init__(self, in_dim, activation=None, size=14, ratio=8, ret_att=False): #size=16
        super(DualCrossModalAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.ret_att = ret_att

        # query conv
        self.key_conv1 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.key_conv2 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.key_conv_share = nn.Conv2d(
            in_channels=in_dim//ratio, out_channels=in_dim//ratio, kernel_size=1)

        self.linear1 = nn.Linear(size*size, size*size)
        self.linear2 = nn.Linear(size*size, size*size)

        # separated value conv
        self.value_conv1 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma1 = nn.Parameter(torch.zeros(1))

        self.value_conv2 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x, y):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        B, C, H, W = x.size()  #4,728,14,14

        def _get_att(a, b):
            proj_key1 = self.key_conv_share(self.key_conv1(a)).view(
                B, -1, H*W).permute(0, 2, 1)  # B, HW, C
            proj_key2 = self.key_conv_share(self.key_conv2(b)).view(
                B, -1, H*W)  # B X C x (*W*H)
            energy = torch.bmm(proj_key1, proj_key2)  # B, HW, HW  4

            attention1 = self.softmax(self.linear1(energy))
            attention2 = self.softmax(self.linear2(
                energy.permute(0, 2, 1)))  # BX (N) X (N)

            return attention1, attention2

        att_y_on_x, att_x_on_y = _get_att(x, y)
        proj_value_y_on_x = self.value_conv2(y).view(
            B, -1, H*W)  # B, C, HW
        out_y_on_x = torch.bmm(proj_value_y_on_x, att_y_on_x.permute(0, 2, 1))
        out_y_on_x = out_y_on_x.view(B, C, H, W)
        out_x = self.gamma1*out_y_on_x + x

        proj_value_x_on_y = self.value_conv1(x).view(
            B, -1, H*W)  # B , C , HW
        out_x_on_y = torch.bmm(proj_value_x_on_y, att_x_on_y.permute(0, 2, 1))
        out_x_on_y = out_x_on_y.view(B, C, H, W)
        out_y = self.gamma2*out_x_on_y + y

        if self.ret_att:
            return out_x, out_y, att_y_on_x, att_x_on_y

        return out_x, out_y  # , attention


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan=2048*2, out_chan=2048, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU()
        )
        self.ca = ChannelAttention(out_chan, ratio=16)
        self.init_weight()

    def forward(self, x, y):
        fuse_fea = self.convblk(torch.cat((x, y), dim=1))
        fuse_fea = fuse_fea + fuse_fea * self.ca(fuse_fea)
        return fuse_fea

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

class HiFE(nn.Module):

    def __init__(self, J=1,num_classes=2):
        super(HiFE, self).__init__()
        self.xception = xception()
        self.xception.last_linear = nn.Linear(2048, num_classes)
        self.xfm1 = DWTForward(J=J, mode='zero', wave='haar')   
        self.ifm1 = DWTInverse(mode='zero', wave='haar')

        self.xfm2 = DWTForward(J=J, mode='zero', wave='haar')   
        self.ifm2 = DWTInverse(mode='zero', wave='haar')

        self.xfm3 = DWTForward(J=J, mode='zero', wave='haar') 
        self.ifm3 = DWTInverse(mode='zero', wave='haar')

        self.xfm4 = DWTForward(J=J, mode='zero', wave='haar') 
        self.ifm4 = DWTInverse(mode='zero', wave='haar')

        self.xfm5 = DWTForward(J=J, mode='zero', wave='haar')  
        self.ifm5 = DWTInverse(mode='zero', wave='haar')

        self.dct = DCT()
        self.idct = ReDCT()

        self.dual_cma0 = DualCrossModalAttention(in_dim=728, ret_att=False)
        self.dual_cma1 = DualCrossModalAttention(in_dim=728, ret_att=False)
        self.fusion0 = FeatureFusionModule(in_chan=728 * 2, out_chan=728)
        self.fusion1 = FeatureFusionModule(in_chan=728 * 2, out_chan=728)

        self.fusion2=nn.Sequential(
            nn.Conv2d(728*2, 728, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(728),
            nn.ReLU(True),

            nn.Conv2d(728, 728, kernel_size=3, stride=1, padding=1),  # 112
            nn.BatchNorm2d(728),
            nn.ReLU(True),
        )

        channels = 192
        self.ca1 = ChannelAttention(channels)

        self.conv_dctx_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 112
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # 112
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        self.conv_dctx_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 56
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 56
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.conv_dctx_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 28
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 28
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 28
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.conv_dctx_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 14
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(512, 728, kernel_size=3, stride=1, padding=1),  # 14
            nn.BatchNorm2d(728),
            nn.ReLU(True),
        )

        self.conv_dct = nn.Sequential(
            nn.Conv2d(192, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        ###################################################

        self.conv_x1 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0, groups=9),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1, groups=3),
            nn.BatchNorm2d(9),
            nn.ReLU(True),
        )
        self.conv_x2 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0, groups=9),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1, groups=3),
            nn.BatchNorm2d(9),
            nn.ReLU(True),
        )
        self.conv_x3 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0, groups=9),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1, groups=3),
            nn.BatchNorm2d(9),
            nn.ReLU(True),
        )
        self.conv_x4 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0, groups=9),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1, groups=3),
            nn.BatchNorm2d(9),
            nn.ReLU(True),
        )
        self.conv_x5 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0, groups=9),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1, groups=3),
            nn.BatchNorm2d(9),
            nn.ReLU(True),
        )
        self.conv_x1_2_1 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0, groups=9),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(True),
        )
        self.conv_x1_2_2 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0, groups=9),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(True),
        )
        self.conv_x1_2_3 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0, groups=9),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(True),
        )
        self.conv_x1_3_1 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0, groups=9),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(True),
        )
        self.conv_x1_3_2 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0, groups=9),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(True),
        )
        self.conv_x1_3_3 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0, groups=9),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(True),
        )
        self.conv_x1_4_1 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0, groups=9),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(True),
        )
        self.conv_x1_4_2 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0, groups=9),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(True),
        )
        self.conv_x1_4_3 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0, groups=9),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(True),
        )
        self.conv_x1_5_1 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0, groups=9),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(True),
        )
        self.conv_x1_5_2 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0, groups=9),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(True),
        )
        self.conv_x1_5_3 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0, groups=9),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(True),
        )
        self.conv_x2_1 = nn.Sequential(
            nn.Conv2d(12, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
        )
        self.conv_x2_2 = nn.Sequential(
            nn.Conv2d(12, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
        )

        self.conv_x2_3 = nn.Sequential(
            nn.Conv2d(12, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
        )
        self.conv_x3_1 = nn.Sequential(
            nn.Conv2d(12, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
        )
        self.conv_x3_2 = nn.Sequential(
            nn.Conv2d(12, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
        )
        self.conv_x3_3 = nn.Sequential(
            nn.Conv2d(12, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
        )
        self.conv_x4_1 = nn.Sequential(
            nn.Conv2d(12, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
        )
        self.conv_x4_2 = nn.Sequential(
            nn.Conv2d(12, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
        )
        self.conv_x4_3 = nn.Sequential(
            nn.Conv2d(12, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
        )
        self.conv_x5_1 = nn.Sequential(
            nn.Conv2d(12, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
        )
        self.conv_x5_2 = nn.Sequential(
            nn.Conv2d(12, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
        )
        self.conv_x5_3 = nn.Sequential(
            nn.Conv2d(12, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
        )
        self.conv_xl = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(True),

            nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(True),

            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(True),

            nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
        )
        self.conv_xh = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 728, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(728),
            nn.ReLU(True),
        )
        self.bn=nn.BatchNorm2d(3)

    def extract_features(self, x):
        ycbcr1, dct_x = self.dct(x)  # YCBCR:(8,3,224,224)   dct_x:(8,192,28,28)
        ca1 = self.ca1(dct_x)
        dct_choose = dct_x * ca1
        dct_choose = self.conv_dct(dct_choose)  # (8,192,28,28)
        dctx, ycbcr2 = self.idct(dct_choose)

        dctx1 = self.conv_dctx_1(dctx)  # （8,32，112,112）
        dctx2 = self.conv_dctx_2(dctx1)  # （8,64，56,56）
        dctx3 = self.conv_dctx_3(dctx2)  # （8,256，28,28）
        dctx4 = self.conv_dctx_4(dctx3)  # （8,728，14,14）

        # one
        Yl_1, Yh_1 = self.xfm1(x)
        x1_1 = Yh_1[0][:, :, 0, :, :].view(x.size(0), -1, 112, 112) 
        x1_2 = Yh_1[0][:, :, 1, :, :].view(x.size(0), -1, 112, 112) 
        x1_3 = Yh_1[0][:, :, 2, :, :].view(x.size(0), -1, 112, 112)  


        x1 = torch.cat((x1_1, x1_2, x1_3), dim=1) 
        x1 = self.conv_x1(x1)  

        x1_1 = x1[:, 0: 3, :, :]  
        x1_2 = x1[:, 3: 6, :, :]  
        x1_3 = x1[:, 6: 9, :, :] 

        # two
        yl_2, yh_2 = self.xfm2(x1_1)
        yl_3, yh_3 = self.xfm2(x1_2)
        yl_4, yh_4 = self.xfm2(x1_3)

        x1_2_1 = torch.cat((yh_2[0][:, :, 0, :, :].view(x.size(0), -1, 56, 56),
                        yh_3[0][:, :, 0, :, :].view(x.size(0), -1, 56, 56),
                        yh_4[0][:, :, 0, :, :].view(x.size(0), -1, 56, 56)), dim=1) 
        x1_2_2 = torch.cat((yh_2[0][:, :, 1, :, :].view(x.size(0), -1, 56, 56),
                        yh_3[0][:, :, 1, :, :].view(x.size(0), -1, 56, 56),
                        yh_4[0][:, :, 1, :, :].view(x.size(0), -1, 56, 56)), dim=1)  
        x1_2_3 = torch.cat((yh_2[0][:, :, 2, :, :].view(x.size(0), -1, 56, 56),
                        yh_3[0][:, :, 2, :, :].view(x.size(0), -1, 56, 56),
                        yh_4[0][:, :, 2, :, :].view(x.size(0), -1, 56, 56)), dim=1)  


        x1_2_1 = self.conv_x1_2_1(x1_2_1)  
        x1_2_2 = self.conv_x1_2_2(x1_2_2)  
        x1_2_3 = self.conv_x1_2_3(x1_2_3)  

        Yl_2, Yh_2 = self.xfm2(Yl_1)  
        x2_1 = Yh_2[0][:, :, 0, :, :].view(x.size(0), -1, 56, 56)  
        x2_2 = Yh_2[0][:, :, 1, :, :].view(x.size(0), -1, 56, 56)  
        x2_3 = Yh_2[0][:, :, 2, :, :].view(x.size(0), -1, 56, 56) 

        x2 = torch.cat((x2_1, x2_2, x2_3), dim=1)  
        x2 = self.conv_x2(x2)  

        x2_1 = x2[:, 0: 3, :, :]  
        x2_2 = x2[:, 3: 6, :, :]  
        x2_3 = x2[:, 6: 9, :, :]  

        x2_1 = torch.cat((x2_1, x1_2_1), dim=1)  
        x2_2 = torch.cat((x2_2, x1_2_2), dim=1)  
        x2_3 = torch.cat((x2_3, x1_2_3), dim=1)  

        x2_1 = self.conv_x2_1(x2_1)  
        x2_2 = self.conv_x2_2(x2_2)  
        x2_3 = self.conv_x2_3(x2_3) 

        # three
        yl_5, yh_5 = self.xfm3(x2_1)  
        yl_6, yh_6 = self.xfm3(x2_2)  
        yl_7, yh_7 = self.xfm3(x2_3)  

        x1_3_1 = torch.cat((yh_5[0][:, :, 0, :, :].view(x.size(0), -1, 28, 28),
                            yh_6[0][:, :, 0, :, :].view(x.size(0), -1, 28, 28),
                            yh_7[0][:, :, 0, :, :].view(x.size(0), -1, 28, 28)), dim=1)  
        x1_3_2 = torch.cat((yh_5[0][:, :, 1, :, :].view(x.size(0), -1, 28, 28),
                            yh_6[0][:, :, 1, :, :].view(x.size(0), -1, 28, 28),
                            yh_7[0][:, :, 1, :, :].view(x.size(0), -1, 28, 28)), dim=1)  
        x1_3_3 = torch.cat((yh_5[0][:, :, 2, :, :].view(x.size(0), -1, 28, 28),
                            yh_6[0][:, :, 2, :, :].view(x.size(0), -1, 28, 28),
                            yh_7[0][:, :, 2, :, :].view(x.size(0), -1, 28, 28)), dim=1) 

        x1_3_1 = self.conv_x1_3_1(x1_3_1)  
        x1_3_2 = self.conv_x1_3_2(x1_3_2) 
        x1_3_3 = self.conv_x1_3_3(x1_3_3)  

        Yl_3, Yh_3 = self.xfm2(Yl_2)  
        x3_1 = Yh_3[0][:, :, 0, :, :].view(x.size(0), -1, 28, 28)  
        x3_2 = Yh_3[0][:, :, 1, :, :].view(x.size(0), -1, 28, 28)  
        x3_3 = Yh_3[0][:, :, 2, :, :].view(x.size(0), -1, 28, 28) 

        x3 = torch.cat((x3_1, x3_2, x3_3), dim=1)  
        x3 = self.conv_x3(x3)  

        x3_1 = x3[:, 0: 3, :, :]  
        x3_2 = x3[:, 3: 6, :, :]  
        x3_3 = x3[:, 6: 9, :, :]  

        x3_1 = torch.cat((x3_1, x1_3_1), dim=1)  
        x3_2 = torch.cat((x3_2, x1_3_2), dim=1)  
        x3_3 = torch.cat((x3_3, x1_3_3), dim=1)  

        x3_1 = self.conv_x3_1(x3_1)  
        x3_2 = self.conv_x3_2(x3_2)  
        x3_3 = self.conv_x3_3(x3_3) 

        # four
        yl_8, yh_8 = self.xfm4(x3_1)  
        yl_9, yh_9 = self.xfm4(x3_2)  
        yl_10, yh_10 = self.xfm4(x3_3)  

        x1_4_1 = torch.cat((yh_8[0][:, :, 0, :, :].view(x.size(0), -1, 14, 14),
                            yh_9[0][:, :, 0, :, :].view(x.size(0), -1, 14, 14),
                            yh_10[0][:, :, 0, :, :].view(x.size(0), -1, 14, 14)), dim=1) 
        x1_4_2 = torch.cat((yh_8[0][:, :, 1, :, :].view(x.size(0), -1, 14, 14),
                            yh_9[0][:, :, 1, :, :].view(x.size(0), -1, 14, 14),
                            yh_10[0][:, :, 1, :, :].view(x.size(0), -1, 14, 14)), dim=1) 
        x1_4_3 = torch.cat((yh_8[0][:, :, 2, :, :].view(x.size(0), -1, 14, 14),
                            yh_9[0][:, :, 2, :, :].view(x.size(0), -1, 14, 14),
                            yh_10[0][:, :, 2, :, :].view(x.size(0), -1, 14, 14)), dim=1) 

        x1_4_1 = self.conv_x1_4_1(x1_4_1)  
        x1_4_2 = self.conv_x1_4_2(x1_4_2)  
        x1_4_3 = self.conv_x1_4_3(x1_4_3)  

        Yl_4, Yh_4 = self.xfm2(Yl_3) 
        x4_1 = Yh_4[0][:, :, 0, :, :].view(x.size(0), -1, 14, 14)  
        x4_2 = Yh_4[0][:, :, 1, :, :].view(x.size(0), -1, 14, 14)  
        x4_3 = Yh_4[0][:, :, 2, :, :].view(x.size(0), -1, 14, 14)  


        x4 = torch.cat((x4_1, x4_2, x4_3), dim=1) 
        x4 = self.conv_x4(x4)  

        x4_1 = x4[:, 0: 3, :, :]  
        x4_2 = x4[:, 3: 6, :, :]  
        x4_3 = x4[:, 6: 9, :, :]  

        x4_1 = torch.cat((x4_1, x1_4_1), dim=1) 
        x4_2 = torch.cat((x4_2, x1_4_2), dim=1) 
        x4_3 = torch.cat((x4_3, x1_4_3), dim=1)  

        x4_1 = self.conv_x4_1(x4_1)  
        x4_2 = self.conv_x4_2(x4_2)  
        x4_3 = self.conv_x4_3(x4_3)  

     
        xh = torch.cat([x4_1, x4_2, x4_3], dim=1) 
        xh = self.conv_xh(xh)  

        x = self.xception.fea_part1(x)  
        x = self.xception.block1(x)  
        x = self.xception.block2(x)  
        x = self.xception.block3(x) 
        x = self.xception.fea_part3(x)
        fusion1 = self.dual_cma0(x, dctx4)  
        f1 = self.fusion0(fusion1[0], fusion1[1]) 
        fusion2 = self.dual_cma1(x, xh) 
        f2 = self.fusion1(fusion2[0], fusion2[1]) 
        f3=torch.cat((f1,f2),dim=1)
        f = self.fusion2(f3)
        f = self.xception.fea_part4(f)
        f = self.xception.fea_part5(f)

        return f

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(x) 
        x = nn.ReLU(inplace=True)(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)) 
        x = x.view(x.size(0), -1) 
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.xception.last_linear(x)
        return x



if __name__=='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input = torch.rand([8, 3, 224, 224])
    net = HiFE(J=1, num_classes=2)
    output=net(input)
    print(output)


    