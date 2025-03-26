import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_

GROUP = 16


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


class SA(nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super(SA, self).__init__()

        self.Avg = nn.AdaptiveAvgPool2d(1)  # b c 1 1
        self.seq = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU6(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        A = self.Avg(x)  # b c 1 1
        score = self.seq(A.view(A.size(0), A.size(1)))  # b c
        out = x * score.view(score.size(0), score.size(1), 1, 1)  # b c h w * b c 1 1

        return out


def Conv_gn_relu(in_channel, group):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, 3, 1, 1),
        nn.GroupNorm(group, in_channel),
        nn.ReLU6()
    )


def downsample(in_channel, out_channel, kernel, stride, group):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=kernel//2-1),
        nn.GroupNorm(group, out_channel),
        nn.ReLU6()
    )


class Skip_L(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.layer1 = nn.Conv2d(in_channel, out_channel, 1, bias=False)
        self.norm = nn.LayerNorm(out_channel)

    def forward(self, x):
        x = self.layer1(x)
        x = rearrange(x.flatten(2), 'b c f -> b f c')
        x = self.norm(x)
        return x


class Multi_MixedAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr1 = nn.AvgPool2d(kernel_size=(sr_ratio, sr_ratio), stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.kv = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(qk_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        h = int(math.sqrt(x.size()[1]))
        q = self.q(x)  # b f 64 -> b f 64 -> b 8 f 8
        q = rearrange(q, 'b f (h hd) -> b h f hd', h=self.num_heads, hd=self.head_dim)  # b f 64 -> b 8 f 8

        if self.sr_ratio > 1:
            # b f 64 -> b 64 32 32
            xx = rearrange(x, 'b (h w) c -> b c h w', h=h, w=h)
            # b 64 32 32 -> b 64 16 16 -> b 256 64
            xx = rearrange(self.sr1(xx), 'b c h1 w1 -> b (h1 w1) c')
            xx = self.norm(xx)
            # b 256 64 -> b 256 64 -> b 8 256 8
            kv = rearrange(self.kv(xx), 'b hw (h hd) -> b h hw hd', h=self.num_heads, hd=self.head_dim)
        else:

            kv = rearrange(self.kv(x), 'b hw (h hd) -> b h hw hd', h=h, hd=h)
        k = kv
        v = kv

        attn = (q @ k.transpose(-2, -1)) * self.scale  # b 16 f 16*16
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = rearrange(out, 'b h f hd -> b f (h hd)')  # b f 64
        out = self.proj(out)
        out = self.proj_drop(out)

        return out  # b f 256


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super().__init__()

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class WS(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(WS, self).__init__()

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = nn.Sequential(
            nn.GroupNorm(4, dim),
            nn.ReLU6()
        )

    def forward(self, x, sa_res):
        h = int(math.sqrt(x.size()[1]))
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=h)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        sa_res = rearrange(sa_res, 'b (h w) c -> b c h w', h=x.size()[2], w=x.size()[3])
        weights = nn.ReLU6()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * sa_res + fuse_weights[1] * x

        x = self.post_conv(x)
        out = rearrange(x.flatten(2), 'b c f -> b f c')
        return out


class Encoder(nn.Module):
    def __init__(self, dim, group, reduction_ratio):
        super().__init__()
        self.sa = SA(dim, reduction_ratio)
        self.cgr = Conv_gn_relu(dim, group)

    def forward(self, x):
        x = self.sa(x)
        x = self.cgr(x)

        return x


class Decoder(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias, qk_drop, proj_drop, sr_ratio,
                 hidden_features, mlp_drop, drop_path):
        super().__init__()
        self.attn = Multi_MixedAttention(dim, num_heads, qkv_bias, qk_drop, proj_drop, sr_ratio)
        self.mlp = MLP(dim, hidden_features, dim, drop=mlp_drop)
        self.drop_path = DropPath(drop_path) if drop_path>0. else nn.Identity()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class Linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=GROUP, act=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.gn = nn.GroupNorm(g, c2)
        self.act = nn.ReLU6() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        out = self.act(x)

        return out


class ClassHead(nn.Module):
    def __init__(self,
                 num_classes=24,
                 in_channels=[64, 128, 256, 512],
                 embed_dim=768,
                 drop_ratio=0.1):
        super(ClassHead, self).__init__()

        self.weights = nn.Parameter(torch.ones(4, dtype=torch.float32), requires_grad=True)
        self.linear1 = Linear(in_channels[0], embed_dim)
        self.linear2 = Linear(in_channels[1], embed_dim)
        self.linear3 = Linear(in_channels[2], embed_dim)
        self.linear4 = Linear(in_channels[3], embed_dim)

        self.linear_fuse = ConvModule(c1=embed_dim * 4, c2=embed_dim)
        self.seg_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=False),
            nn.GroupNorm(GROUP, embed_dim),
            nn.ReLU6(),
            nn.Dropout2d(p=drop_ratio, inplace=True),
            nn.Conv2d(embed_dim, num_classes, 1, bias=False)
        )

    def forward(self, c1, c2, c3, c4):
        h1 = int(math.sqrt(c1.size()[1]))
        h2 = int(math.sqrt(c2.size()[1]))
        h3 = int(math.sqrt(c3.size()[1]))
        h4 = int(math.sqrt(c4.size()[1]))

        c1_ = rearrange(self.linear1(c1), 'b (h1 w1) c -> b c h1 w1', h1=h1, w1=h1)

        c2_ = rearrange(self.linear2(c2), 'b (h2 w2) c -> b c h2 w2', h2=h2, w2=h2)
        c2_ = F.interpolate(c2_, size=(h1, h1), mode='bilinear', align_corners=False)

        c3_ = rearrange(self.linear3(c3), 'b (h3 w3) c -> b c h3 w3', h3=h3, w3=h3)
        c3_ = F.interpolate(c3_, size=(h1, h1), mode='bilinear', align_corners=False)

        c4_ = rearrange(self.linear4(c4), 'b (h4 w4) c -> b c h4 w4', h4=h4, w4=h4)
        c4_ = F.interpolate(c4_, size=(h1, h1), mode='bilinear', align_corners=False)

        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + 1e-8)
        out = torch.cat((fuse_weights[0]*c1_, fuse_weights[1]*c2_, fuse_weights[2]*c3_, fuse_weights[3]*c4_), dim=1)
        out = self.linear_fuse(out)
        out = self.seg_head(out)

        return out


class UM2Former(nn.Module):
    def __init__(self, num_classes=24, in_channel=32, out_channel=(64, 128, 256, 512), embed_dim=128,
                 gruop=GROUP, kernel=4, stride=2, reduction_ratio=16,
                 num_heads=8, qkv_bias=False, qk_drop=0., proj_drop=0., sr_ratio=(4, 8, 16, 32),
                 mlp_ratio=2, mlp_drop=0.3, drop_path=0.3, drop_class=0.1):
        super().__init__()

        self.kernel = kernel
        self.stride = stride

        self.preprocess = downsample(in_channel, out_channel[0], kernel, stride, gruop)

        self.encode1 = Encoder(out_channel[0], gruop, reduction_ratio)  # catch feature
        self.down1 = downsample(out_channel[0], out_channel[1], kernel, stride, gruop)

        self.encode2 = Encoder(out_channel[1], gruop, reduction_ratio)  # catch feature
        self.down2 = downsample(out_channel[1], out_channel[2], kernel, stride, gruop)

        self.encode3 = Encoder(out_channel[2], gruop, reduction_ratio)  # catch feature
        self.down3 = downsample(out_channel[2], out_channel[3], kernel, stride, gruop)

        self.encode4 = Encoder(out_channel[3], gruop, reduction_ratio)

        self.skip1 = Skip_L(out_channel[-1], embed_dim)
        self.decode1 = Decoder(embed_dim, num_heads, qkv_bias, qk_drop, proj_drop, sr_ratio[0],
                               hidden_features=mlp_ratio*embed_dim, mlp_drop=mlp_drop, drop_path=drop_path)

        self.skip2 = Skip_L(out_channel[-2], embed_dim)
        self.ws2 = WS(embed_dim)
        self.decode2 = Decoder(embed_dim, num_heads, qkv_bias, qk_drop, proj_drop, sr_ratio[1],
                               hidden_features=mlp_ratio*embed_dim, mlp_drop=mlp_drop, drop_path=drop_path)

        self.skip3 = Skip_L(out_channel[-3], embed_dim)
        self.ws3 = WS(embed_dim)
        self.decode3 = Decoder(embed_dim, num_heads, qkv_bias, qk_drop, proj_drop, sr_ratio[2],
                               hidden_features=mlp_ratio*embed_dim, mlp_drop=mlp_drop, drop_path=drop_path)

        self.skip4 = Skip_L(out_channel[-4], embed_dim)
        self.ws4 = WS(embed_dim)
        self.decode4 = Decoder(embed_dim, num_heads, qkv_bias, qk_drop, proj_drop, sr_ratio[3],
                               hidden_features=mlp_ratio*embed_dim, mlp_drop=mlp_drop, drop_path=drop_path)

        self.class_head = ClassHead(num_classes, (embed_dim, embed_dim, embed_dim, embed_dim),
                                    embed_dim, drop_class)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, init):
        h, w = init.size()[2:]
        ini = self.preprocess(init)
        e1 = self.encode1(ini)

        e2 = self.down1(e1)
        e2 = self.encode2(e2)

        e3 = self.down2(e2)
        e3 = self.encode3(e3)

        e4 = self.down3(e3)
        e4 = self.encode4(e4)

        d1 = self.skip1(e4)
        d1 = self.decode1(d1)

        d2 = self.skip2(e3)
        d2 = self.ws2(d1, d2)
        d2 = self.decode2(d2)

        d3 = self.skip3(e2)
        d3 = self.ws3(d2, d3)
        d3 = self.decode3(d3)

        d4 = self.skip4(e1)
        d4 = self.ws4(d3, d4)
        d4 = self.decode4(d4)

        out = self.class_head(d4, d3, d2, d1)

        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)

        return out


# for S8 and T5
class UM2Former_OP8(nn.Module):
    def __init__(self, num_classes=24, in_channel=32, out_channel=(64, 128, 256, 512), embed_dim=128,
                 gruop=GROUP, kernel=8, stride=2, reduction_ratio=16,
                 num_heads=8, qkv_bias=False, qk_drop=0., proj_drop=0., sr_ratio=(4, 8, 16, 32),
                 mlp_ratio=2, mlp_drop=0.3, drop_path=0.3, drop_class=0.1):
        super().__init__()

        self.kernel = kernel
        self.stride = stride

        self.preprocess = downsample(in_channel, out_channel[0], kernel, stride, gruop)

        self.encode1 = Encoder(out_channel[0], gruop, reduction_ratio)  # catch feature
        self.down1 = downsample(out_channel[0], out_channel[1], kernel, stride, gruop)

        self.encode2 = Encoder(out_channel[1], gruop, reduction_ratio)  # catch feature
        self.down2 = downsample(out_channel[1], out_channel[2], kernel, stride, gruop)

        self.encode3 = Encoder(out_channel[2], gruop, reduction_ratio)  # catch feature
        self.down3 = downsample(out_channel[2], out_channel[3], kernel, stride, gruop)

        self.encode4 = Encoder(out_channel[3], gruop, reduction_ratio)

        self.skip1 = Skip_L(out_channel[-1], embed_dim)
        self.decode1 = Decoder(embed_dim, num_heads, qkv_bias, qk_drop, proj_drop, sr_ratio[0],
                               hidden_features=mlp_ratio*embed_dim, mlp_drop=mlp_drop, drop_path=drop_path)

        self.skip2 = Skip_L(out_channel[-2], embed_dim)
        self.ws2 = WS(embed_dim)
        self.decode2 = Decoder(embed_dim, num_heads, qkv_bias, qk_drop, proj_drop, sr_ratio[1],
                               hidden_features=mlp_ratio*embed_dim, mlp_drop=mlp_drop, drop_path=drop_path)

        self.skip3 = Skip_L(out_channel[-3], embed_dim)
        self.ws3 = WS(embed_dim)
        self.decode3 = Decoder(embed_dim, num_heads, qkv_bias, qk_drop, proj_drop, sr_ratio[2],
                               hidden_features=mlp_ratio*embed_dim, mlp_drop=mlp_drop, drop_path=drop_path)

        self.skip4 = Skip_L(out_channel[-4], embed_dim)
        self.ws4 = WS(embed_dim)
        self.decode4 = Decoder(embed_dim, num_heads, qkv_bias, qk_drop, proj_drop, sr_ratio[3],
                               hidden_features=mlp_ratio*embed_dim, mlp_drop=mlp_drop, drop_path=drop_path)

        self.class_head = ClassHead(num_classes, (embed_dim, embed_dim, embed_dim, embed_dim),
                                    embed_dim, drop_class)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, init):
        h, w = init.size()[2:]
        ini = self.preprocess(init)
        e1 = self.encode1(ini)

        e2 = self.down1(e1)
        e2 = self.encode2(e2)

        e3 = self.down2(e2)
        e3 = self.encode3(e3)

        e4 = self.down3(e3)
        e4 = self.encode4(e4)

        d1 = self.skip1(e4)
        d1 = self.decode1(d1)

        d2 = self.skip2(e3)
        d2 = self.ws2(d1, d2)
        d2 = self.decode2(d2)

        d3 = self.skip3(e2)
        d3 = self.ws3(d2, d3)
        d3 = self.decode3(d3)

        d4 = self.skip4(e1)
        d4 = self.ws4(d3, d4)
        d4 = self.decode4(d4)

        out = self.class_head(d4, d3, d2, d1)

        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)

        return out


