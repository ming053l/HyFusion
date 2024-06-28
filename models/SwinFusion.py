import torch,argparse
from torch import nn
import numpy as np, math
from torch.nn import functional as F
from torch.autograd import Variable
import functools
from module_util import *
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import pdb
from ptflops import get_model_complexity_info
    
def parse_args():
    parser = argparse.ArgumentParser(description='Train Convex-Optimization-Aware SR net')
    
    parser.add_argument('--SEED', type=int, default=1029)
    parser.add_argument('--batch_size', type=int, default=2)

    parser.add_argument('--epochs', type=int, default=900)
    parser.add_argument('--lr_scheduler', type=str, default="cosine")
    parser.add_argument('--resume_ind', type=int, default=0)
    parser.add_argument('--resume_ckpt', type=str, default="")
    parser.add_argument('--snr', type=int, default=35)
    
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--step_size', type=int, default=200)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--eval_step', type=int, default=2)
    parser.add_argument('--finetuning_step', type=int, default=300, help='Works only if the mixed_align_opt is on')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay rate, 0 means training without weight decay')
    
    
    ## Data generator configuration
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--bands', type=int, default=172)
    parser.add_argument('--msi_bands', type=int, default=4)
    parser.add_argument('--mis_pix', type=int, default=0)
    parser.add_argument('--mixed_align_opt', type=int, default=0)
    parser.add_argument('--joint_loss', type=int, default=1)
    
    # Network architecture configuration
    parser.add_argument("--network_mode", type=int, default=1, help="Training network mode: 0) Single mode, 1) LRHSI+HRMSI, 2) COCNN (LRHSI+HRMSI+CO), Default: 2")     
    parser.add_argument('--num_base_chs', type=int, default=172, help='The number of the channels of the base feature')
    parser.add_argument('--num_blocks', type=int, default=6, help='The number of the repeated blocks in backbone')
    parser.add_argument('--num_agg_feat', type=int, default=172//4, help='the additive feature maps in the block')
    parser.add_argument('--groups', type=int, default=1, help="light version the group value can be >1, groups=1 for full COCNN version, groups=4 is COCNN-Light for 4 HRMSI version")
    
    # Others
    parser.add_argument("--root", type=str, default="./HSI/val/", help='data root folder')   
    parser.add_argument("--val_file", type=str, default="./val.txt")   
    parser.add_argument("--train_file", type=str, default="./train.txt")   
    parser.add_argument("--prefix", type=str, default="DCSN_cocnn_light_adv")  
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:device_id or cpu")  
    parser.add_argument("--DEBUG", type=bool, default=False)  
    parser.add_argument("--gpus", type=int, default=1)  
    
    
    args = parser.parse_args()

    return args
##===============MFB + MFA ===============================================
class MultiScaleFeatFusionBlock_Depthwise(nn.Module):   ## MFB
    def __init__(self, nf=64, gc=32, bias=False, groups=4):
        super(MultiScaleFeatFusionBlock_Depthwise, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=bias,dilation=1, groups=nf)
        self.conv2 = nn.Conv2d(nf + gc, nf + gc, 3, 1, 1, bias=bias,dilation=1, groups=nf + gc)
        self.conv3 = nn.Conv2d(nf + 2 * gc, nf + 2 * gc, 3, 1, 1, bias=bias,dilation=1, groups=nf + 2 * gc)
        self.conv4 = nn.Conv2d(nf + 3 * gc, nf + 3 * gc, 3, 1, 1, bias=bias,dilation=1, groups=nf + 3 * gc)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf + 4 * gc, 3, 1, 1, bias=bias,dilation=1, groups=nf + 4 * gc)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.pointwise1 = nn.Conv2d(nf,gc,1,1,0,1,1,bias=bias)
        self.pointwise2 = nn.Conv2d(nf + gc,gc,1,1,0,1,1,bias=bias)
        self.pointwise3 = nn.Conv2d(nf + 2 * gc,gc,1,1,0,1,1,bias=bias)
        self.pointwise4 = nn.Conv2d(nf + 3 * gc,gc,1,1,0,1,1,bias=bias)
        self.pointwise5 = nn.Conv2d(nf + 4 * gc,nf,1,1,0,1,1,bias=bias)



        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)
        initialize_weights([self.pointwise1 ,self.pointwise2,self.pointwise3,self.pointwise4,self.pointwise5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.pointwise1(self.conv1(x)))
        x2 = self.lrelu(self.pointwise2(self.conv2(torch.cat((x, x1), 1))))
        x3 = self.lrelu(self.pointwise3(self.conv3(torch.cat((x, x1, x2), 1))))
        x4 = self.lrelu(self.pointwise4(self.conv4(torch.cat((x, x1, x2, x3), 1))))
        x5 = self.pointwise5(self.conv5(torch.cat((x, x1, x2, x3, x4), 1)))
        return x5 * 0.2 + x
class MultiScaleFeatFusionBlock(nn.Module):   ## MFB
    def __init__(self, nf=64, gc=32, bias=True, groups=4):
        super(MultiScaleFeatFusionBlock, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias, groups=groups)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias, groups=groups)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias, groups=groups)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias, groups=groups)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias, groups=groups)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class MultiScaleFeatAggregation(nn.Module):   ## MFA


    def __init__(self, nf, gc=32, groups=4,mode=1):
        super(MultiScaleFeatAggregation, self).__init__()
        if mode ==0:
            self.MFB1 = MultiScaleFeatFusionBlock(nf, gc, groups=groups)
            self.MFB2 = MultiScaleFeatFusionBlock(nf, gc, groups=groups)
            self.MFB3 = MultiScaleFeatFusionBlock(nf, gc, groups=groups)
        elif mode ==1:
            self.MFB1 = MultiScaleFeatFusionBlock_Depthwise(nf, gc, groups=groups)
            self.MFB2 = MultiScaleFeatFusionBlock_Depthwise(nf, gc, groups=groups)
            self.MFB3 = MultiScaleFeatFusionBlock_Depthwise(nf, gc, groups=groups)

    def forward(self, x):
        out = self.MFB1(x)
        out = self.MFB2(out)
        out = self.MFB3(out)
        return out * 0.2 + x



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x,x_size)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops
    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # 结构为 [B, num_patches, C]
        if self.norm is not None:
            x = self.norm(x)  # 归一化
        return x


    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    输入:
        img_size (int): 图像的大小，默认为 224*224.
        patch_size (int): Patch token 的大小，默认为 4*4.
        in_chans (int): 输入图像的通道数，默认为 3.
        embed_dim (int): 线性 projection 输出的通道数，默认为 96.
        norm_layer (nn.Module, optional): 归一化层， 默认为N None.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)  # 图像的大小，默认为 224*224
        patch_size = to_2tuple(patch_size)  # Patch token 的大小，默认为 4*4
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]  # patch 的分辨率
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]  # patch 的个数，num_patches

        self.in_chans = in_chans  # 输入图像的通道数
        self.embed_dim = embed_dim  # 线性 projection 输出的通道数

    def forward(self, x, x_size):
        B, HW, C = x.shape  # 输入 x 的结构
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # 输出结构为 [B, Ph*Pw, C]
        return x


class COGuidedDCSN(nn.Module):
    
    def make_layer(block, n_layers):
        layers = []
        for _ in range(n_layers):
            layers.append(block())
        return nn.Sequential(*layers)
    
    def __init__(self, in_nc,out_nc, nf, nb, in_msi=4, gc=32, useCO=False, groups=4,
                img_size=128, patch_size=4, in_chans=172, embed_dim=96, 
                depths=[6,6,6,6], num_heads=[6,6,6,6],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super(COGuidedDCSN, self).__init__()
        
        self.useCO = useCO
        # for LRHSI
        in_nc_group = groups
        if in_nc % groups != 0:
            in_nc_group = 1
        self.up = torch.nn.Upsample(scale_factor=4)
        
        # For fusion
        
        self.conv_fuse = nn.Sequential(nn.Conv2d(embed_dim, out_nc, 3, 1, 1, bias=True, groups=in_nc_group), 
                                     nn.LeakyReLU(negative_slope=0.2, inplace=True))
    
        ##transformer
        self.window_size = window_size
        self.conv_first_HSI = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)  # HSI輸入
        self.conv_first_MSI = nn.Conv2d(in_msi, embed_dim, 3, 1, 1)  # MSI輸入
        self.conv_first_YFD = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)  # YFD輸入
        self.conv_last_YFD = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)  # YFD輸出


        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        

        # Feature enhancement embed
        self.patch_embed_FE = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed_FE.num_patches
        patches_resolution_FE = self.patch_embed_FE.patches_resolution
        self.patches_resolution_FE = patches_resolution_FE

        self.patch_unembed_FE = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # HSI embed
        self.patch_embed_HSI = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        # num_patches_HSI = self.patch_embed_HSI.num_patches
        patches_resolution_HSI = self.patch_embed_HSI.patches_resolution
        self.patches_resolution_HSI = patches_resolution_HSI

        self.patch_unembed_HSI = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # MSI embed
        self.patch_embed_MSI = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_msi, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        # num_patches_MSI = self.patch_embed_MSI.num_patches
        patches_resolution_MSI = self.patch_embed_MSI.patches_resolution
        self.patches_resolution_MSI = patches_resolution_MSI
        
        self.patch_unembed_MSI = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        
        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers of FE
        self.layers_FE = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution_FE[0],patches_resolution_FE[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint

                         )
            self.layers_FE.append(layer)

        # build layers of HSI
        self.layers_HSI = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution_HSI[0],patches_resolution_HSI[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint

                         )
            self.layers_HSI.append(layer)

        # build layers of MSI
        self.layers_MSI = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution_MSI[0],patches_resolution_MSI[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint

                         )
            self.layers_MSI.append(layer)

        self.norm = norm_layer(self.num_features)

        # self.apply(self._init_weights)
        # for bly in self.layers:
        #     bly._init_respostnorm()
        
        ##transformer
       
        self.last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=False)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed_FE(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers_FE:
            x = layer(x,x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed_FE(x, x_size)
        # x = self.avgpool(x.transpose(1, 2))  # B C 1
        # x = torch.flatten(x, 1)
        return x
    def forward_hsi(self,x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed_HSI(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers_HSI:
            x = layer(x,x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed_HSI(x, x_size)
        # x = self.avgpool(x.transpose(1, 2))  # B C 1
        # x = torch.flatten(x, 1)
        return x
    def forward_msi(self,x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed_MSI(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers_MSI:
            x = layer(x,x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed_MSI(x, x_size)
        # x = self.avgpool(x.transpose(1, 2))  # B C 1
        # x = torch.flatten(x, 1)
        return x
    def forward(self, co, lrhsi, hrmsi):

        lrhsi = self.up(lrhsi)
        lrhsi = self.conv_first_HSI(lrhsi)
        lrhsi = self.forward_hsi(lrhsi)

        hrmsi = self.conv_first_MSI(hrmsi)
        hrmsi = self.forward_msi(hrmsi)
        

        yfd = self.conv_fuse(hrmsi + lrhsi) #YFD
        ##transformer
        codraft = self.conv_first_YFD(yfd)
        codraft=self.forward_features(codraft)
        codraft=self.conv_last_YFD(codraft)
        ##transformer

    
        out = self.last(codraft+yfd)
        
        return out
    

##=========Swin_DCSN_Test full module==========
class SwinIR_DCSN(nn.Module): 
    def __init__(self, args):
        super(SwinIR_DCSN, self).__init__()
        self.snr = args.snr
        self.joint = args.network_mode
        self.decoder = COGuidedDCSN(in_nc=172, out_nc=172, nf=172, nb=6, gc=172//4, in_msi=4,groups=1)
        print('Use the COCNN!')
    def awgn(self, x):
        snr = 10**(self.snr/10.0)
        xpower = torch.sum(x**2)/x.numel()
        npower = torch.sqrt(xpower / snr)
        return x + torch.randn(x.shape).cuda() * npower


    def forward(self,data, LRHSI, HRMSI, mode=0): ### Mode=0, default, mode=1: encode only, mode=2: decoded only
        if self.snr>0 and mode==0 and (self.joint==2 or self.joint==0):
            data = self.awgn(data)
        elif self.snr>0 and mode==0 and self.joint==1:
            LRHSI = self.awgn(LRHSI)

        if self.joint==0:
            return self.decoder(data)
        elif self.joint==1: #co, lrhsi, hrmsi (No co, input lrhsi and hrmsi)
            return self.decoder(None, LRHSI, HRMSI)
        else:
            return self.decoder(data, LRHSI,HRMSI)
##=========Swin_DCSN_Test_FLOP_TEST==========
class Swin_DCSN_Test(nn.Module): 
    def __init__(self, args):
        super(Swin_DCSN_Test, self).__init__()
        self.snr = 35
        self.joint=1
        self.decoder = COGuidedDCSN(in_nc=172, out_nc=172, nf=172, nb=6, gc=172//4, in_msi=4,groups=1)
        print('Use the COCNN!')
    def awgn(self, x):
        snr = 10**(self.snr/10.0)
        xpower = torch.sum(x**2)/x.numel()
        npower = torch.sqrt(xpower / snr)
        return x + torch.randn(x.shape) * npower


    def forward(self,x,mode=0): ### Mode=0, default, mode=1: encode only, mode=2: decoded only
        data,lrhsi,hrhsi = x[0],x[1],x[2]
        if self.snr>0 and mode==0 and (self.joint==2 or self.joint==0):
            data = self.awgn(data)
        elif self.snr>0 and mode==0 and self.joint==1:
            lrhsi = self.awgn(lrhsi)

        if self.joint==0:
            return self.decoder(data)
        elif self.joint==1: #co, lrhsi, hrmsi (No co, input lrhsi and hrmsi)
            return self.decoder(None,  lrhsi,hrhsi)
        else:
            return self.decoder(data,lrhsi,hrhsi)

def prepare_input(resolution):
    data = torch.FloatTensor(1, 172,128,128)
    lr_hsi = torch.FloatTensor(1,172,32,32)
    hr_msi = torch.FloatTensor(1,4,128,128)
    return  dict(x = [data,lr_hsi,hr_msi])

if __name__ == '__main__':
    args = parse_args()
        # Create a network and a corresponding input
    device = 'cuda:0'
    model = Swin_DCSN_Test(args)
    flops, params = get_model_complexity_info(model, input_res=(1, 224, 224), 
                                              input_constructor=prepare_input,
                                              as_strings=True, print_per_layer_stat=False)
    print('      - Flops:  ' + flops)
    print('      - Params: ' + params)


