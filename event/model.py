import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from neuron import MultiStepNegIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from functools import partial
from wavelet_layers import *

__all__ = ['swformer']

class Erode(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        )

    def forward(self, x):
        return self.pool(x)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)

        self.fc2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm2d(out_features)
        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T,B,C,H,W = x.shape
        
        x = self.fc1_lif(x) # T, B, C, H, W
        x = self.fc1_conv(x.flatten(0, 1).contiguous()) # T*B, C, H, W
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, H, W).contiguous()
        
        x = self.fc2_lif(x) #T, B, C, H, W
        x = self.fc2_conv(x.flatten(0,1).contiguous()) # T*B, C, H, W
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous() # T, B, C, H, W
        
        return x

class FATM(nn.Module):
    def __init__(self, dim, FLblocks = 16):
        super(FATM, self).__init__()
        self.hidden_size = dim

        self.num_blocks = FLblocks
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0

        self.x_neuron = MultiStepLIFNode(tau=2.0, detach_reset=True)

        # 2D discrete wavelet transform
        self.haar_neuron = MultiStepLIFNode(tau=2.0, detach_reset=True)
        self.haar_forward = Haar2DForward(MultiStepNegIFNode)
        self.haar_inverse = Haar2DInverse(MultiStepNegIFNode)
        self.haar_forward_bn = nn.BatchNorm2d(dim)
        self.haar_multiply_bn = nn.BatchNorm2d(dim)
        self.haar_inverse_bn = nn.BatchNorm2d(dim)
        
        # Learnable weights for wavelet coefficients
        self.scale = 0.02
        self.haar_weight = torch.nn.Parameter(self.scale * torch.randn(self.num_blocks, self.block_size, self.block_size))
        
        # Convolution layers for skip connections
        self.conv1 = nn.Conv2d(self.block_size, self.block_size, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(self.block_size, self.block_size, kernel_size=3, padding=1)

        self.conv1_bn = nn.BatchNorm2d(dim)
        self.conv2_bn = nn.BatchNorm2d(dim)

        self.haar_conv_neuron = MultiStepLIFNode(tau=2.0, detach_reset=True)

        self.haar_matrix_built = False

        self.pool = Erode()

    #@torch.compile
    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)
        
    #@torch.compile
    def forward(self, x):
        T, B, C, H, W = x.shape #T, B, C, H, W

        if not self.haar_matrix_built:
            self.haar_forward.build(H, x.device)
            self.haar_inverse.build(H, x.device)
            self.haar_matrix_built=True

        x = self.x_neuron(x) #T, B, C, H, W
        x = self.pool(x)

        # Wavelet decomposition
        haar = self.haar_forward(x) #T, B, C, H, W
        haar = self.haar_forward_bn(haar.flatten(0, 1).contiguous()).reshape(T, B, C, H, W).contiguous()
        haar = self.haar_neuron(haar)

        # Convolution on wavelet coefficients 
        haar = haar.reshape(T*B, self.num_blocks, self.block_size, H, W).permute(0, 3, 4, 1, 2).contiguous() #T*B, h, w, numblocks, blocksize
        haar = self.multiply(haar, self.haar_weight)

        #T*B, h, w, numblocks, blocksize --> T*B, N, C, --> T*B, C, N
        haar = haar.reshape(T*B, H, W, C).permute(0, 3, 1, 2).contiguous()
        haar = self.haar_multiply_bn(haar).reshape(T, B, C, H, W).contiguous()#T*B*,C, N --> T, B, C, h, w

        # Inverse wavelet transform
        haar = self.haar_inverse(haar) #T, B, C, h, w
        haar = self.haar_inverse_bn(haar.reshape(T*B, C, H, W)).reshape(T, B, C, H, W).contiguous() #T, B, C, h, w
        haar = self.pool(haar)
        
        # ---------time domain
        x = haar.reshape(T*B*self.num_blocks, self.block_size, H, W).contiguous() # for conv: T*B*block_num, block_size, h, w
        
        # Skip connections
        conv_1 = self.conv1(x) # T*B*block_num, block_size, h, w
        conv_1 = self.conv1_bn(conv_1.reshape(T* B, -1, H, W)).reshape(T, B, -1, H, W).contiguous() #T, B, C, h, w
        
        conv_2 = self.conv2(x) # T* B, C, h, w
        conv_2 = self.conv2_bn(conv_2.reshape(T* B, -1, H, W)).reshape(T, B, -1, H, W).contiguous()  #T, B, C, h, w
        
        # Aggregate branches
        out = conv_1 + conv_2 #T, B, C, H, W
        
        return out

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., FLblocks = 16):
        super().__init__()
        self.mixer = FATM(dim, FLblocks = FLblocks)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.mixer(x) #T, B, C, H, W
        x = x + self.mlp(x) #T, B, C, H, W
        return x


class SPS(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj_conv = nn.Conv2d(in_channels, embed_dims//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims//8)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv1 = nn.Conv2d(embed_dims//8, embed_dims//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims//4)
        self.proj_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv2 = nn.Conv2d(embed_dims//4, embed_dims//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims//2)
        self.proj_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = MultiStepLIFNode(tau=2.0, detach_reset=True)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        # self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.proj_conv(x.flatten(0, 1).contiguous()) # T*B, C, H, W
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous() # T, B, C, H, W
        x = self.proj_lif(x) # T, B, C, H, W
        x = self.maxpool(x.flatten(0, 1).contiguous()) # T * B, C, H, W

        x = self.proj_conv1(x) 
        x = self.proj_bn1(x).reshape(T, B, -1, 64, 64).contiguous()
        x = self.proj_lif1(x) # T, B, C, H, W
        x = self.maxpool1(x.flatten(0, 1).contiguous()) # T * B, C, H, W

        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, 32, 32).contiguous()
        x = self.proj_lif2(x)
        x = self.maxpool2(x.flatten(0, 1).contiguous())

        x = self.proj_conv3(x) # T*B, C, H, W
        x = self.proj_bn3(x).reshape(T * B, -1, 16, 16).contiguous()
        x = self.maxpool3(x)

        x_feat = x
        x = self.proj_lif3(x.reshape(T, B, -1, H//16, W//16)).contiguous()
        x = self.rpe_conv(x.flatten(0, 1).contiguous())
        x = self.rpe_bn(x)
        x = (x + x_feat).reshape(T, B, -1, (H//16), (W//16)).contiguous()
        return x


class SWformer(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], mlp_ratios=[4, 4, 4], FL_blocks = 16, 
                 drop_rate=0., depths=[6, 8, 6], T = 4
                 ):
        super().__init__()

        self.num_classes = num_classes
        self.depths = depths
        self.T = T

        patch_embed = SPS(img_size_h=img_size_h,
                                 img_size_w=img_size_w,
                                 patch_size=patch_size,
                                 in_channels=in_channels,
                                 embed_dims=embed_dims)

        self.FLblocks = FL_blocks

        block = nn.ModuleList([Block(
            dim=embed_dims,  mlp_ratio=mlp_ratios, drop=drop_rate, FLblocks=self.FLblocks)
            for j in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)

        # classification head
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):

        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")

        x = patch_embed(x)
        for blk in block:
            x = blk(x)
        
        return x.flatten(-2).mean(3) # T, B, C, H, W --> T, B, C, N --> T, B, C

    def forward(self, x):
        if len(x.shape) < 5:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        else:
            x = x.transpose(0, 1).contiguous()

        x = self.forward_features(x)
        x = self.head(x)

        return x


@register_model
def swformer(pretrained= False, pretrained_cfg=None, **kwargs):
    model = SWformer(
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


if __name__ == '__main__':
    model = SWformer(
        patch_size=16, embed_dims=256,  mlp_ratios=4,
        in_channels=2, num_classes=10,  depths=2
    )

    input = torch.randn(16, 4, 2, 128, 128)
    output = model(input)
    print(output.shape)

