import torch.nn as nn
import torch
from timm.models.layers import DropPath
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
from .neuron import MultiStepNegIFNode
from .wavelet_layers import Haar2DForward, Haar2DInverse
backend = 'torch'
import torch._dynamo as _dynamo 
_dynamo.config.suppress_errors = True

class MS_MLP_Conv(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.res = in_features == hidden_features
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.fc2_conv = nn.Conv2d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm2d(out_features)
        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape
        identity = x

        x = self.fc1_lif(x)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, H, W).contiguous()
        if self.res:
            x = identity + x
            identity = x
        x = self.fc2_lif(x)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()

        x = x + identity
        return x


class FATM(nn.Module):
    def __init__(
        self,
        dim,
        haar_vth=1.,
        FL_blocks = 16,
    ):
        super().__init__()
        self.hidden_size = dim

        self.num_blocks = FL_blocks
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0

        self.x_neuron = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.haar_neuron = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)

        # 2D discrete wavelet transform
        self.haar_forward = Haar2DForward(MultiStepNegIFNode, vth=haar_vth)
        self.haar_inverse = Haar2DInverse(MultiStepNegIFNode, vth=haar_vth)
        
        self.scale = 0.02
        # Learnable weights for wavelet coefficients
        self.haar_weight = torch.nn.Parameter(self.scale * torch.randn(self.num_blocks, self.block_size, self.block_size))
        self.haar_forward_bn = nn.BatchNorm2d(dim)
        self.haar_bn = nn.BatchNorm2d(dim)

        # Convolution layers for skip connections
        self.conv1 = nn.Conv2d(self.block_size, self.block_size, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(self.block_size, self.block_size, kernel_size=3, padding=1)

        self.conv1_bn = nn.BatchNorm2d(dim)
        self.conv2_bn = nn.BatchNorm2d(dim)

        self.haar_inverse_neuron = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend)
        self.haar_inverse_bn = nn.BatchNorm2d(dim)

        self.haar_matrix_built = False

    @torch.compile
    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)

    @torch.compile
    def forward(self, x):
        T, B, C, H, W = x.shape #T, B, C, H, W
        identity = x

        if not self.haar_matrix_built:
            self.haar_forward.build(H, x.device)
            self.haar_inverse.build(H, x.device)
            self.haar_matrix_built=True

        x = self.x_neuron(x) #T, B, C, H, W
            
        # Wavelet decomposition
        haar = self.haar_forward(x) #T, B, C, H, W
        haar = self.haar_forward_bn(haar.flatten(0, 1).contiguous()).reshape(T, B, C, H, W).contiguous()
        haar = self.haar_neuron(haar)

        # Convolution on wavelet coefficients 
        haar = haar.reshape(T*B, self.num_blocks, self.block_size, H, W).permute(0, 3, 4, 1, 2).contiguous() #T*B, h, w, numblocks, blocksize
        haar = self.multiply(haar, self.haar_weight)

        #T*B, h, w, numblocks, blocksize --> T*B, H, W, C, --> T*B, C, H, W
        haar = haar.reshape(T*B, H, W, C).permute(0, 3, 1, 2).contiguous()

        #T*B, C, H, W --> T, B, C, h, w
        haar = self.haar_bn(haar).reshape(T, B, C, H, W).contiguous()#T, B, C, h, w

        # Inverse wavelet transform
        haar = self.haar_inverse(haar) #T, B, C, h, w
        haar = self.haar_inverse_bn(haar.reshape(T*B, C, H, W)).reshape(T, B, C, H, W).contiguous() #T, B, C, h, w

        # ---------time domain
        x = x.reshape(T*B*self.num_blocks, self.block_size, H, W).contiguous() # for conv: T*B*block_num, block_size, h, w

        # Skip connections
        conv_1 = self.conv1(x) # T*B*block_num, block_size, h, w
        conv_1 = self.conv1_bn(conv_1.reshape(T* B, -1, H, W)).reshape(T, B, -1, H, W).contiguous() #T, B, C, h, w

        conv_2 = self.conv2(x) # T* B, C, h, w
        conv_2 = self.conv2_bn(conv_2.reshape(T* B, -1, H, W)).reshape(T, B, -1, H, W).contiguous()  #T, B, C, h, w
        
        # Aggregate branches
        out = haar + conv_1 + conv_2 #T, B, C, H, W

        out = out + identity
        return out


class MS_Block_Conv(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        haar_vth=1.,
        FL_blocks = 16,
    ):
        super().__init__()
        self.attn = FATM(
            dim=dim,
            haar_vth=haar_vth,
            FL_blocks = FL_blocks,
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP_Conv(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
        )

    def forward(self, x):
        x_attn = self.attn(x)
        x = self.mlp(x_attn)
        return x
