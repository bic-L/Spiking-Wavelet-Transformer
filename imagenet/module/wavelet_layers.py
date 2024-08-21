import torch
from torch import nn
import numpy as np

backend = 'torch'
import torch._dynamo as _dynamo 
_dynamo.config.suppress_errors = True

def normalize_haar_matrix(H, device):
    # Compute norms of rows
    norms = torch.linalg.norm(H, axis=1, keepdims=True)
    # Divide each row by its norm
    H = H / norms
    return H.to(device).to(torch.float32)

def haar_matrix(N, device):
    H = torch.tensor(haar_1d_matrix(N)).to(torch.float32)
    return normalize_haar_matrix(H, device)

def haar_1d_matrix(n):
    # This function generates an nxn Haar matrix
    # n must be a power of 2
    
    # Check if n is a power of 2
    if np.log2(n) % 1 > 0:
        raise ValueError("n must be a power of 2")

    # Initial condition
    if n == 1:
        return np.array([[1]])
    
    # Recursive case
    else:
        # Recursive call
        H_next = haar_1d_matrix(n // 2)
        # Create the upper part
        upper = np.kron(H_next, [1, 1])
        # Create the lower part
        lower = np.kron(np.eye(len(H_next)), [1, -1])
        # Stack them together
        H = np.vstack((upper, lower))
        return H


class Haar1DForward(nn.Module):

    def __init__(self, neuron_type, vth=1.0):
        super().__init__()
        self.haar_neuron = neuron_type(v_threshold = vth)

    def build(self, N):
        self.H = haar_matrix(N)

    @torch.compile
    def haar_1d(self, x):
        return torch.matmul(self.H, x)

    @torch.compile
    def forward(self, x):
        haar = self.haar_1d(x)
        return self.haar_neuron(haar)
    

class Haar1DInverse(nn.Module):

    def __init__(self, neuron_type, vth=1.0):
        super().__init__()
        #self.H = haar_matrix(N)
        self.haar_inv_neu = neuron_type(v_threshold = vth)

    def build(self, N):
        self.H = haar_matrix(N)

    def haar_1d_inverse(self, x):
        return torch.matmul(self.H.T, x)

    @torch.compile
    def forward(self, x):
        haar_inverse = self.haar_1d_inverse(x)
        return self.haar_inv_neu(haar_inverse)


class Haar2DForward(nn.Module):

    def __init__(self, neuron_type, vth=1.0, layer=0):
        super().__init__()
        
        self.row_haar_neuron = neuron_type(v_threshold = vth, backend=backend)
        self.col_haar_neuron = neuron_type(v_threshold = vth, backend=backend)
        self.layer = layer

    def build(self, N, device):
        self.H = haar_matrix(N, device)

    @torch.compile
    def forward(self, x):
        # Apply the 1D Haar transform to each row of the image
        #x = self.row_haar_neuron(x) # input has been converted, removed
        x = torch.matmul(x, self.H.T)
        
        # Apply the 1D Haar transform to each column of the transformed rows
        x = self.col_haar_neuron(x)    
        x = torch.matmul(self.H, x)
        return x

class Haar2DInverse(nn.Module):

    def __init__(self, neuron_type, vth=1.0, layer=0):
        super().__init__()
        self.row_haar_neuron = neuron_type(v_threshold = vth)
        self.col_haar_neuron = neuron_type(v_threshold = vth)
        self.layer = layer

    def build(self, N, device):
        self.H = haar_matrix(N, device)

    @torch.compile
    def forward(self, x): #T, B, C, H, W
        # Apply the 1D Haar transform to each row of the image
        x = self.row_haar_neuron(x)
        x = torch.matmul(x, self.H)

        # # Apply the 1D Haar transform to each column of the transformed rows
        x = self.col_haar_neuron(x)  
        x = torch.matmul(self.H.T, x)
        return x #T, B, C, H, W
