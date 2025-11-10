"""
Defines the neural network architectures used by the PDE solvers.

Includes:
- Standard feed-forward networks (Net1, TimeNet)
- Set-Transformer components (MAB, SAB, ISAB, PMA)
- Deep BSDE models (DeepBSDE, FBSNNNetwork, TimeDependentNetwork)

Set-Transformer code based on: https://github.com/juho-lee/set_transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Linear, BatchNorm1d, Tanh, ReLU
import math
from typing import List

# --- Set Transformer Components ---

class MAB(Module):
    """Multi-Head Attention Block (MAB)."""
    def __init__(self, dim_q: int, dim_k: int, dim_v: int, num_heads: int, ln: bool = False):
        super(MAB, self).__init__()
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.fc_q = Linear(dim_q, dim_v)
        self.fc_k = Linear(dim_k, dim_v)
        self.fc_v = Linear(dim_k, dim_v)
        self.ln0 = nn.LayerNorm(dim_v) if ln else None
        self.ln1 = nn.LayerNorm(dim_v) if ln else None
        self.fc_o = Linear(dim_v, dim_v)

    def forward(self, q: Tensor, k: Tensor) -> Tensor:
        q_fc = self.fc_q(q)
        k_fc, v_fc = self.fc_k(k), self.fc_v(k)

        dim_split = self.dim_v // self.num_heads
        q_ = torch.cat(q_fc.split(dim_split, 2), 0)
        k_ = torch.cat(k_fc.split(dim_split, 2), 0)
        v_ = torch.cat(v_fc.split(dim_split, 2), 0)

        attention = torch.softmax(q_.bmm(k_.transpose(1, 2)) / math.sqrt(self.dim_v), 2)
        out = torch.cat((q_ + attention.bmm(v_)).split(q.size(0), 0), 2)
        
        out = self.ln0(out) if self.ln0 is not None else out
        out = out + F.relu(self.fc_o(out))
        out = self.ln1(out) if self.ln1 is not None else out
        return out


class SAB(Module):
    """Self-Attention Block (SAB)."""
    def __init__(self, dim_in: int, dim_out: int, num_heads: int, ln: bool = False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, x: Tensor) -> Tensor:
        return self.mab(x, x)


class ISAB(Module):
    """Induced Self-Attention Block (ISAB)."""
    def __init__(self, dim_in: int, dim_out: int, num_heads: int, num_inds: int, ln: bool = False):
        super(ISAB, self).__init__()
        self.i = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.i)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, x: Tensor) -> Tensor:
        h = self.mab0(self.i.repeat(x.size(0), 1, 1), x)
        return self.mab1(x, h)


class PMA(Module):
    """Pooling Multi-Head Attention (PMA)."""
    def __init__(self, dim: int, num_heads: int, num_seeds: int, ln: bool = False):
        super(PMA, self).__init__()
        self.s = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.s)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, x: Tensor) -> Tensor:
        return self.mab(self.s.repeat(x.size(0), 1, 1), x)


class TimeNetForSet(Module):
    """
    Time-dependent feature transformation for SetTransformer.
    
    Applies separate linear layers to time (t) and features (x),
    then combines them as exp(t) * x.
    """
    def __init__(self, in_features: int = 1, out_features: int = 64):
        super(TimeNetForSet, self).__init__()
        self.feature_layer = Linear(in_features, out_features)
        
        self.time_layer1 = Linear(1, 10)
        self.time_layer2 = Linear(10, 10)
        self.time_layer3 = Linear(10, 10)
        self.time_layer4 = Linear(10, out_features)
        
        self.relu_stack = nn.Sequential(ReLU(), ReLU(), ReLU())

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        t_feat = self.relu_stack[0](self.time_layer1(t))
        t_feat = self.relu_stack[1](self.time_layer2(t_feat))
        t_feat = self.relu_stack[2](self.time_layer3(t_feat))
        t_feat = self.time_layer4(t_feat)
        
        x_feat = self.feature_layer(x)
        
        return torch.exp(t_feat) * x_feat


class DeepTimeSetTransformer(Module):
    """
    Full Deep Time Set Transformer model.
    """
    def __init__(self, input_dim: int):
        super(DeepTimeSetTransformer, self).__init__()
        
        # Feature extractor layers
        self.layer1 = Linear(input_dim, 32)
        self.layer2 = Linear(32, 32)
        self.layer3 = Linear(32, 32)
        self.layer4 = Linear(32, 32)
        self.layer5 = Linear(32, 32)
        self.activation = ReLU()

        # Pooling layer
        self.regressor = PMA(dim=32, num_heads=4, num_seeds=1)

        # Time-dependent output layers
        self.time_layer2 = TimeNetForSet(in_features=32, out_features=32)
        self.time_layer3 = TimeNetForSet(in_features=32, out_features=32)
        self.time_layer4 = TimeNetForSet(in_features=32, out_features=32)
        self.time_layer5 = TimeNetForSet(in_features=32, out_features=1)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        x = self.activation(self.layer1(x))
        x = x - torch.mean(x, 2, keepdim=True)
        
        x = self.activation(self.layer2(x))
        x = x - torch.mean(x, 2, keepdim=True)
        
        x = self.activation(self.layer3(x))
        x = x - torch.mean(x, 2, keepdim=True)
        
        x = self.activation(self.layer4(x))
        x = x - torch.mean(x, 2, keepdim=True)
        
        x = self.activation(self.layer5(x))
        
        output = self.regressor(x)
        output = torch.squeeze(output)

        output = self.activation(self.time_layer2(t, output))
        output = self.activation(self.time_layer3(t, output))
        output = self.activation(self.time_layer4(t, output))
        output = self.time_layer5(t, output)

        return output


# --- Standard Networks ---

class TimeNet(Module):
    """Simple feed-forward network for time features."""
    def __init__(self, output_dim: int):
        super(TimeNet, self).__init__()
        self.layers = nn.ModuleList([
            Linear(4, 100),
            Linear(100, 150),
            Linear(150, 200),
            Linear(200, 300),
            Linear(300, 200),
            Linear(200, output_dim)
        ])
        self.tanh_stack = nn.ModuleList([Tanh() for _ in range(6)])

    def forward(self, x: Tensor) -> Tensor:
        for i in range(len(self.layers)):
            x = self.tanh_stack[i](self.layers[i](x))
        return x


class Net1(Module):
    """A simple Linear + BatchNorm layer."""
    def __init__(self, input_dim: int, output_dim: int):
        super(Net1, self).__init__()
        self.layer = Linear(input_dim, output_dim)
        self.bn = BatchNorm1d(output_dim) # Note: bn is not used in forward

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)

# --- BSDE Solver Networks ---

class FBSNNNetwork(Module):
    """Feed-forward network for the FBSNN solver."""
    def __init__(self, layer_sizes: List[int]):
        super(FBSNNNetwork, self).__init__()
        self.n_layer = len(layer_sizes) - 1
        self.layers = nn.ModuleList([])
        
        for i in range(self.n_layer):
            self.layers.append(Linear(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x: Tensor) -> Tensor:
        for i in range(self.n_layer):
            x = self.layers[i](x)
            if i < self.n_layer - 1:
                x = torch.sin(x)  # Use sin activation
        return x


class DeepBSDE(Module):
    """Network for the Deep BSDE method (one net per time step)."""
    def __init__(self, layer_sizes: List[int]):
        super(DeepBSDE, self).__init__()
        self.n_layer = len(layer_sizes) - 1
        self.layers = nn.ModuleList([])
        self.batch_layer = nn.ModuleList([])

        for i in range(self.n_layer):
            self.layers.append(Linear(layer_sizes[i], layer_sizes[i + 1], bias=False))
            self.batch_layer.append(BatchNorm1d(layer_sizes[i], eps=1e-06, momentum=0.01))
            
        self.batch_layer.append(BatchNorm1d(layer_sizes[-1], eps=1e-06, momentum=0.01))
        self.activation = ReLU()

    def forward(self, x: Tensor) -> Tensor:
        # Note: Original code doesn't use batch_layer[0]
        for i in range(self.n_layer - 1):
            x = self.layers[i](x)
            # x = self.batch_layer[i+1](x) # Original code commented this out
            x = self.activation(x)
            
        x = self.layers[-1](x)
        # x = self.batch_layer[-1](x) # Original code commented this out
        return x


class TimeDependentNetwork(Module):
    """Time-dependent network for BSDE solver."""
    def __init__(self, indim: int, layersize: List[int], outdim: int):
        super(TimeDependentNetwork, self).__init__()
        self.n_layer = len(layersize)
        self.layers = nn.ModuleList([])
        self.time_layer = nn.ModuleList([])
        self.batch_layer = nn.ModuleList([])

        # Input layer
        self.layers.append(Net1(indim, layersize[0]))
        self.time_layer.append(TimeNet(indim))
        self.batch_layer.append(BatchNorm1d(indim, eps=1e-06, momentum=0.01))
        self.batch_layer.append(BatchNorm1d(layersize[0], eps=1e-06, momentum=0.01))

        # Hidden layers
        for i in range(len(layersize) - 1):
            self.layers.append(Net1(layersize[i], layersize[i + 1]))
            self.time_layer.append(TimeNet(layersize[i]))
            self.batch_layer.append(BatchNorm1d(layersize[i + 1], eps=1e-06, momentum=0.01))

        # Output layer
        self.time_layer.append(TimeNet(outdim))
        self.linear = Linear(layersize[-1], outdim)
        self.batch_layer.append(BatchNorm1d(outdim, eps=1e-06, momentum=0.01))

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        for i in range(self.n_layer):
            # (t, t**2, t**3, exp(t))
            time_features = torch.cat((t, t**2, t**3, torch.exp(t)), 1)
            time_weight = self.time_layer[i](time_features)
            
            x = self.batch_layer[i](x)
            x = x * (1 + time_weight)
            x = self.layers[i](x)
            x = torch.sin(x)

        x = self.linear(x)
        # x = self.batch_layer[-1](x) # Original code commented this out
        return x


class TimeDependentNetworkMonteCarlo(Module):
    """Time-dependent network for BSDE solver with Monte Carlo gradient."""
    def __init__(self, indim: int, layersize: List[int], outdim: int, sigma: float):
        super(TimeDependentNetworkMonteCarlo, self).__init__()
        self.n_layer = len(layersize)
        self.layers = nn.ModuleList([])
        self.time_layer = nn.ModuleList([])
        self.batch_layer = nn.ModuleList([])
        self.sigma = sigma

        # Input layer
        self.layers.append(Net1(indim, layersize[0]))
        self.time_layer.append(TimeNet(indim))
        self.batch_layer.append(BatchNorm1d(indim))
        self.batch_layer.append(BatchNorm1d(layersize[0]))

        # Hidden layers
        for i in range(len(layersize) - 1):
            self.layers.append(Net1(layersize[i], layersize[i + 1]))
            self.time_layer.append(TimeNet(layersize[i]))
            self.batch_layer.append(BatchNorm1d(layersize[i + 1]))

        # Output layer
        self.time_layer.append(TimeNet(outdim))
        self.linear = Linear(layersize[-1], outdim)
        self.activation = ReLU()

    def forward(self, t: Tensor, x: Tensor, y_mc: Tensor) -> Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        t : Tensor
            Time tensor.
        x : Tensor
            Input features (asset price).
        y_mc : Tensor
            Monte Carlo estimate of the gradient (Z).
        """
        x_prim = x # Store original input

        for i in range(self.n_layer):
            time_features = torch.cat((t, t**2, t**3, torch.exp(t)), 1)
            time_weight = self.time_layer[i](time_features)
            
            x = x * (1 + time_weight)
            x = self.layers[i](x)
            x = self.activation(x)

        # Output layer
        time_features = torch.cat((t, t**2, t**3, torch.exp(t)), 1)
        time_weight = self.time_layer[self.n_layer](time_features)
        
        # Combine network output with MC estimate
        # Z = sigma * S * Network(t,S) + (1 - time_weight) * Z_MC
        x = (self.sigma * x_prim * self.linear(x) + (1 - time_weight) * y_mc)
        return x