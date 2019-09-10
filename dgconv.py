import torch
import torch.nn as nn
import random
from torch.nn import functional as F
import math


def aggregate(gate, D, I, K, sort=False):
    if sort:
        _, ind = gate.sort(descending=True)
        gate = gate[:, ind[0, :]]

    U = [(gate[0, i] * D + gate[1, i] * I) for i in range(K)]
    while len(U) != 1:
        temp = []
        for i in range(0, len(U) - 1, 2):
            temp.append(kronecker_product(U[i], U[i + 1]))
        if len(U) % 2 != 0:
            temp.append(U[-1])
        del U
        U = temp

    return U[0], gate


def kronecker_product(mat1, mat2):
    return torch.ger(mat1.view(-1), mat2.view(-1)).reshape(*(mat1.size() + mat2.size())).permute(
        [0, 2, 1, 3]).reshape(mat1.size(0) * mat2.size(0), mat1.size(1) * mat2.size(1))


class DGConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, sort=False):
        super(DGConv2d, self).__init__()
        self.register_buffer('D', torch.eye(2))
        self.register_buffer('I', torch.ones(2, 2))
        self.K = int(math.log2(in_channels))
        gate_init = [-1e-8 for _ in range(self.K)]
        self.register_parameter('gate', nn.Parameter(torch.Tensor(gate_init)))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sort = sort

    def forward(self, x):
        setattr(self.gate, 'org', self.gate.data.clone())
        self.gate.data = ((self.gate.org - 0).sign() + 1) / 2.
        U_regularizer =  2 ** (self.K  + torch.sum(self.gate))
        gate = torch.stack((1 - self.gate, self.gate))
        self.gate.data = self.gate.org # Straight-Through Estimator
        U, gate = aggregate(gate, self.D, self.I, self.K, sort=self.sort)
        masked_weight = self.conv.weight * U.view(self.out_channels, self.in_channels, 1, 1)
        x = F.conv2d(x, masked_weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation)
        return x, U_regularizer

