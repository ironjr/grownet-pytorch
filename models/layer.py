import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SeparableConv(nn.Module):
    def __init__(self, in_planes, planes, kernel_size=3, padding=1, stride=1, bias=False):
        super(SeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, padding=padding, stride=stride, bias=True, groups=in_planes)
        self.pointwise = nn.Conv2d(in_planes, planes, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class Node(nn.Module):
    def __init__(self, in_planes, planes, fin, downsample=False, depthwise=False, monitor=None, opt=dict(alpha=0.9), device='cuda'):
        super(Node, self).__init__()
        stride = 2 if downsample else 1
        self.depthwise = depthwise
        self._monitor = monitor
        self._monitor_opt = opt
        if opt is not None:
            # Coefficient for running mean
            if 'alpha' in opt:
                self.alpha = opt['alpha']
        self.device = device

        # Top layer receives input from aggregation node (one)
        # NOTE: Top layer nodes require input weight
        if fin == 0:
            fin = 1
        self.strengths = [0,] * fin
        self.nsamples = 0
        self.fin = fin
        self.w = nn.Parameter(torch.randn((1, fin))) # TODO Initialization?
        if depthwise:
            self.conv = SeparableConv(in_planes, planes, stride=stride)
        else:
            self.conv = nn.Conv2d(
                in_planes, planes, kernel_size=3, padding=1, stride=stride
            )
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        '''

        Arguments:
            x: A Tensor with fanin dimension in size (N, C, H, W, F)

        Returns:
            Single Tensor with size (N, C, H, W)
        '''
        if self._monitor is not None:
            x = x * self.w # (N,Cin,H,W,F)

            # For each fanin branch, evaluate average norm
            numel = x[:,:,:,:,0].numel()
            for i in range(x.size(4)):
                # Monitored value consists of the parameter and its statistics
                # First, we consider about the parameter and then retrieve its
                # statistics

                # Euclidean norm
                if self._monitor['param'] == 'l2norm':
                    strength = torch.norm(x[:,:,:,:,i].data) / numel
                # maximum (sup norm)
                elif self._monitor['param'] == 'max':
                    strength = torch.max(x[:,:,:,:,i].data)
                else:
                    raise NotImplementedError

                # simple moving average
                if self._monitor['stat'] == 'ma':
                    strength = self.strengths[i] * self.alpha + \
                            strength * \
                            (1 - self.alpha)
                # cumulative moving average
                elif self._monitor['stat'] == 'cma':
                    strength = (self.nsamples * self.strengths[i] + strength) / \
                            (self.nsamples + 1)
                    self.nsamples += 1
                else:
                    raise NotImplementedError

                self.strengths[i] = strength.item()
            x = torch.sum(x, 4).squeeze(-1) # (N,Cin,H,W)
        else:
            x = F.linear(x, self.w).squeeze(-1) # (N,Cin,H,W)
        out = self.bn(self.conv(F.relu(x)))
        return out

    def add_input_edge(self, index, normalization=None):
        if index >= self.fin:
            # Append at the end
            index = self.fin
        w = self.w.data.cpu()
        w = torch.cat((w[:, :index], torch.randn(1, 1) * torch.norm(w), w[:, index:]), 1)
        self.w = nn.Parameter(w.to(self.device))
        self.fin += 1
        self.strengths = self.strengths[:index] + [0,] + self.strengths[index:]

    def del_input_edge(self, index):
        assert index < self.fin, 'index unavailable'
        w = self.w.data.cpu()
        w = torch.cat((w[:, :index], w[:, (index + 1):]), 1)
        self.w = nn.Parameter(w.to(self.device))
        self.fin -= 1
        self.strengths = self.strengths[:index] + self.strengths[(index + 1):]

    def scale_input_edge(self, index, scale):
        self.w[0, index] *= scale

    def begin_monitor(self, monitor):
        self._monitor = monitor
        self.strengths = [0,] * self.fin
        self.nsamples = 0

    def stop_monitor(self):
        self._monitor = None

