import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import networkx as nx

from graph import get_graphs
from layer import SeparableConv
from randomnet import RandomNetwork
from util import *


class RandWireRegular(nn.Module):
    def __init__(self, Gs, nmaps=None, num_classes=1000, planes=109, drop_edge=0.1, dropout=0.2): #, cfg=None
        '''RandWire network in regular regime from Saining Xie et al. (Apr, 2019)

        Arguments:
            Gs (list(DiGraph)) : A dict of DAGs from random graph generator
            nmaps (list(dict)): saved node id map for optimal traversal
            num_classes (int): number of classes in classifier
            planes (int): number of channels after conv1 layer (C in the paper), small(78), regular(109|154)
            drop_edge (float): dropout probability of dropping edge
            dropout (float): dropout probability in the fully connected layer
        '''
        super(RandWireRegular, self).__init__()
        self.planes = planes
        self.num_classes = num_classes
        half_planes = math.ceil(planes / 2)

        if nmaps is None:
            nmaps = [None,] * 4

        self.layer1 = nn.Sequential(
                SeparableConv(3, half_planes, stride=2),
                nn.BatchNorm2d(half_planes))

        # Since the first random layer is the memory bottleneck, its size is halved
        self.layer2 = RandomNetwork(half_planes, planes, Gs[0], drop_edge=drop_edge, nmap=nmaps[0])
        self.layer3 = RandomNetwork(planes, 2 * planes, Gs[1], drop_edge=drop_edge, nmap=nmaps[1])
        self.layer4 = RandomNetwork(2 * planes, 4 * planes, Gs[2], drop_edge=drop_edge, nmap=nmaps[2])
        self.layer5 = RandomNetwork(4 * planes, 8 * planes, Gs[3], drop_edge=drop_edge, nmap=nmaps[3])

        #  self.conv6 = nn.Conv2d(8 * planes, 1280, kernel_size=1)
        self.conv6 = SeparableConv(8 * planes, 1280, kernel_size=1, bias=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1280, num_classes)
        self.dropout = nn.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        out = self.avgpool(self.conv6(out))
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
    
    def get_graphs(self):
        return [layer.G for layer in [self.layer2, self.layer3, self.layer4, self.layer5]], \
                [layer.nmap for layer in [self.layer2, self.layer3, self.layer4, self.layer5]]

    def get_complexity(self, feature_size):
        if isinstance(feature_size, (list, tuple)):
            feature_size = feature_size[0] * feature_size[1]
        else:
            feature_size *= feature_size

        # Evaluate the complexity of random nets
        half_planes = math.ceil(self.planes / 2)
        layer_nparams = [
            3 * (9 + half_planes),
            self.layer2.nparams,
            self.layer3.nparams,
            self.layer4.nparams,
            self.layer5.nparams,
        ]
        layers = [
            None,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5,
        ]
        
        flops = 0
        nparams = 0
        for i, (n, l) in enumerate(zip(layer_nparams, layers)):
            # Conv
            feature_size /= 4
            flops += feature_size * n
            nparams += n
            
            # ReLU
            if i > 0:
                ntops = len([n for n, deg in l.in_degree if deg == 0])
                nnodes = len(l.nodes)
                flops += feature_size * (ntops * l.in_planes + (nnodes - ntops) * l.planes)

        # Evaluate the complexity of classification nets
        # Layer conv6 has bias term
        n = 8 * self.planes * (1 + 1280)
        flops += n * feature_size
        nparams += n
        # Fully connected layer also has bias term
        n = (1280 + 1) * self.num_classes
        flops += n
        nparams += n

        return int(flops), nparams


class RandWireSmall(nn.Module):
    def __init__(self, Gs, nmaps=None, num_classes=1000, planes=78): #, cfg=None
        '''RandWire network in small regime from Saining Xie et al. (Apr, 2019)

        Arguments:
            Gs (list(DiGraph)) : A list of DAGs from random graph generator
            nmaps (list(dict)): saved node id map for optimal traversal
            num_classes (int): number of classes in classifier
            planes (int): number of channels after conv1 layer (C in the paper), small(78), regular(109|154)
        '''
        super(RandWireSmall, self).__init__()
        self.planes = planes
        self.num_classes = num_classes
        half_planes = math.ceil(planes / 2)

        if nmaps is None:
            nmaps = [None,] * 3

        self.layer1 = nn.Sequential(
                SeparableConv(3, half_planes, stride=2),
                nn.BatchNorm2d(half_planes))

        self.layer2 = nn.Sequential(
                nn.ReLU(inplace=True),
                SeparableConv(half_planes, planes, stride=2),
                nn.BatchNorm2d(planes))

        self.layer3 = RandomNetwork(planes, planes, Gs[0], nmap=nmaps[0])
        self.layer4 = RandomNetwork(planes, 2 * planes, Gs[1], nmap=nmaps[1])
        self.layer5 = RandomNetwork(2 * planes, 4 * planes, Gs[2], nmap=nmaps[2])

        #  self.conv6 = nn.Conv2d(4 * planes, 1280, kernel_size=1)
        self.conv6 = SeparableConv(4 * planes, 1280, kernel_size=1, bias=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1280, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        out = self.avgpool(self.conv6(out))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
    def get_graphs(self):
        return [layer.G for layer in [self.layer3, self.layer4, self.layer5]], \
                [layer.nmap for layer in [self.layer3, self.layer4, self.layer5]]

    def get_complexity(self, feature_size):
        if isinstance(feature_size, (list, tuple)):
            feature_size = feature_size[0] * feature_size[1]
        else:
            feature_size *= feature_size

        # Evaluate the complexity of random nets
        half_planes = math.ceil(self.planes / 2)
        layer_nparams = [
            3 * (9 + half_planes),
            half_planes * (9 + self.planes),
            self.layer3.nparams,
            self.layer4.nparams,
            self.layer5.nparams,
        ]
        layers = [
            None,
            None,
            self.layer3,
            self.layer4,
            self.layer5,
        ]
        
        flops = 0
        nparams = 0
        for i, (n, l) in enumerate(zip(layer_nparams, layers)):
            # Conv
            feature_size /= 4
            flops += feature_size * n
            nparams += n

            # ReLU
            if i == 1:
                flops += feature_size * half_planes
            elif i > 1:
                ntops = len([n for n, deg in l.in_degree if deg == 0])
                nnodes = len(l.nodes)
                flops += feature_size * (ntops * l.in_planes + (nnodes - ntops) * l.planes)

        # Evaluate the complexity of classification nets
        # Layer conv6 has bias term
        n = 4 * self.planes * (1 + 1280)
        flops += n * feature_size
        nparams += n
        # Fully connected layer also has bias term
        n = (1280 + 1) * self.num_classes
        flops += n
        nparams += n
        
        return int(flops), nparams


class RandWireTiny(nn.Module):
    def __init__(self, Gs, nmaps=None, num_classes=10, planes=32, depthwise=False, drop_edge=0, dropout=0): #, cfg=None
        '''RandWire network in tiny regime for CIFAR training

        Arguments:
            Gs (list(DiGraph)) : A list of DAGs from random graph generator
            nmaps (list(dict)): saved node id map for optimal traversal
            num_classes (int): number of classes in classifier
            planes (int): number of channels after conv1 layer (C in the paper), small(78), regular(109|154)
            depthwise (bool): whether to use depthwise separable convolution
            drop_edge (float): dropout probability of dropping edge
            dropout (float): dropout probability in the fully connected layer
        '''
        super(RandWireTiny, self).__init__()
        self.depthwise = depthwise
        self.planes = planes
        self.num_classes = num_classes
        half_planes = math.ceil(planes / 2)

        if nmaps is None:
            nmaps = [None,] * 4

        if depthwise:
            self.layer1 = nn.Sequential(
                SeparableConv(3, half_planes, stride=2),
                nn.BatchNorm2d(half_planes)
            )
        else:
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, half_planes, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(half_planes)
            )

        self.layer2 = RandomNetwork(half_planes, planes, Gs[0],
                drop_edge=drop_edge, nmap=nmaps[0], downsample=False,
                depthwise=depthwise)
        self.layer3 = RandomNetwork(planes, 2 * planes, Gs[1],
                drop_edge=drop_edge, nmap=nmaps[1], depthwise=depthwise)
        self.layer4 = RandomNetwork(2 * planes, 4 * planes, Gs[2],
                drop_edge=drop_edge, nmap=nmaps[2], depthwise=depthwise)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4 * planes, num_classes)
        self.dropout = nn.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
    
    def get_graphs(self):
        return [layer.G for layer in [self.layer2, self.layer3, self.layer4]], \
                [layer.nmap for layer in [self.layer2, self.layer3, self.layer4]]

    def get_complexity(self, feature_size):
        if isinstance(feature_size, (list, tuple)):
            feature_size = feature_size[0] * feature_size[1]
        else:
            feature_size *= feature_size

        # Evaluate the complexity of random nets
        half_planes = math.ceil(self.planes / 2)
        if self.depthwise:
            layer1_nparams = 3 * (9 + half_planes)
        else:
            layer1_nparams = 3 * (9 * half_planes)
        layer_nparams = [
            layer1_nparams,
            self.layer2.nparams,
            self.layer3.nparams,
            self.layer4.nparams,
        ]
        layers = [
            None,
            self.layer2,
            self.layer3,
            self.layer4,
        ]
        
        flops = 0
        nparams = 0
        for i, (n, l) in enumerate(zip(layer_nparams, layers)):
            # Conv
            feature_size /= 4
            flops += feature_size * n
            nparams += n

            # ReLU
            if i == 1:
                flops += feature_size * half_planes
            elif i > 1:
                ntops = len([n for n, deg in l.in_degree if deg == 0])
                nnodes = len(l.nodes)
                flops += feature_size * (ntops * l.in_planes + (nnodes - ntops) * l.planes)

        # Evaluate the complexity of classification nets
        # Fully connected layer also has bias term
        n = (4 * self.planes + 1) * self.num_classes
        flops += n
        nparams += n
        
        return int(flops), nparams


# Wrappers for both network and graph configuration
def RandWireSmall78(Gs=None, nmaps=None, model=None, params=None, nnodes=32, num_classes=1000, seeds=None):
    assert (Gs is not None or (model is not None and params is not None)), 'Graph or its generating method should be given'
    if Gs is None:
        # Generate random graph
        Gs = get_graphs(model, params, 3, nnodes, seeds)

    # Generate network from graph configurations
    net = RandWireSmall(Gs=Gs, nmaps=nmaps, num_classes=num_classes, planes=78)
    Gs, nmaps = net.get_graphs()
    return net, Gs, nmaps

def RandWireRegular109(Gs=None, nmaps=None, model=None, params=None, nnodes=32, num_classes=1000, seeds=None):
    assert (Gs is not None or (model is not None and params is not None)), 'Graph or its generating method should be given'
    if Gs is None:
        # Generate random graph
        nnodes = [nnodes // 2, nnodes, nnodes, nnodes]
        Gs = get_graphs(model, params, 4, nnodes, seeds)
    
    # Generate network from graph configurations
    net = RandWireRegular(Gs=Gs, nmaps=nmaps, num_classes=num_classes, planes=109)
    Gs, nmaps = net.get_graphs()
    return net, Gs, nmaps

def RandWireRegular154(Gs=None, nmaps=None, model=None, params=None, nnodes=32, num_classes=1000, seeds=None):
    assert (Gs is not None or (model is not None and params is not None)), 'Graph or its generating method should be given'
    if Gs is None:
        # Generate random graph
        nnodes = [nnodes // 2, nnodes, nnodes, nnodes]
        Gs = get_graphs(model, params, 4, nnodes, seeds)
    
    # Generate network from graph configurations
    net = RandWireRegular(Gs=Gs, nmaps=nmaps, num_classes=num_classes, planes=154)
    Gs, nmaps = net.get_graphs()
    return net, Gs, nmaps

# CIFAR training
def RandWireTinyNormal(Gs=None, nmaps=None, model=None, params=None, nnodes=32, num_classes=10, seeds=None, depthwise=False, drop_edge=0, dropout=0):
    assert (Gs is not None or (model is not None and params is not None)), 'Graph or its generating method should be given'
    if Gs is None:
        # Generate random graph
        nnodes = [nnodes // 2, nnodes, nnodes]
        Gs = get_graphs(model, params, 3, nnodes, seeds)
    
    # Generate network from graph configurations
    net = RandWireTiny(
        Gs=Gs,
        nmaps=nmaps,
        num_classes=num_classes,
        planes=18,
        depthwise=depthwise,
        drop_edge=drop_edge,
        dropout=dropout
    )
    Gs, nmaps = net.get_graphs()
    return net, Gs, nmaps

def RandWireTinyWide(Gs=None, nmaps=None, model=None, params=None, nnodes=32, num_classes=10, seeds=None, depthwise=False, drop_edge=0, dropout=0):
    assert (Gs is not None or (model is not None and params is not None)), 'Graph or its generating method should be given'
    if Gs is None:
        # Generate random graph
        nnodes = [nnodes, nnodes, nnodes]
        Gs = get_graphs(model, params, 3, nnodes, seeds)
    
    # Generate network from graph configurations
    net = RandWireTiny(
        Gs=Gs,
        nmaps=nmaps,
        num_classes=num_classes,
        planes=16,
        depthwise=depthwise,
        drop_edge=drop_edge,
        dropout=dropout
    )
    Gs, nmaps = net.get_graphs()
    return net, Gs, nmaps


def test():
    from visualize import draw_graph, draw_network

    graph_type= 'WS'
    graph_params = {
        'P': 0.75,
        'K': 4,
    }
    network_list = [RandWireTinyWide,]#[RandWireSmall78, RandWireRegular109, RandWireRegular154]

    test_complexity(network_list, graph_type, graph_params, input_size=32, num_samples=10)

    #  for gen in network_list:
    #      net, Gs, _ = gen(model=graph_type, params=graph_params, seeds=None)
    #
    #      # Evaluate the network
    #      x = torch.randn(16, 3, 32, 32)
    #      out = net(x)
    #      print(out.size())
    #
    #      # Draw the network
    #      for G in Gs:
    #          draw_graph(G)
    #      draw_network(net, x, label=gen.__name__)
    #
    #      # Pause
    #      input()


if __name__ == '__main__':
    test()
