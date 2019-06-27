import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import networkx as nx

from graph import get_graphs
from layer import SeparableConv
from randomnet import RandomNetwork
from util import *


class RandGrowTiny(nn.Module):
    def __init__(self, Gs=None, nmaps=None, num_classes=10, planes=32, depthwise=False, drop_edge=0, dropout=0, cfg=None):
        '''RandGrow network in tiny regime for CIFAR training

        Arguments:
            Gs (list(DiGraph)) : A list of DAGs from the graph generator
            nmaps (list(dict)): saved node id map for optimal traversal
            num_classes (int): number of classes in classifier
            planes (int): number of channels after conv1 layer (C in the paper), small(78), regular(109|154)
            depthwise (bool): whether to use depthwise separable convolution
            drop_edge (float): dropout probability of dropping edge
            dropout (float): dropout probability in the fully connected layer
            cfg (dict): configuration of growing policy
        '''
        super(RandGrowTiny, self).__init__()
        self.cfg = cfg
        self.depthwise = depthwise
        self.planes = planes
        self.num_classes = num_classes
        half_planes = math.ceil(planes / 2)

        if nmaps is None:
            nmaps = [None,] * 4
        if Gs is None:
            Gs = []
            # Create simple graph
            for _ in range(3):
                G = nx.DiGraph()
                G.add_node(0)
                G.add_node(1)
                G.add_edge(0, 1)
                Gs.append(G)

        if depthwise:
            self.layer1 = nn.Sequential(
                SeparableConv(3, half_planes, stride=1),
                nn.BatchNorm2d(half_planes)
            )
        else:
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, half_planes, kernel_size=3, padding=1, stride=1),
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
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(4 * planes, num_classes)

        # Initialization
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

    def get_sublayers(self):
        layer_names = ['layer2', 'layer3', 'layer4']
        layers = [self.layer2, self.layer3, self.layer4]
        return zip(layer_names, layers)

    def begin_monitor(self):
        for _, layer in self.get_sublayers():
            layer.begin_monitor()

    def stop_monitor(self):
        for _, layer in self.get_sublayers():
            layer.stop_monitor()

    def expand(self, policy='MaxEdgeStrengthPolicy', options=None):
        '''Expand network layer by specified policy

        Arguments:
            policy (str): name of the policy to execute
            options (dict): optional arguments of the given policy
        '''
        # Information to the optimizer
        info = {}
        info['new_nodes'] = []
        info['changed_params'] = []

        if policy == 'MaxEdgeStrengthPolicy':
            # Options to define frequency of expansion of each layer
            #  if options is None:
            #

            # Expand layer by layer
            for lname, layer in self.get_sublayers():
                # Maximum edge excitation expansion policy
                weights = layer.get_edge_weights()
                edge_from, edge_to = max(weights.keys(), key=(lambda key: weights[key]))
                depthrate = self.cfg['depthrate']
                if torch.bernoulli(torch.Tensor([depthrate])) == 1:
                    info['new_nodes'].append(layer.increase_depth(edge_from, edge_to))
                else:
                    info['new_nodes'].append(layer.increase_width(edge_from, edge_to))

                # Name of the expanded node input weight
                pname = lname + '.nodes.' + str(edge_to) + '.w'
                info['changed_params'].append((pname, layer.nodes[edge_to].w,))
        elif policy == 'RandomPolicy':
            # Options to define frequency of expansion of each layer
            #  if options is None:
            #

            # Expand layer by layer
            for lname, layer in self.get_sublayers():
                # Random edge selection
                edge_from, edge_to = random.choice(list(layer.G.edges))
                depthrate = self.cfg['depthrate']
                if torch.bernoulli(torch.Tensor([depthrate])) == 1:
                    info['new_nodes'].append(layer.increase_depth(edge_from, edge_to))
                else:
                    info['new_nodes'].append(layer.increase_width(edge_from, edge_to))

                # Name of the expanded node input weight
                pname = lname + '.nodes.' + str(edge_to) + '.w'
                info['changed_params'].append((pname, layer.nodes[edge_to].w,))
            
        else:
            raise NotImplementedError
            
        return info
    
    def get_graphs(self):
        return [layer.G for layer in [self.layer2, self.layer3, self.layer4]], \
                [layer.nmap for layer in [self.layer2, self.layer3, self.layer4]]

    def get_complexity(self, feature_size):
        if isinstance(feature_size, (list, tuple)):
            feature_size = feature_size[0] * feature_size[1]
        else:
            feature_size *= feature_size

        # Evaluate the complexity of subnets
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
            flops += feature_size * n
            nparams += n

            # ReLU
            if i == 1:
                flops += feature_size * half_planes
            elif i > 1:
                ntops = len([n for n, deg in l.in_degree if deg == 0])
                nnodes = len(l.nodes)
                flops += feature_size * (ntops * l.in_planes + (nnodes - ntops) * l.planes)

            # Downsample
            feature_size /= 4

        # Evaluate the complexity of classification nets
        # Fully connected layer also has bias term
        n = (4 * self.planes + 1) * self.num_classes
        flops += n
        nparams += n
        
        return int(flops), nparams


# Wrappers for both network and graph configuration
# CIFAR training
def RandGrowTinyNormal(Gs=None, nmaps=None, num_classes=10, seeds=None, depthwise=False, drop_edge=0, dropout=0, cfg=None):
    net = RandGrowTiny(
        Gs=Gs,
        nmaps=nmaps,
        num_classes=num_classes,
        planes=16,
        depthwise=depthwise,
        drop_edge=drop_edge,
        dropout=dropout,
        cfg=cfg
    )
    Gs, nmaps = net.get_graphs()
    return net, Gs, nmaps

def RandGrowTiny18(Gs=None, nmaps=None, num_classes=10, seeds=None, depthwise=False, drop_edge=0, dropout=0, cfg=None):
    net = RandGrowTiny(
        Gs=Gs,
        nmaps=nmaps,
        num_classes=num_classes,
        planes=18,
        depthwise=depthwise,
        drop_edge=drop_edge,
        dropout=dropout,
        cfg=cfg
    )
    Gs, nmaps = net.get_graphs()
    return net, Gs, nmaps


def test():
    from visualize import draw_graph, draw_network

    network_list = [RandGrowTinyNormal,]
    cfg = {
        'depthrate': 0.2,
    }

    for gen in network_list:
        net, Gs, _ = gen(cfg=cfg)

        # Evaluate the network
        x = torch.randn(16, 3, 32, 32)
        out = net(x)
        print(out.size())
        
        # Draw the network
        for G in Gs:
            draw_graph(G)
        draw_network(net, x, label=gen.__name__)

        # Expand and Pause
        net.expand(policy='RandomPolicy')
        input()

        # Again, evaluate
        out = net(x)
        print(out.size())
        
        # Draw the network
        for G in Gs:
            draw_graph(G)
        draw_network(net, x, label=gen.__name__)

if __name__ == '__main__':
    test()
