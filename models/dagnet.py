import copy
import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import Node


class DAGNet(nn.Module):
    def __init__(self, in_planes, planes, G, nmap=None, downsample=True, depthwise=False, drop_edge=0, monitor_flow=False, device='cuda'):
        '''Random DAG network of nodes

        Arguments:
            in_planes (int): number of channels from the previous layer
            planes (int): number of channels each nodes have
            G (DiGraph): DAG from random graph generator
            nmap (dict): saved node id map for optimal traversal
            downsample (bool): overrides downsample setting of the top layer
            depthwise (bool): whether to use depthwise separable convolution
            monitor_flow (bool): monitor dataflow through the graph
            device (str): default device
        '''
        super(DAGNet, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.G = G
        self.downsample = downsample
        self.depthwise = depthwise
        self.drop_edge = drop_edge
        self.monitor_flow = monitor_flow
        self.device = device

        # Generate nodes based on the final graph
        self.pred = self.G.pred
        self.in_degree = self.G.in_degree
        self.out_degree = self.G.out_degree
        self.bottom_layer = []
        self.nodes = nn.ModuleList()
        self.nparams = 0 # Computational complexity measure
        for nodeid in G.nodes:
            # Top layer nodes
            in_degree = self.in_degree(nodeid)
            if in_degree == 0:
                node = Node(in_planes, planes, in_degree,
                        downsample=downsample, depthwise=depthwise)
                if depthwise:
                    self.nparams += in_planes * (9 + planes)
                else:
                    self.nparams += in_planes * (9 * planes)
            else:
                node = Node(planes, planes, in_degree, depthwise=depthwise)
                if depthwise:
                    self.nparams += planes * (9 + planes)
                else:
                    self.nparams += planes * (9 * planes)
            self.nparams += in_degree # w

            # Bottom layer nodes
            if self.out_degree(nodeid) == 0:
                self.bottom_layer.append(nodeid)

            # Build ReLU-conv-BN triplet node
            self.nodes.append(node)

        if nmap is not None:
            self.nmap = nmap
            self.nxorder, self.live = self._get_livevars(G, nmap)
        else:
            self.nmap, self.nxorder, self.live = self._optimize_graph(G)
        self.nmap_rev = { v: k for k, v in self.nmap.items() }

    def forward(self, x):
        '''

        TODO do parallel processing of uncorrelated nodes (using compiler techniques)

        Arguments:
            x: A Tensor with size (N, C, H, W)

        Returns:
            A Tensor with size (N, C, H, W)
        '''
        # Traversal in the sorted graph
        outs = []
        for order, nmapid in enumerate(self.nxorder):
            nodeid = self.nmap[nmapid]
            node = self.nodes[nodeid]
            to_delete = []
            # Top layer nodes receive data from the upper layer
            if self.in_degree(nodeid) == 0:
                out = node(x.unsqueeze(-1)) # (N,Cin,H,W,F=1)
            else:
                y = []

                # Apply random edge drop if drop_edge is nonzero
                # Randomly remove one edge with some probability
                # if input degree > 1 (following the paper)
                drop_idx = -1
                if self.training and self.drop_edge is not 0:
                    # Whether to drop an input edge or not
                    if torch.bernoulli(torch.Tensor((self.drop_edge,))) == 1:
                        # Get random index out of predecessors
                        drop_idx = torch.randint(len(self.pred[nodeid]), (1,))

                # Aggregate input values to each node
                # Traversal over predecessors are ordered in node ID
                for i, p in enumerate(self.pred[nodeid]):
                    ipred = self.nxorder.index(self.nmap_rev[p])
                    iout = self.live[order].index(ipred)
                    if i == drop_idx:
                        # Drop simply by not passing the value of predecessor
                        y.append(torch.zeros_like(outs[iout]))
                    else:
                        # Normal data flow
                        y.append(outs[iout])
                    if order is len(self.nxorder) - 1:
                        if p not in self.bottom_layer:
                            to_delete.append(iout)
                    else:
                        if ipred not in self.live[order + 1]:
                            to_delete.append(iout)
                y = torch.stack(y) # (F,N,Cin,H,W)
                y = y.permute(1, 2, 3, 4, 0) # (N,Cin,H,W,F)
                out = node(y)

            # Make output layer compact by deleting values not to be used
            if len(to_delete) is not 0:
                # Delete element in backwards in order to maintain consistency
                to_delete.sort(reverse=True)
                for i in to_delete:
                    del outs[i]
            outs.append(out)

        # Aggregation on the output node
        out = torch.stack(outs) # (F,N,Cin,H,W)
        out = torch.mean(out, 0) # (N,Cin,H,W)
        return out

    def _optimize_graph(self, G):
        '''Optimize graph traversal order

        Since initialization of the graphs takes minor proportion on compu-
        tation time, we try multiple random permutation and sort to find
        optimal configuration for minimum memory consumption. Empirically,
        the number of tries are set to 20 which provides suboptimal result on
        network with nodes <= 32.

        Arguments:
            G (DiGraph): graph of topology of the random network

        Returns:
            mapopt (dict): node reassignment from the given graph to optimized
                           one
            nxorderopt (list): node traversal order
            liveopt (list): list of list containing live intermediate outcome
                            at each iteration
        '''
        num_reorder = 20
        min_lives = len(G.nodes)
        mapping = { i: i for i in range(len(G.nodes)) }
        for i in range(num_reorder):
            nxorder, live = self._get_livevars(G, mapping)

            # Maximum #live-vars
            nlives = max([len(nodes) for nodes in live])
            if nlives < min_lives:
                min_lives = nlives
                mapopt = mapping
                nxorderopt = nxorder
                liveopt = live

            # Reorder graph
            if i is not num_reorder - 1:
                new_order = np.random.permutation(len(G.nodes))
                mapping = { new_order[i]: i for i in range(len(G.nodes)) }
        return mapopt, nxorderopt, liveopt

    def _get_livevars(self, G, nmap):
        '''Get node traversal order for optimal memory usage

        Arguments:
            G (DiGraph): graph of topology of the random network
            nmap (dict): node id map for optimal traversal

        Returns:
            nxorder (list): node traversal order, stored in nmap-ed order
            live (list): list of list containing live intermediate outcome at
                         each iteration, stores nxorder indices
        '''
        # TODO maybe this is highly redundant
        nmap_rev = { v: k for k, v in nmap.items() }

        # Nodes are sorted in topological order (edge start nodes fisrt)
        G = nx.relabel_nodes(G, nmap_rev)
        nxorder = [n for n in nx.lexicographical_topological_sort(G)]

        # Count live variable to reduce the memory usage
        ispans = [] # indices of nxorder
        succ = G.succ
        for nmapid in nxorder:
            if nmap[nmapid] in self.bottom_layer:
                span = G.number_of_nodes()
            else:
                nextnodes = [nxorder.index(n) for n in succ[nmapid]]
                span = max(nextnodes) if len(nextnodes) != 0 else G.number_of_nodes()
            ispans.append(span)

        live = [None,] * len(nxorder) # list of nxorder indices stored in topological order
        for order in range(len(nxorder)):
            live[order] = [inode for inode, ispan in enumerate(ispans) \
                    if ispan >= order and inode < order]
        return nxorder, live

    def begin_monitor(self, policy=dict(param='max', stat='cma')):
        for n in self.nodes:
            n.begin_monitor(policy)
        self.monitor_flow = True

    def stop_monitor(self):
        for n in self.nodes:
            n.stop_monitor()
        self.monitor_flow = False

    def get_edge_strength(self):
        weights = {}
        for ni, n in enumerate(self.nodes):
            weights.update({
                (p, ni): n.strengths[pi]
                for pi, p in enumerate(self.pred[ni])
            })
        return weights

    def add_node(self, edges, template=None):
        '''Add a node to the network with additional edges

        Added node get a new ID in sequential order. Edges are provided in
        order to maintain the network connected.

        TODO Do explicit type and dimension check of template

        Arguments:
            edges (list(tuple)): List of additional directed edges containing
                                 the added node
            template (Node): Template of the duplicative initialization,
                             randomly initialize if the value is None
        '''
        # New node ID is always larger than the largest current node ID
        newid = max(self.G.nodes) + 1

        # Update the graph
        self.G.add_node(newid)
        in_degree = 0
        for edge_from, edge_to in edges:
            # Check consistency of the edges and the node
            if edge_from == newid:
                if self.in_degree(edge_to) == 0:
                    raise Exception('No new nodes can be created over the top layer')
                self.G.add_edge(edge_from, edge_to)
                self.nodes[edge_to].add_input_edge(self.nodes[edge_to].fin)
            elif edge_to == newid:
                self.G.add_edge(edge_from, edge_to)
                in_degree += 1
            else:
                raise Exception('New edges must contain the new node')

            # Check if no cycles are created
            # TODO
            pass

        # Update fields related to the graph topology
        self.pred = self.G.pred
        self.in_degree = self.G.in_degree
        self.out_degree = self.G.out_degree

        # Bottom layer nodes
        # NOTE: Bottom layer keeps expanding
        if self.out_degree(newid) == 0:
            self.bottom_layer.append(newid)
        
        # Update the network
        if in_degree == 0:
            in_planes = self.in_planes
        else:
            in_planes = self.planes
        if self.depthwise:
            self.nparams += in_planes * (9 + self.planes) # depthwise conv
        else:
            self.nparams += in_planes * (9 * self.planes) # convolution
        self.nparams += in_degree # w

        if template is None:
            # Create new node
            if in_degree == 0:
                node = Node(in_planes, self.planes, in_degree,
                        downsample=self.downsample, depthwise=self.depthwise)
            else:
                node = Node(in_planes, self.planes, in_degree,
                        depthwise=self.depthwise)
            node.to(self.device)

            # Initialization
            for m in node.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out',
                            nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            # Copy node from the template
            node = copy.deepcopy(template)

        # Build ReLU-conv-BN triplet node
        self.nodes.append(node)

        # Update mapping
        self.nmap, self.nxorder, self.live = self._optimize_graph(self.G)
        self.nmap_rev = { v: k for k, v in self.nmap.items() }

        return node

    def add_edge(self, edge_from, edge_to):
        '''Add a directed edge to the network

        Arguments:
            edge_from (int): ID of the node where the edge starts from
            edge_to (int): ID of the node where the edge ends
        '''
        # Check consistency of the edges and the node
        if (edge_from, edge_to) not in self.G.nodes:
            raise Exception('Nodes in the edge should be in the graph')

        # Update the graph
        self.G.add_edge(edge_from, edge_end)

        # Check if no cycles are created
        # TODO
        pass

        # Update fields related to the graph topology
        self.pred = self.G.pred
        self.in_degree = self.G.in_degree
        self.out_degree = self.G.out_degree

        # Update the network
        self.nparams += 1 # w
        self.nodes[edge_to].add_input_edge(self.nodes[edge_to].fin)

        # Update mapping
        self.nmap, self.nxorder, self.live = self._optimize_graph(self.G)
        self.nmap_rev = { v: k for k, v in self.nmap.items() }

    def increase_depth(self, edge_from, edge_to):
        '''Increase depth of an edge

        There are three possible ways to increase depth of the network:
        1. (top) Node cannot be added over a top node.
        2. (bottom) Aggregation node is the edge end node.
        3. (middle) New node is inserted as follows:

                    (from)        (from)
                      |             |  \  
                      |    --->     | (new)
                      |             |  /
                     (to)          (to)

        TODO perhaps new node with fanin larger than 1?

        Arguments:
            edge_from (int): ID of a node where edge begins
            edge_to (int or None): ID of a node where edge ends
        '''
        newid = max(self.G.nodes) + 1
        if edge_to is None:
            # Bottom node
            edges = [(edge_from, newid),]
        elif self.G.has_edge(edge_from, edge_to):
            edges = [(edge_from, newid), (newid, edge_to),]
        else:
            return
        return self.add_node(edges)
    
    def increase_width(self, edge_from, edge_to):
        '''Increase width of the network by its edge

        There are three possible ways to increase width of the network:
        1. (top) Node can be add as middle case, in_planes should be matched.
        2. (bottom) Node cannot be added at the bottom.
        3. (middle) New node is inserted as follows:

           (pred2) (pred1) (pred3)       (pred2) (pred1) (pred3)
                  \   |    /                   \   |    /      
                    (from)       --->         (from| new)
                    1 |                       0.5 \ / 0.5
                     (to)                         (to)

        Arguments:
            edge_from (int): ID of a node where edge begins
            edge_to (int): ID of a node where edge ends
        '''
        newid = max(self.G.nodes) + 1
        edges = []
        if self.G.has_edge(edge_from, edge_to):
            for p in self.pred[edge_from]:
                edges.append((p, newid))
            edges.append((newid, edge_to))
        else:
            return
        node = self.add_node(edges, template=self.nodes[edge_from])

        # Scale the weight of duplicated edge to half
        preds = [k for k in self.pred[edge_to]]
        node_to = self.nodes[edge_to]
        node_to.scale_input_edge(preds.index(edge_from), 0.5)
        node_to.scale_input_edge(preds.index(newid), 0.5)
        return node

    def delete_edge(self, index):
        # TODO
        pass


def test():
    from graph import GraphGenerator
    from visualize import draw_graph, draw_network

    # Generate random graph
    graphgen = GraphGenerator('WS', { 'K': 4, 'P': 0.75, })
    G = graphgen.generate(nnode=16)
    randnet = DAGNet(in_planes=3, planes=16, G=G, downsample=False)
    x = torch.randn(8, 3, 32, 32) # Sample image

    # Monitor
    randnet.begin_monitor()
    out = randnet(x)
    edge_weights = randnet.get_edge_strength()
    randnet.stop_monitor()

    for i, e in enumerate(edge_weights):
        print(i, edge_weights[e] * 100000)

    # Draw the network
    draw_graph(randnet.G)
    #  draw_network(randnet, x, label='Vanilla')

    # Test increase_width
    if False:
        # Modify the network online
        print(randnet.G.edges)
        index = 7
        for i, e in enumerate(randnet.G.edges):
            if index == i:
                edge_from, edge_to = e
                break
        print(e)
        randnet.increase_width(edge_from, edge_to)

        # Draw the network
        draw_graph(randnet.G)
        draw_network(randnet, x, label='IncreaseWidth')

    # Test increase_depth
    if False:
        # Modify the network online
        print(randnet.G.edges)
        index = len(randnet.G.edges) - 1
        for i, e in enumerate(randnet.G.edges):
            if index == i:
                edge_from, edge_to = e
                break
        print(e)
        randnet.increase_depth(edge_from, edge_to)

        for nid in G.nodes:
            if randnet.G.out_degree(nid) == 0:
                break
        print(nid)
        randnet.increase_depth(nid, None)

        # Draw the network
        draw_graph(randnet.G)
        draw_network(randnet, x, label='IncreaseDepth')

    input()


if __name__ == '__main__':
    test()
