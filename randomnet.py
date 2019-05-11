import copy
import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import Node


class RandomNetwork(nn.Module):
    def __init__(self, in_planes, planes, G, nmap=None, downsample=True, drop_edge=0, monitor_flow=True):
        '''Random DAG network of nodes

        Arguments:
            in_planes (int): number of channels from the previous layer
            planes (int): number of channels each nodes have
            G (DiGraph): DAG from random graph generator
            nmap (dict): saved node id map for optimal traversal
            downsample (bool): overrides downsample setting of the top layer
            monitor_flow (bool): monitor dataflow through the graph
        '''
        super(RandomNetwork, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.G = G
        self.drop_edge = drop_edge
        self.monitor_flow = monitor_flow

        # Generate nodes based on the final graph
        self.pred = self.G.pred
        self.in_degree = self.G.in_degree
        out_degree = self.G.out_degree
        self.bottom_layer = []
        self.nodes = nn.ModuleList()
        self.nparams = 0 # Computational complexity measure
        for nodeid in G.nodes:
            # Top layer nodes
            in_degree = self.in_degree(nodeid)
            if in_degree == 0:
                node = Node(in_planes, planes, in_degree,
                        downsample=downsample)
                self.nparams += in_planes * (9 + planes)
            else:
                node = Node(planes, planes, in_degree)
                self.nparams += planes * (9 + planes)
            self.nparams += in_degree # w

            # Bottom layer nodes
            if out_degree(nodeid) == 0:
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
                for i, p in enumerate(self.pred[nodeid]):
                    ipred = self.nxorder.index(self.nmap_rev[p])
                    iout = self.live[order].index(ipred)
                    if i == drop_idx:
                        # Drop simply by not passing the value of predecessor
                        y.append(torch.zeros_like(outs[iout]))
                    else:
                        # Normal data flow
                        y.append(outs[iout])
                    if order is not len(self.nxorder) - 1 and \
                            ipred not in self.live[order + 1]:
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
            nextnodes = [nxorder.index(n) for n in succ[nmapid]]
            span = max(nextnodes) if len(nextnodes) != 0 else G.number_of_nodes()
            ispans.append(span)

        live = [None,] * len(nxorder) # list of nxorder indices stored in topological order
        for order in range(len(nxorder)):
            live[order] = [inode for inode, ispan in enumerate(ispans) \
                    if ispan >= order and inode < order]

        return nxorder, live

    def _shuffle_with_consistency(self):
        pass

    def begin_monitor(self):
        for n in self.nodes:
            n.begin_monitor()

    def stop_monitor(self):
        for n in self.nodes:
            n.stop_monitor()

    def add_node(self, edges):
        '''Add a node to the network with additional edges

        Added node get a new ID in sequential order. Edges are provided in
        order to maintain the network connected.

        Arguments:
            edges (list(tuple)): List of additional directed edges containing
                                 the added node
        '''
        # New node ID is always larger than the largest current node ID
        newid = max(self.G.nodes) + 1

        # Update the graph
        self.G.add_node(newid)
        for edge_from, edge_to in edges:
            # Check consistency of the edges and the node
            if edge_from == newid or edge_to == newid:
                self.G.add_edge(edge_from, edge_to)
            else:
                assert False, 'New edges must contain the new node'

            # Check if no cycles are created
            pass

        # Update the network
        # TODO
        pass

    def add_edge(self, nid_from, nid_to):
        '''Add a directed edge to the network

        Arguments:
            nid_from (int): ID of the node where the edge starts from
            nid_to (int): ID of the node where the edge ends
        '''
        # Update the graph
        self.G.add_edge(nid_from, nid_end)

        # Check consistency of the edges and the node
        # Check if no cycles are created
        pass

        # Update the network
        node_to = self.nodes[nid_to]
        pass

    def increase_depth(self, edge_start, edge_end):
        if not self.G.has_edge(edge_start, edge_end):
            return

        # NOTE: Increasing depth cannot insert node at the top
        # What about at the end?
        node_start = self.nodes[edge_start]
        node_end = self.nodes[edge_end]
        # TODO perhaps different fanin?
        node_new = Node(node_start.planes, node_start.planes, 1)

        # Update the graph
        print(len(self.G.nodes))
        newid = len(self.G.nodes)
        self.G.add_node(newid)
        self.G.add_edge(edge_start, newid)
        self.G.add_edge(newid, edge_end)
        print(self.G.nodes)
        print(self.G.edges)
        print(len(self.G.nodes))
    
    def increase_width(self, previd, currid, nextid):
        if not self.G.has_edge(previd, currid) or not self.G.has_edge(currid, nextid):
            return

        # NOTE: Two nodes are connected by at most one edge
        node_prev = self.nodes[previd]
        node_curr = self.nodes[currid]
        node_next = self.nodes[nextid]
        node_new = copy.deepcopy(node_currid)
        self.nodes.append(node_new)
        #  node_next.scale_input_edge(
        
        # Update the graph
        print(len(self.G.nodes))
        newid = len(self.G.nodes)
        self.G.add_node(newid)
        self.G.add_edge(previd, newid)
        self.G.add_edge(newid, nextid)
        print(self.G.nodes)
        print(self.G.edges)
        print(len(self.G.nodes))

    def delete_edge(self, index):
        # TODO
        pass


def test():
    from torchviz import make_dot
    from graph import GraphGenerator
    graphgen = GraphGenerator('WS', { 'K': 6, 'P': 0.25, })
    G = graphgen.generate(nnode=16)
    randnet = RandomNetwork(in_planes=3, planes=16, G=G, downsample=False)
    x = torch.randn(8, 3, 224, 224)

    # Evaluate the network
    out = randnet(x)
    print(out.size())
    #  dot = make_dot(out, params=dict(randnet.named_parameters()))
    #  dot.render(view=True)

    # Modify the network online
    #  randnet.increase_width(4)

    #  out = randnet(x)
    #  print(out.size())
    #  dot = make_dot(out, params=dict(randnet.named_parameters()))
    #  dot.render(view=True)


if __name__ == '__main__':
    test()
