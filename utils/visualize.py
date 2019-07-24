import os
from io import BytesIO

import torch

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchviz import make_dot


# Visualize graphs
def plot_topology(label, model, batch_size, feature_size=32, view=False, wrapped=True, save_dir='graph'):
    # Evaluate the network
    if wrapped:
        model.eval()
        model = model.module
    Gs, _ = model.get_graphs()

    # Draw the network
    x = torch.randn(batch_size, 3, feature_size, feature_size).cuda()
    for i, G in enumerate(Gs):
        draw_graph(G, view=view, label=(os.path.join(save_dir, label + '.' + str(i))))
    draw_network(model, x, view=view, label=(os.path.join(save_dir, label)))


def plot_infoflow(label, model, colormap, view=False, wrapped=True, save_dir='graph'):
    # Evaluate the network
    if wrapped:
        model.eval()
        model = model.module
    Gs, _ = model.get_graphs()

    colors = []
    for lname, layer in model.get_sublayers():
        # Get current weights
        weights = layer.get_edge_strength()
        # Re-normalize weights to fit in [0, 1]
        bias = min(weights.values()) if len(weights) > 1 else 0.0
        scale = max(weights.values()) - bias
        colors.append({ k: colormap.get_colors((v - bias) / scale) for k, v in weights.items() })

    for i, (G, color) in enumerate(zip(Gs, colors)):
        name = label + '.' + str(i) + '.' + 'infoflow'
        draw_graph(G, edge_colors=color, view=view, label=(os.path.join(save_dir, name)))


def draw_graph(G, edge_colors=None, view=True, label=None):
    '''Draw NetworkX graph using pydot and graphviz

    Arguments:
        G (Graph or DiGraph): networkx graph to visualize
        edge_colors (dict): dict of edge -> color in "#%2x%2x%2x"
        view (bool): show graph
        label (str): graph is saved by specifying its name
    '''
    P = nx.nx_pydot.to_pydot(G)
    edge_list = P.get_edge_list()

    # Apply colors
    if edge_colors is not None:
        for e in edge_list:
            s = int(e.get_source())
            d = int(e.get_destination())
            e.set_color(edge_colors[(s, d)])

    if view:
        fig = plt.figure(figsize=(8, 6), dpi=100)
        data = P.create_png(prog=['dot', '-Gsize=9,12\\!', '-Gdpi=400'])
        img = mpimg.imread(BytesIO(data))
        plot = plt.imshow(img, aspect='equal')
        plt.axis('off')
        plt.show(block=False)
        plt.close(fig)
    if label is not None:
        P.write_png(label + '.png')
        #  fig.savefig(label + '.png')
        #  plt.close(fig)


def draw_network(net, input, view=True, label='RandomNet'):
    '''Draw pytorch neural network using torchviz

    Needs evaluation of input data

    Arguments:
        net (nn.Module): neural network model to visualize
        input (Tensor): input tensor for net
        view (bool): show graph
        label (str, optional): name network plot is saved under
    '''
    out = net(input)
    dot = make_dot(out, params=dict(net.named_parameters()))
    dot.render(filename=label, format='png', view=view)
