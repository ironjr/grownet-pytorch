from io import BytesIO

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchviz import make_dot


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
