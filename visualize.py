from io import BytesIO

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchviz import make_dot


def draw_graph(G, view=True, label=None):
    '''Draw NetworkX graph using pydot and graphviz

    Arguments:
        G (Graph or DiGraph): networkx graph to visualize
        view (bool): show graph
        label (str): graph is saved by specifying its name
    '''
    fig = plt.figure()
    P = nx.nx_pydot.to_pydot(G)
    data = P.create_png(prog=['dot', '-Gsize=9,15\\!', '-Gdpi=100'])
    img = mpimg.imread(BytesIO(data))
    plot = plt.imshow(img, aspect='equal')
    plt.axis('off')
    if view:
        plt.show(block=False)
    if label is not None:
        print(label)
        input()
        fig.savefig(label + '.png')
        plt.close(fig)


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
