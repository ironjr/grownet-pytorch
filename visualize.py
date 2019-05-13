from io import BytesIO

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchviz import make_dot


def draw_graph(G):
    '''Draw NetworkX graph using pydot and graphviz

    Arguments:
        G (Graph or DiGraph): networkx graph to visualize
    '''
    plt.figure()
    P = nx.nx_pydot.to_pydot(G)
    data = P.create_png(prog=['dot', '-Gsize=9,15\\!', '-Gdpi=100'])
    img = mpimg.imread(BytesIO(data))
    plot = plt.imshow(img, aspect='equal')
    plt.axis('off')
    plt.show(block=False)


def draw_network(net, input, label='RandomNet'):
    '''Draw pytorch neural network using torchviz

    Needs evaluation of input data

    Arguments:
        net (nn.Module): neural network model to visualize
        input (Tensor): input tensor for net
        label (str, optional): name network plot is saved under
    '''
    out = net(input)
    dot = make_dot(out, params=dict(net.named_parameters()))
    dot.render(filename=label, view=True)
