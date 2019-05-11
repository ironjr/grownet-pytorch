from torchviz import make_dot
from graph import GraphGenerator

# TODO
def visualize_graph():
    graphgen = GraphGenerator('WS', { 'K': 6, 'P': 0.25, })
    G = graphgen.generate(nnode=16)
    randnet = RandomNetwork(in_planes=3, planes=16, G=G, downsample=False)
    x = torch.randn(8, 3, 224, 224)

    # Evaluate the network
    out = randnet(x)
    print(out.size())
    dot = make_dot(out, params=dict(randnet.named_parameters()))
    dot.render(view=True)
