import torch
import torch.nn as nn

from tqdm import tqdm
from ptflops import get_model_complexity_info

import layer


class AverageMeter(object):
    '''Computes and stores the average and current value
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    '''Computes the precision@k for the specified values of k
    '''
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

        # Custom modules
        elif isinstance(m, layer.Node):
            group_decay.append(m.w)

    #  assert len(list(module.parameters())) == len(group_decay) + \
    #      len(group_no_decay)
    groups = [dict(params=group_decay),
        dict(params=group_no_decay, weight_decay=.0)]

    return groups


def test_complexity(network_list, graph_types, graph_params, input_size=224, num_samples=100):
    '''Test computational complexity of the given networks
    '''
    # Handle single test samples
    if not isinstance(network_list, (list, tuple)):
        network_list = [network_list,]
    num_networks = len(network_list)
    if isinstance(graph_types, str):
        graph_types = [graph_types,] * num_networks
    if isinstance(graph_params, dict):
        graph_params = [graph_params,] * num_networks
    if isinstance(input_size, int):
        input_size = (input_size, input_size)

    # Evaluate for all the given networks
    test_names = []
    flops_tlist = []
    nparams_tlist = []
    for gen, gtype, gparam in zip(network_list, graph_types, graph_params):
        # Make name of the test
        gparam_str = ''
        for k in sorted(gparam):
            gparam_str += '{}={},'.format(k, gparam[k])
        test_name = '{}[\'{}\'({})]'.format(gen.__name__, gtype, gparam_str)
        test_names.append(test_name)
        print(test_name)

        # Evaluate
        flops_list = []
        nparams_list = []
        for i in tqdm(range(num_samples)):
            tqdm.write('Evaluate {} ({}/{})'.format(gen.__name__, i + 1, num_samples))
            net, _, _ = gen(model=gtype, params=gparam, seeds=None)
            #  flops, nparams = net.get_complexity(input_size)
            flops, nparams = get_model_complexity_info(net, input_size, \
                    as_strings=False, print_per_layer_stat=False)
            flops_list.append(flops)
            nparams_list.append(nparams)

        # Data are presented in millions (default)
        flops_tlist.append(torch.Tensor(flops_list) / 1000000.0)
        nparams_tlist.append(torch.Tensor(nparams_list) / 1000000.0)

    # Print out statistics
    for name, flops_tensor, nparams_tensor in zip(test_names, flops_tlist, nparams_tlist):
        print('Result of ', name)
        if torch.mean(flops_tensor) > 1000.0:
            flops_tensor /= 1000.0
            print('FLOPs   : {:.3f}B ({:.3f}B)'.format(torch.mean(flops_tensor), torch.std(flops_tensor)))
        else:
            print('FLOPs   : {:.3f}M ({:.3f}M)'.format(torch.mean(flops_tensor), torch.std(flops_tensor)))
        print('#Params : {:.3f}M ({:.3f}M)'.format(torch.mean(nparams_tensor), torch.std(nparams_tensor)))


