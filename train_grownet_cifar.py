import argparse
import os
import time
from copy import deepcopy

from tqdm import tqdm
from ptflops import get_model_complexity_info

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from models.grownet import GrowNetTinyNormal, GrowNetTiny18, GrowNetTinyWide
from models.loss import CEWithLabelSmoothingLoss
from utils.scheduler import CosineAnnealingWithRestartsLR
from utils.util import *
from utils.visualize import *
from utils.logger import Logger
from utils.colormap import ColorMap


def main(args):
    iteration = 0

    # Tensorboard loggers
    log_root = './log'
    log_names = ['', 'train_loss', 'test_loss', 'model_size']
    log_dirs = list(map(lambda x: os.path.join(log_root, args.label, x), log_names))
    if not os.path.isdir(log_root):
        os.mkdir(log_root)
    for log_dir in log_dirs:
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
    _, train_logger, val_logger, size_logger = \
            list(map(lambda x: Logger(x), log_dirs))

    # Checkpoint save directory
    checkpoint_root = './checkpoint'
    if not os.path.isdir(checkpoint_root):
        os.mkdir(checkpoint_root)
    checkpoint_dir = os.path.join(checkpoint_root, args.label)
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # Graph save directory
    graph_root = './graph'
    if not os.path.isdir(graph_root):
        os.mkdir(graph_root)
    graph_dir = os.path.join(graph_root, args.label)
    if not os.path.isdir(graph_dir):
        os.mkdir(graph_dir)

    # Infoflow graph
    if True: # TODO add option
        cmap = ColorMap('colormap/gyr.txt')

    # Data transforms and loaders
    print('==> Preparing data ..')
    traintf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    valtf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = datasets.CIFAR10(
            root=args.data_root, train=True, download=True, transform=traintf)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valset = datasets.CIFAR10(
            root=args.data_root, train=False, download=True, transform=valtf)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model and optimizer
    start_iter = 0
    if args.resume:
        # Load from preexisting models
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.checkpoint)
        # Model from existing random graphs
        Gs, nmaps = checkpoint['graphs']
        cfg = {
            'depthrate': args.depthrate,
        }
        if args.model in ['tiny16', 'tiny']:
            model, _, _ = GrowNetTinyNormal(
                Gs=Gs,
                nmaps=nmaps,
                drop_edge=args.drop_edge,
                dropout=args.dropout,
                cfg=cfg
            )
        elif args.model in ['tiny18']:
            model, _, _ = GrowNetTiny18(
                Gs=Gs,
                nmaps=nmaps,
                drop_edge=args.drop_edge,
                dropout=args.dropout,
                cfg=cfg
            )
        elif args.model in ['tinywide']:
            model, _, _ = GrowNetTinyWide(
                Gs=Gs,
                nmaps=nmaps,
                drop_edge=args.drop_edge,
                dropout=args.dropout,
                cfg=cfg
            )
        else:
            raise NotImplementedError
        if not args.reset_model:
            model.load_state_dict(checkpoint['model'])
        
        optimizer = optim.SGD(
            group_weight(model),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        if not args.reset_model:
            optimizer.load_state_dict(checkpoint['optim'])
            for group in optimizer.param_groups:
                group['lr'] = args.lr
                group['momentum'] = args.momentum
            # Overwrite default hyperparameters for new run
            group_decay, _, group_in_weights = optimizer.param_groups
            group_decay['weight_decay'] = args.weight_decay

            args.start_epoch = checkpoint['epoch']
            start_iter = checkpoint['iteration']
            print("Last loss: %.3f" % (checkpoint['loss']))
            print("Training start from epoch %d iteration %d" % (args.start_epoch, start_iter))
        else:
            print("Model loaded with reset weights")
    else:
        # Start from scratch
        print('==> Generating new model..')

        cfg = {
            'depthrate': args.depthrate,
        }
        if args.model in ['tiny16', 'tiny']:
            model, _, _ = GrowNetTinyNormal(
                drop_edge=args.drop_edge,
                dropout=args.dropout,
                cfg=cfg
            )
        elif args.model in ['tiny18']:
            model, _, _ = GrowNetTiny18(
                drop_edge=args.drop_edge,
                dropout=args.dropout,
                cfg=cfg
            )
        elif args.model in ['tinywide']:
            model, _, _ = GrowNetTinyWide(
                drop_edge=args.drop_edge,
                dropout=args.dropout,
                cfg=cfg
            )
        else:
            raise NotImplementedError

        optimizer = optim.SGD(
            group_weight(model),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )

        print('Model generated with depth rate : ', cfg['depthrate'])

    # Use cosine annealing scheduler unless noted
    scheduler = None
    if not args.no_cosine_annealing:
        scheduler = CosineAnnealingWithRestartsLR(
            optimizer,
            T_max=args.t_max,
            T_mult=args.t_mult
        )

    # Criterion has no internal parameters
    if args.no_label_smoothing:
        criterion = F.cross_entropy
    else:
        criterion = CEWithLabelSmoothingLoss

    # Use CUDA and enable multi-GPU learning
    model = torch.nn.DataParallel(model,
            device_ids=range(torch.cuda.device_count()))
    model.cuda()
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    # Generate monitor policy
    monitor_policy = dict(param=args.monitor_param, stat=args.monitor_stat)

    # Run a single test
    if args.test_only:
        model.module.begin_monitor(monitor_policy)
        test(valloader, model, criterion, (args.start_epoch + 1) * len(trainloader),
                val_logger=val_logger)
        plot_infoflow(args.label, model, cmap, view=False)
        model.module.stop_monitor()

        # Complexity test
        flops, nparams = get_model_complexity_info(model.module.cpu(), (3, 32, 32), \
                as_strings=False, print_per_layer_stat=False)
        flops /= 1000000.0
        nparams /= 1000000.0

        print('Complexity result')
        if flops > 1000.0:
            flops /= 1000.0
            print('FLOPs   : {:.3f}B'.format(flops))
        else:
            print('FLOPs   : {:.3f}M'.format(flops))
        print('#Params : {:.3f}M'.format(nparams))
        return

    # Main run
    for epoch in range(args.start_epoch, args.start_epoch + args.num_epochs):
        if scheduler is not None:
            scheduler.step()
            if scheduler.save_flag:
                save('restart' + str(epoch - 1), model, optimizer, epoch)

        # Train
        train(trainloader, model, criterion, optimizer, epoch,
                train_logger=train_logger, start_iter=start_iter)

        # Begin monitoring weights
        model.module.begin_monitor(monitor_policy)

        # Validate
        avgloss = test(valloader, model, criterion, (epoch + 1) * len(trainloader),
                val_logger=val_logger)

        # End monitoring weights
        model.module.stop_monitor()

        # Save after test
        save('done' + str(epoch), model, optimizer, avgloss, epoch + 1)

        # Expand the network
        if (epoch + 1) % args.expand_period == 0 and \
                epoch is not (args.start_epoch + args.num_epochs - 1):
            # Save expansion criterion based on the information flow
            plot_infoflow(str(epoch), model, cmap, view=False,
                    save_dir=graph_dir)

            # Modify the model
            # TODO add options
            expand_model(model, optimizer, policy=args.expand_policy, options=None)

            # Save intermediate network topology
            plot_topology(str(epoch), model, args.batch_size, view=False,
                    save_dir=graph_dir)

        # Log the model size
        if size_logger is not None:
            flops, nparams = model.module.get_complexity(32)
            info = {
                'MFLOPs': flops / 1000000.0,
                'Mega params': nparams / 1000000.0,
            }
            for tag, value in info.items():
                size_logger.scalar_summary(tag, value, epoch + 1)


# Train
def train(trainloader, model, criterion, optimizer, epoch, start_iter=0,
        train_logger=None):
    print('\nEpoch: %d' % epoch)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    #  top5 = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
        # Data loading time
        data_time.update(time.time() - end)

        # Offset correction
        idx = batch_idx + start_iter

        # FORWARD
        input_vars = Variable(inputs.cuda())
        target_vars = Variable(targets.cuda())

        outputs = model(input_vars)
        loss = criterion(outputs, target_vars)#, eps=0.1)

        # Update log
        prec = accuracy(outputs.data, target_vars, topk=(1,))
        top1.update(prec[0], inputs.size(0))
        losses.update(loss.data, inputs.size(0))

        # BACKWARD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Batch evaluation time
        batch_time.update(time.time() - end)

        # Log current performance
        step = idx + len(trainloader) * epoch
        if train_logger is not None:
            info = {
                'loss': loss.data,
                'average loss': losses.avg,
                'top1 precision': np.asscalar(top1.val.cpu().numpy()),
                'top1 average precision': np.asscalar(top1.avg.cpu().numpy()),
            }
            for tag, value in info.items():
                train_logger.scalar_summary(tag, value, step + 1)

        tqdm.write('batch (%d/%d) | loss: %.3f | avg_loss: %.3f | Prec@1: %.3f %% (%.3f %%)'
                % (batch_idx, len(trainloader), losses.val, losses.avg,
                    np.asscalar(top1.val.cpu().numpy()),
                    np.asscalar(top1.avg.cpu().numpy())))

        # Update the base time
        end = time.time()

        # Finish when total iterations match the number of batches
        if start_iter != 0 and (idx + 1) % len(trainloader) == 0:
            break

# Test
def test(valloader, model, criterion, iteration, val_logger=None):
    print('\nTest')

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    #  top5 = AverageMeter()

    model.eval()

    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(valloader)):
            # FORWARD
            input_vars = Variable(inputs.cuda())
            target_vars = Variable(targets.cuda())

            outputs = model(input_vars)
            loss = criterion(outputs, target_vars)#, eps=0.1)

            # Batch evaluation time
            batch_time.update(time.time() - end)

            # Update log
            prec = accuracy(outputs.data, target_vars, topk=(1,))
            top1.update(prec[0], inputs.size(0))
            losses.update(loss.data, inputs.size(0))

            tqdm.write('batch (%d/%d) | loss: %.3f | avg_loss: %.3f | Prec@1: %.3f %% (%.3f %%)' 
                    % (batch_idx, len(valloader), losses.val, losses.avg,
                        np.asscalar(top1.val.cpu().numpy()),
                        np.asscalar(top1.avg.cpu().numpy())))

            # Update the base time
            end = time.time()

    # Log results
    if val_logger is not None:
        info = {
            'average loss': losses.avg,
            'top1 average precision': np.asscalar(top1.avg.cpu().numpy()),
        }
        for tag, value in info.items():
            val_logger.scalar_summary(tag, value, iteration)
    print('average validation loss: %.3f' % (losses.avg))
    print('average top1 precision : %.3f' % (np.asscalar(top1.avg.cpu().numpy())))

    return losses.avg


def expand_model(model, optimizer, policy='MaxEdgeStrengthPolicy', options=None):
    # Update both module and the optimizer
    with torch.no_grad():
        model = model.module

        # Cache input weights
        group_in_weights = optimizer.param_groups[2]
        id_in_weights = [id(p) for p in group_in_weights['params']]
        input_weight_bufs = {}
        for p in model.named_parameters():
            if p[0].endswith('.w'):
                input_weight_bufs[p[0]] = (id_in_weights.index(id(p[1])), p[1])

        # Grow network a single step
        expand_info = model.expand(policy=policy, options=options)

        # 1. Update optimizer with newly created nodes
        for new_node in expand_info['new_nodes']:
            for gi, g in enumerate(group_weight(new_node)):
                for p in g['params']:
                    optimizer.param_groups[gi]['params'].append(p)

        # 2. Update optimizer with changed input edge weights
        for pname, param in expand_info['changed_params']:
            # Changed param is (currently) input weights
            pid, old_param = input_weight_bufs[pname]
            old_state = optimizer.state.pop(old_param)

            # Update state of optimizer
            # Assume optimizer on CUDA
            # Specific state to optim.SGD
            mbuf = old_state['momentum_buffer']
            mbuf = torch.cat((mbuf, torch.zeros(1, 1).cuda()), 1)
            optimizer.state[param]['momentum_buffer'] = mbuf
            # Remove old parameter
            del old_param
            del old_state

            # Update param_groups of optimizer
            del group_in_weights['params'][pid]
            group_in_weights['params'].insert(pid, param)

        # Model back to work
        model = torch.nn.DataParallel(model,
                device_ids=range(torch.cuda.device_count()))
        model.cuda()


# Save checkpoints
def save(label, model, optimizer, loss=float('inf'), epoch=0, iteration=0):
    tqdm.write('==> Saving checkpoint')

    # Get current network topology
    Gs, nmaps = model.module.get_graphs()

    # Maintain optimizer.param_groups in order for consistency
    in_group_order = []
    for groups in optimizer.param_groups:
        orders = []
        for np in model.named_parameters():
            for pid, p in enumerate(groups['params']):
                if id(np[1]) == id(p):
                    orders.append(pid)
        in_group_order.append(orders)
    optim_state = deepcopy(optimizer.state_dict())
    param_groups = optim_state['param_groups']
    for orders, group in zip(in_group_order, param_groups):
        group['params'] = [group['params'][i] for i in orders]

    state = {
        'model': model.module.state_dict(),
        'graphs': model.module.get_graphs(),
        'optim': optim_state,
        'loss': loss,
        'epoch': epoch,
        'iteration': iteration,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + args.label + '/ckpt_' + label + '.pth')
    tqdm.write('==> Save done!')


if __name__ == '__main__':
    assert torch.cuda.is_available(), 'CUDA is required!'

    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument('--label', default='default', type=str,
            help='labels checkpoints and logs saved under')
    parser.add_argument('--model', default='tinywide', type=str,
            choices=('tiny16', 'tiny', 'tiny18', 'tinywide',),
            help='model to run')
    parser.add_argument('--depthrate', default=0.2, type=float,
            help='increase depth probability')
    parser.add_argument('--drop-edge', default=0, type=float,
            help='drop rate of edges, does not apply to regular regime')
    parser.add_argument('--dropout', default=0, type=float,
            help='dropout rate before the fc layer')
    parser.add_argument('--monitor-param', default='max', type=str,
            choices=('l2norm', 'max', 'wabs', 'wpos',),
            help='type of monitored parameter in each of the nodes')
    parser.add_argument('--monitor-stat', default='cma', type=str,
            choices=('val', 'ma', 'cma',),
            help='stat of monitored parameter in each of the nodes')
    parser.add_argument('--expand-period', default=1, type=int,
            help='period of network expansion')
    parser.add_argument('--expand-policy', default='MaxEdgeStrengthPolicy', type=str,
            choices=('MaxEdgeStrengthPolicy', 'RandomPolicy',),
            help='policy of network expansion')
    parser.add_argument('--no-label-smoothing', action='store_true',
            help='use vanilla cross entropy loss instead of label smoothing')
    parser.add_argument('--num-workers', default=2, type=int,
            help='number of workers in dataloader')
    parser.add_argument('--data-root', default='../common/datasets/CIFAR-10', type=str,
            help='CIFAR dataset root')
    parser.add_argument('--no-cosine-annealing', action='store_true',
            help='use uniform scheduling instead of cosine annealing')
    parser.add_argument('--t-max', default=10, type=int,
            help='restart period of cosine annealing')
    parser.add_argument('--t-mult', default=1.0, type=float,
            help='factor of increment of restart period each restart')
    parser.add_argument('--lr', default=1e-1, type=float,
            help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
            help='SGD momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
            help='weight decay for SGD (tiny: 1e-4, small: 5e-5, large: 1e-5)')
    parser.add_argument('--batch-size', default=128, type=int,
            help='size of a minibatch')
    parser.add_argument('--start-epoch', default=0, type=int,
            help='epoch index to start log')
    parser.add_argument('--num-epochs', default=1, type=int,
            help='number of epochs to run')
    parser.add_argument('--resume', '-r', action='store_true',
            help='resume from checkpoint')
    parser.add_argument('--checkpoint', default='./checkpoint/ckpt.pth', type=str,
            help='path to the checkpoint to load')
    parser.add_argument('--test-only', action='store_true',
            help='run test sequence only once')
    parser.add_argument('--reset-model', action='store_true',
            help='reset all weights in the loaded model')
    args = parser.parse_args()

    # Run main routine
    main(args)
