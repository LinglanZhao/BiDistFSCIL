import os
import tqdm
import math
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter 

from dataloader.data_utils import *
from dataloader.samplers import *
from methods.cosine_classifier import CosClassifier
from utils.utils import * 
from utils.fsl_inc import * 
from sync_batchnorm import convert_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # experiments arguments
    parser.add_argument('--dataroot', type=str, default='PATH/CEC-CVPR2021/data/')
    parser.add_argument('--dataset', type=str, default='mini-imagenet')
    parser.add_argument('--method', type=str, default='imprint')
    parser.add_argument('--base_mode', type=str, default='avg_cos')
    parser.add_argument('--norm_first', action='store_true')
    parser.add_argument('--exp_dir', type=str, default='experiment')
    # training arguments
    parser.add_argument('--epoch', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--batch_size_new', type=int, default=0)
    parser.add_argument('--batch_size_test', type=int, default=100)
    parser.add_argument('--init_lr', type=float, default=-1)
    parser.add_argument('--schedule', type=str, default='Milestone', choices=['Step', 'Milestone'])
    parser.add_argument('--milestones', nargs='+', type=int, default=-1)
    parser.add_argument('--step', type=int, default=40)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--val_start', type=int, default=50)
    parser.add_argument('--val_interval', type=int, default=5)
    parser.add_argument('--change_val_interval', type=int, default=70)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--report_binary', action='store_true')
    args = parser.parse_args()
    args = set_up_datasets(args)
    args.norm_first = True

    if args.init_lr == -1:
        if args.dataset == 'cifar100':
            args.init_lr = 0.1
        elif args.dataset == 'mini_imagenet':
            args.init_lr = 0.1
        elif args.dataset == 'cub200':
            args.init_lr = 0.01
        else:
            Exception('Undefined dataset name!')

    if args.milestones == -1:
        if args.dataset == 'cifar100':
            args.milestones = [120, 160]
        elif args.dataset == 'mini_imagenet':
            args.milestones = [120, 160]
        elif args.dataset == 'cub200':
            args.milestones = [50, 70, 90]
        else:
            Exception('Undefined dataset name!')

    args.checkpoint_dir = '%s/%s' %(args.exp_dir, args.dataset)
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    logging.basicConfig(filename=os.path.join(args.checkpoint_dir, 'train.log'), level=logging.INFO)
    logging.info(args)

    print(args)

    # init model
    model = CosClassifier(args, phase='pre_train')
    model.cuda()

    # init optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.init_lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    if args.schedule == 'Step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)
    elif args.schedule == 'Milestone':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    loss_fn = nn.CrossEntropyLoss()

    # dataset in pre-training phase
    trainset, trainloader, testloader = model.get_dataloader(0)
    print(len(trainset))
    
    # training
    best_test_acc_base    = 0
    best_test_epoch_base  = 0 
    for epoch in range(args.epoch):
        if epoch >= args.change_val_interval:
            args.val_interval = 1
        
        # torch.cuda.empty_cache()
        model.train()
        if args.schedule != 'Step' and args.schedule != 'Milestone':
            adjust_learning_rate(optimizer, epoch, init_lr=args.init_lr, n_epoch=args.epoch)
        
        tqdm_gen = tqdm.tqdm(trainloader)
        loss_avg = 0
        for i, X in enumerate(tqdm_gen):
            data, label = X
            data = data.cuda()
            label = label.cuda()
            pred = model(flag='base_forward', input=data)
            loss = loss_fn(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tqdm_gen.set_description('e:%d loss = %.4f' % (epoch, loss.item()))
            loss_avg += loss.item()
        if args.schedule == 'Step' or args.schedule == 'Milestone':
            lr_scheduler.step()
        
        out_str = '======epoch: %d avg loss: %.6f======'%(epoch, loss_avg/len(trainloader))
        print(out_str)
        logging.info(out_str)
        
        # testing
        model.eval()
        if (epoch == 0) or (epoch > args.val_start and (epoch+1) % args.val_interval == 0):
            acc_list = model.test_inc_loop(epoch=epoch)
            test_acc_base = acc_list[0]
            if test_acc_base > best_test_acc_base:
                best_test_acc_base = test_acc_base
                best_test_epoch_base = epoch
                outfile = os.path.join(args.checkpoint_dir, 'best_model_%s.tar'%(args.base_mode))
                torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
    
    out_str = '==========Epoch: %d Best Base Test acc = %.2f%%==========='%(best_test_epoch_base, 100*best_test_acc_base)
    print(out_str)
    logging.info(out_str)