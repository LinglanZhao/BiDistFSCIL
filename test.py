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
    parser.add_argument('--no_data_init', action='store_true')
    parser.add_argument('--norm_first', action='store_true')
    parser.add_argument('--exp_dir', type=str, default='experiment')
    parser.add_argument('--load_tag', type=str, default='avg_cos')
    parser.add_argument('--log_tag', type=str, default='')
    # training arguments
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--batch_size_new', type=int, default=0)
    parser.add_argument('--batch_size_test', type=int, default=100)
    parser.add_argument('--batch_size_exemplar', type=int, default=50)
    parser.add_argument('--batch_size_current', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=8)
    # finetuning arguments
    parser.add_argument('--needs_finetune', action='store_true')
    parser.add_argument('--using_exemplars', action='store_true', help='save 1 exemplars per class')
    parser.add_argument('--save_all_data_base', action='store_true', help='save 5 instead of 1 exemplars per base class')
    parser.add_argument('--save_all_data_novel', action='store_true', help='save 5 instead of 1 exemplars per novel class')
    parser.add_argument('--imprint_ft_weight', action='store_true', help='imprint the ft_weight using traning data and exemplars')
    parser.add_argument('--bn_eval', action='store_true', help='set the mean and variance of BN fixed')
    parser.add_argument('--part_frozen', action='store_true', help='only the last residual block will be updated')
    parser.add_argument('--ft_optimizer', type=str, default='SGD', help='finetune optimizer')
    parser.add_argument('--ft_lr', type=float, default=0.01, help='finetune lr')
    parser.add_argument('--ft_factor', type=float, default=0.1, help='backbone lr factor')
    parser.add_argument('--ft_iters', type=int, default=100, help='number of finetune iterations')
    parser.add_argument('--ft_n_repeat', type=int, default=1, help='construct a batch using multiple get_item()')
    parser.add_argument('--ft_T', type=float, default=4.0, help='Knowledge Distillation temperature')
    parser.add_argument('--ft_teacher', type=str, default='fixed', help='choose which model as teacher: fixed, prev, ema')
    parser.add_argument('--ft_momentum', type=float, default=0.9, help='EMA momentum factor on model parameters (--ft_ema_alpha)')
    parser.add_argument('--ft_momentum_type', type=int, default=1, help='EMA momentum type: ema0 or ema1')
    parser.add_argument('--ft_KD_all', action='store_true', help='distill all the logits or only previous logits')
    parser.add_argument('--ft_reinit', action='store_true', help='re-initialize student model at the beginning of each session')
    parser.add_argument('--w_cls', type=float, default=1, help='weight of softmax cross-entropy classification loss on novel classes')
    parser.add_argument('--w_e', type=float, default=5, help='weight of extra loss functions (e.g. CE on seen classes)')
    parser.add_argument('--w_d', type=float, default=100, help='weight of Knowledge Distillation loss')
    parser.add_argument('--w_l', type=float, default=0, help='weight of L1/L2 normalization loss')
    parser.add_argument('--w_l_order', type=int, default=1, help='L1/L2 normalization loss')
    parser.add_argument('--margin', type=float, default=0, help='using margin based softmax loss function')
    parser.add_argument('--triplet', action='store_true', help='triplet loss')
    parser.add_argument('--triplet_gap', type=float, default=0, help='triplet loss gap')
    parser.add_argument('--KD_rectified', action='store_true', help='rectify the logits of the teacher model')
    parser.add_argument('--KD_rectified_factor', type=float, default=0.8, help='KD rectified factor')
    parser.add_argument('--weighted_kd', action='store_true', help='using different distillation weights for different classes')
    parser.add_argument('--w_kd_novel', type=float, default=1.0, help='distillation weights for novel classes')
    parser.add_argument('--vis_exemplars', action='store_true', help='save and visualize exemplars after finetuning')
    parser.add_argument('--vis_exemplars_nrow', type=int, default=10, help='number of exemplars displayed in each row of the grid')
    parser.add_argument('--vis_logits', action='store_true', help='save logits after finetuning (--save_logits)')
    parser.add_argument('--logits_tag', type=str, default='saved_logits')
    # EMA logits arguments
    parser.add_argument('--EMA_logits', action='store_true', help='using exponential moving average in teacher model logits (--EMA_teacher)')
    parser.add_argument('--EMA_prob', action='store_true', help='EMA softmaxed distributions instead of logits')
    parser.add_argument('--EMA_type', type=str, default='learnable_mlp_b', help='linear, window, linear_t, learnable_mpl_b')
    parser.add_argument('--EMA_w_size', type=int, default=3, help='EMA_type=window, EMA window size')
    parser.add_argument('--EMA_scalar', type=float, default=0, help='EMA_type=learnable or learnable_v/s, EMA learnable parameter initialization')
    parser.add_argument('--EMA_scalar_lr', type=float, default=0.01, help='EMA_type=learnable or learnable_v/s, EMA learnable parameter lr')
    parser.add_argument('--EMA_factor_b_1', type=float, default=1.0, help='EMA_type=linear/window/linear_t, factor for base logits begins')
    parser.add_argument('--EMA_factor_b_2', type=float, default=1.0, help='EMA_type=linear/window/linear_t, factor for base logits ends')
    parser.add_argument('--EMA_factor_n_1', type=float, default=0.5, help='EMA_type=linear/window/linear_t, factor for novel logits begins')
    parser.add_argument('--EMA_factor_n_2', type=float, default=0.5, help='EMA_type=linear/window/linear_t, factor for novel logits ends')
    parser.add_argument('--EMA_top_k', type=int, default=1, help='EMA_type=learnable_s, topk for computing base similarity')
    parser.add_argument('--EMA_FC_dim',type=int, default=64, help='EMA_type=learnable_mpl_v/c/b, hidden dim')
    parser.add_argument('--EMA_FC_lr', type=float, default=0.01, help='EMA_type=learnable_s/mpl_v/c, lr')
    parser.add_argument('--EMA_FC_K',  type=float, default=1, help='EMA_type=learnable_s, K * x + b')
    parser.add_argument('--EMA_FC_b',  type=float, default=1, help='EMA_type=learnable_s, K * x + b')
    parser.add_argument('--EMA_s_type', type=int, default=0)
    parser.add_argument('--EMA_reinit', action='store_true', help='EMA_type=learnable_s, reinit K and b')
    # bilateral-branch arguments
    parser.add_argument('--bilateral', action='store_true', help='bilateral-branch during testing')
    parser.add_argument('--report_binary', action='store_true')
    parser.add_argument('--main_branch', type=str, default='current', help='current, ema')
    parser.add_argument('--second_branch', type=str, default='fixed', help='fixed, ema')
    parser.add_argument('--merge_strategy', type=str, default='attn', help='how to ensmeble two branches')
    parser.add_argument('--branch_selector', type=str, default='logits_current')
    parser.add_argument('--masking_novel', action='store_true') 
    parser.add_argument('--branch_weights', type=float, default=0.5)
    parser.add_argument('--BC_hidden_dim', type=int, default=64)
    parser.add_argument('--BC_lr', type=float, default=0.01)
    parser.add_argument('--BC_flatten', type=str, default='org')
    parser.add_argument('--BC_detach', action='store_true')
    parser.add_argument('--BC_detach_f', action='store_true')
    parser.add_argument('--BC_binary_factor', type=float, default=1.0)
    parser.add_argument('--w_BC_cls', type=float, default=5)
    parser.add_argument('--w_BC_binary', type=float, default=50)
    args = parser.parse_args()
    args = get_default_args(args)
    args = set_up_datasets(args)
    if args.save_all_data_base: assert args.save_all_data_novel
    args.checkpoint_dir = '%s/%s' %(args.exp_dir, args.dataset)
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    logging.basicConfig(filename=os.path.join(args.checkpoint_dir, 'test%s.log'%(args.log_tag)), level=logging.INFO)
    logging.info(args)

    args.model_path = '%s/%s/best_model_%s.tar' %(args.exp_dir, args.dataset, args.load_tag)
    print(args)

    # load model
    ## teacher model (base branch)
    model = CosClassifier(args, phase='meta_test')
    tmp = torch.load(args.model_path)
    model.load_state_dict(tmp['state'], strict=False)
    model.cuda() 
    if not args.no_data_init: model.base_mode = 'avg_cos'
    print("Evaluation only on the base branch:")
    acc_list = model.test_inc_loop()
    model.base_weight.data = model.joint_weight.data[:model.base_way]

    # testing
    ## student model (for testing)
    model_test = CosClassifier(args, phase='meta_test', needs_finetune=args.needs_finetune)
    model_test.cuda()
    print("Evaluation on our full method:")
    # init the student model using the teacher model
    model_test.load_trained_model(model)
    # evaluation on all incremental sessions
    acc_list = model_test.test_inc_loop()