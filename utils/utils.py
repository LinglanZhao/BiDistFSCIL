import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import warnings
import numpy as np
from torch.distributions import beta
from copy import deepcopy

def top1_acc(pred, label):
    '''
    compute top1 accuarcy
    pred.shape  = (n_batch, n_class)
    label.shape = (n_batch,)
    '''
    _, pred_y = torch.max(pred, dim=1)
    acc = (pred_y == label).float().mean().item()
    return acc


def set_bn_layers(model, bn_eval=False, bn_frozen=False):
    if bn_eval:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                if bn_frozen:
                    for params in m.parameters():
                        params.requires_grad = False


def set_bn_state(model, set_eval=True):
    if set_eval:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    else:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()


def label2session(x_input, base_way, n_way):
    x = deepcopy(x_input)
    for i in range(len(x)):
        if x[i] < base_way:
            x[i] = 0
        else:
            x[i] = ((x[i] - base_way) // n_way + 1)
    return x
    
def normalize_l2(x, axis=-1):
    '''x.shape = (num_samples, feat_dim)'''
    x_norm = np.linalg.norm(x, axis=axis, keepdims=True)
    x = x / (x_norm + 1e-8)
    return x


def QRreduction(datas):
    '''
    ndatas.shape = (bs, n_samples, feat_dim)
    '''
    ndatas = torch.qr(datas.permute(0,2,1)).R
    ndatas = ndatas.permute(0,2,1)
    return ndatas


def centerDatas(datas):
    '''
    datas.shape = (bs, n_samples, feat_dim)
    '''
    datas = datas - datas.mean(1, keepdim=True)
    datas = F.normalize(datas, p=2, dim=-1, eps=1e-12)
    return datas


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(1).mean()
        return b


def convert2tempLabel(loc_lb, n_base_class=0):
    label_set = list(torch.unique(loc_lb))
    tmp_lb = torch.tensor([label_set.index(x) for x in loc_lb]) + n_base_class
    return tmp_lb.long()


def adjust_learning_rate(optimizer, epoch, init_lr=0.1, n_epoch=400):
    lr = init_lr * (10 ** (-4 * epoch / n_epoch)) # initial learning rate = 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_lr(optimizer, start_lr, lr_step, epoch):
    """Sets the learning rate to the initial LR decayed by 0.05 every 10 epochs"""
    lr = start_lr * (0.1 ** (epoch / lr_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def base_evaluate_acc(model, dataloader):
    model.eval()
    anumerator = 0
    denominator = 0
    with torch.no_grad():
        for i, X in enumerate(dataloader):
            data, label = X
            data = data.cuda()
            label = label.cuda()
            pred = model.base_forward(data)
            _, pred = torch.max(pred, dim=1)
            anumerator += (pred == label).sum()
            denominator += data.shape[0]
        return float(anumerator) / (denominator + 1e-10)


def accuracy(preds, label):
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum
            

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


class log():
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path


    def print(self, string):
        '''
        Write one line of log into screen and file.
        log_file_path: Path of log file.
        string:        String to write in log file.
        '''
        with open(self.log_file_path, 'a+') as f:
            f.write(string + '\n')
            f.flush()
        print(string)


class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T=1):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t, islabel=False):
        p_s = F.log_softmax(y_s/self.T, dim=1) # mathematically equivalent to log(softmax(x))
        if islabel == False:
            p_t = F.softmax(y_t/self.T, dim=1)
        else:
            p_t = y_t
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0] # If the field size_average is set to False, the losses are instead summed for each minibatch
        return loss


def KDloss(input, target, tmp=4):
    log_p = torch.log_softmax(input/tmp, dim=1)
    q = torch.softmax(target/tmp, dim=1)
    kd = torch.nn.KLDivLoss(reduction='sum')
    loss = kd(log_p, q) * (tmp**2) / input.size(0)
    return loss


def w_KDloss(input, target, tmp=4, weights=None, t_prob=False):
    log_p = torch.log_softmax(input/tmp, dim=1) 
    if (t_prob == True):
        q = target
    else:
        q = torch.softmax(target/tmp, dim=1)
    if weights is None:
        weights = torch.ones(log_p.shape[0]).cuda()
    loss = cal_kl_div(log_p, q, weights)
    loss = loss * (tmp**2) / weights.sum()
    return loss


def cal_kl_div(log_p, q, weights):
    '''
    log_p.shape = (B, n_way_all)
    q.shape = (B, n_way_all)
    '''
    log_q = torch.log(q)
    loss = q * (log_q - log_p)              # (B, n_way_all)
    loss = loss * weights.unsqueeze(-1)     # (B, n_way_all)
    loss = loss.sum()
    return loss


def convert2one_hot(label, num_class):
    '''return one-hot label of shape: (B, n_class)
    '''
    return torch.zeros((len(label), num_class)).to(label.device).scatter_(1, label.unsqueeze(1), 1)


def mix_up(data_l, target_l, data_r, target_r, alpha=0.75,
           force_left_major=True, norm_first=False):
    """
    Args:
        data_l, data_r: (B, C) or (B, 3, H, W)
        target_l, target_r: (B, num_cls)
    Returns: 
        mix_data: (B, C) or (B, 3, H, W)
        mix_target: (B, num_cls)
    """
    assert data_l.size() == data_r.size() and target_l.size() == target_r.size() and data_l.dim() in {2, 4}
    if data_l.dim() == 4: # input mixup
        mix_lamda = beta.Beta(alpha, alpha).sample(sample_shape=(data_l.size(0), 1, 1, 1)).cuda()
    else: # feature mixup
        mix_lamda = beta.Beta(alpha, alpha).sample(sample_shape=(data_l.size(0), 1)).cuda()
    
    if force_left_major:
        mix_lamda = torch.max(mix_lamda, 1 - mix_lamda)

    if data_l.dim() == 2: # feature mixup
        if norm_first == True:
            data_l = F.normalize(data_l, p=2, dim=1, eps=1e-12)
            data_r = F.normalize(data_r, p=2, dim=1, eps=1e-12)
        mix_data = mix_lamda * data_l + (1 - mix_lamda) * data_r    
        mix_data = F.normalize(mix_data, p=2, dim=1, eps=1e-12)     # (B, C)
    else:
        mix_data = mix_lamda * data_l + (1 - mix_lamda) * data_r    # (B, 3, H, W)

    if data_l.dim() == 4: # input mixup
        mix_target = mix_lamda[:, :, 0, 0] * target_l + (1 - mix_lamda[:, :, 0, 0]) * target_r
    else:
        mix_target = mix_lamda * target_l + (1 - mix_lamda) * target_r
    
    return mix_data, mix_target


def incremental_mixup_pairs(que_ab, y_ab, num_class):
    '''
    input:
        que_ab.shape = (B, C) or (B, 3, H, W)
        y_ab.shape = (B, )
    return:
        data.shape = (B, C) or (B, 3, H, W)
        target.shape = (B, num_class)
    '''
    assert que_ab.size(0) == y_ab.size(0)
    que_ab = que_ab.cuda()
    y_ab = y_ab.cuda()
    bs = que_ab.size(0)
    bs0 = bs // 2                                       # n_way * n_query

    que_a, que_b = que_ab[:bs0], que_ab[bs0:]
    y_a, y_b = y_ab[:bs0], y_ab[bs0:]

    data_l = torch.cat([que_a, que_a], dim=0)
    target_l = torch.cat([y_a, y_a], dim=0)
    target_l = convert2one_hot(target_l, num_class)
    data_r = que_ab
    target_r = y_ab
    target_r = convert2one_hot(target_r, num_class)
    
    idx = torch.randperm(bs)
    data_l = data_l[idx]                                     # (B, C) or (B, 3, H, W)
    target_l = target_l[idx]                                 # (B, )
    
    return data_l, target_l, data_r, target_r


def generate_mixup_pairs_backup(x, y, nearest=False):
    '''
    x.shape = (B, C) or (B, 3, H, W)
    y.shape = (B, )
    '''
    x = x.cuda()
    y = y.cuda()
    if x.dim() == 2 and nearest == True:
        x_norm = F.normalize(x, p=2, dim=1, eps=1e-12)  # (B, C)
        sim_matrix = torch.mm(x_norm, x_norm.t())       # (B, B)
        diff = y.unsqueeze(1) - y.unsqueeze(0)          # (B, B)
        mask = (diff == 0)                              # (B, B) 1 if same, else 0
        sim_matrix[mask] = -2                           # (B, B)
        _, idx = torch.max(sim_matrix, dim=1)           # (B, )
    else:
        idx = torch.randperm(x.size(0))
    data_l = x                                          # (B, C)
    data_r = x[idx]                                     # (B, C) or (B, 3, H, W)
    target_l = y                                        # (B, )
    target_r = y[idx]                                   # (B, )
    
    return data_l, target_l, data_r, target_r


def mix_up_backup(data_l, data_r, alpha=0.75, force_left_major=False, norm_first=False):
    """
    mixup two group of samples:
        lamda~Beta(alpha)
        new_x  = lamda*x1 + (1-lamda)*x2
    """

    assert data_l.size() == data_r.size()
    
    mix_lamda = np.random.beta(alpha, alpha)
    if force_left_major:
        mix_lamda = max(mix_lamda, 1 - mix_lamda)

    if data_l.dim() == 2: # is feature
        if norm_first == True:
            data_l = F.normalize(data_l, p=2, dim=1, eps=1e-12)
            data_r = F.normalize(data_r, p=2, dim=1, eps=1e-12)
        mix_data = mix_lamda * data_l + (1 - mix_lamda) * data_r    
        mix_data = F.normalize(mix_data, p=2, dim=1, eps=1e-12)     # (B, C)
    else:
        mix_data = mix_lamda * data_l + (1 - mix_lamda) * data_r    # (B, 3, H, W)
    
    return mix_data, mix_lamda


def mixup_criterion(pred, y_a, y_b, lam):
    criterion = nn.CrossEntropyLoss()
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_default_args(args):
    if 'mini' in args.dataset.lower():
        args.bn_eval = True
    args.norm_first = True
    args.using_exemplars = True
    args.BC_detach = True
    torch.manual_seed(86)
    torch.cuda.manual_seed_all(86)
    np.random.seed(86)
    random.seed(86)
    torch.backends.cudnn.deterministic = True
    return args


def kl_loss(pred, target, weight=None, reduction='mean'):
    """
    consistent with nn.KLDivLoss
    Args:
        pred (torch.Tensor): softmax result of network output
        target (torch.Tensor): target prob vector
        weight ():
        reduction (str):

    Returns: torch.Tensor

    """

    return F.kl_div((pred + 1e-10).log(), target, reduction=reduction)


def symmetric_kl_loss(pred, target, reduction='mean'):
    '''
    symmetric format of kl loss
    '''

    return F.kl_div((pred + 1e-10).log(), target, reduction=reduction) + \
           F.kl_div((target + 1e-10).log(), pred, reduction=reduction)