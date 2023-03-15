import os
import tqdm
import logging
import warnings
import random

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

def incremental_loop(model, baseloader, novelloader, opts=None, train_mode=False, 
                     optimizer=None, epoch=0, update_interval=1, aug=False):
    '''
    base and True novel data are used for few-shot incremental training or testing
    '''
    loss_avg = 0
    loss_fn = nn.CrossEntropyLoss()
    acc_all  = []
    acc_a2a  = []   # base to base
    acc_b2b  = []   # novel to novel
    acc_a2ab = []   # base to base+novel
    acc_b2ab = []   # novel to base+novel
    delta_a_all = []
    delta_b_all = []      
    iter_num = min(len(baseloader), len(novelloader))
    baseloader = iter(baseloader)
    novelloader = iter(novelloader)
    tqdm_gen = tqdm.tqdm(range(iter_num))

    n_way = model.module.n_way 
    n_shot = model.module.n_support
    n_query = model.module.n_query
    feat_dim = model.module.feat_dim
    base_way = model.module.base_way
    using_unlabeled = ((hasattr(opts, 'transductive') and opts.transductive) \
                        or (hasattr(opts, 'semi') and opts.semi))
    
    if hasattr(opts, 'n_unlabel') and opts.n_unlabel > 0:
        n_unlabel = opts.n_unlabel
    else:
        n_unlabel = 0
    if (hasattr(opts, 'semi') and opts.semi):
        assert n_unlabel > 0

    for i in tqdm_gen:
        # process images
        x_a, y_a = next(baseloader)                                 # (bs*[n_way*(n_query+n_unlabel)], 3, H, W), (bs*[n_way*(n_query+n_unlabel)],)
        x_b, y_b_= next(novelloader)                                # (bs*[n_way*(n_shot+n_query+n_unlabel)], 3, H, W), (bs*[n_way*(n_shot+n_query+n_unlabel)],)
        bs = (x_a.size(0) // n_way) // (n_query + n_unlabel)
        if train_mode: assert bs == opts.task_per_batch
        if aug:
            x_a = x_a[:, 4]
            x_b = x_b[:, 4]
        # feature embeddings
        x_ab = torch.cat([x_a, x_b], dim=0).cuda()                  # (N_ALL, 3, H, W)
        z_ab = model(flag='embedding', input=x_ab)                  # (N_ALL, C)
        z_a = z_ab[:x_a.size(0)]                                    # (bs*[n_way*(n_query+n_unlabel)], C)
        z_b = z_ab[x_a.size(0):]                                    # (bs*[n_way*(n_shot+n_query+n_unlabel)], C)
        # process base features
        z_a = z_a.view(bs, -1, feat_dim)                            # (bs, n_way*n_query + n_way*n_unlabel, C)
        z_a_que = z_a[:, :n_way*n_query]                            # (bs, n_way*n_query, C)
        if n_unlabel > 0:
            z_a_unl = z_a[:, n_way*n_query:]                        # (bs, n_way*n_unlabel, C)
        # process novel features
        z_b = z_b.view(bs, n_way, -1, feat_dim)                     # (bs, n_way, n_shot+n_query+n_unlabel, C)
        z_b_sup = z_b[:, :, :n_shot]                                # (bs, n_way, n_shot, C)
        z_b_sup_org = z_b_sup.clone()                               # (bs, n_way, n_shot, C)
        z_b_que = z_b[:, :, n_shot:n_shot+n_query]                  # (bs, n_way, n_query, C)
        z_b_que = z_b_que.reshape(bs, -1, feat_dim)                 # (bs, n_way*n_query, C)
        if n_unlabel > 0:
            z_b_unl = z_b[:, :, n_shot+n_query:]                    # (bs, n_way, n_unlabel, C)
            z_b_unl = z_b_unl.reshape(bs, -1, feat_dim)             # (bs, n_way*n_unlabel, C)
            z_ab_unl = torch.cat([z_a_unl, z_b_unl], dim=1)         # (bs, 2*n_way*n_unlabel, C)
        # process queries
        z_ab_que = torch.cat([z_a_que, z_b_que], dim=1)             # (bs, 2*n_way*n_query, C)
        y_a = y_a.view(bs, -1)                                      # (bs*n_way*(n_query+n_unlabel),) -> (bs, n_way*(n_query+n_unlabel))
        y_a_que = y_a[:, :n_way*n_query]                            # (bs, n_way*n_query)
        y_b_que = make_nk_label(n_way, n_query, bs, base_way)       # (bs, n_way, n_query)
        y_b_que = y_b_que.view(bs, -1).long()                       # (bs, n_way*n_query)
        y_ab_que = torch.cat([y_a_que, y_b_que], dim=1)             # (bs, 2*n_way*n_query)
        y_ab_que = y_ab_que.view(-1).cuda()                         # (bs*2*n_way*n_query)
        # support labels
        t_sup = make_nk_label(n_way, n_shot, 1, 0)                  # (1, n_way, n_shot)
        t_sup = t_sup.view(-1, 1).long()                            # (n_way*n_shot, 1)
        l_sup = torch.zeros(n_way*n_shot, n_way)                    # (n_way*n_shot, n_way)
        l_sup = l_sup.scatter(-1, t_sup, 1).cuda()                  # (n_way*n_shot, n_way)
        l_sup = l_sup.unsqueeze(0).expand(bs, -1, -1)               # (bs, n_way*n_shot, n_way)
        # unlabeled data
        if hasattr(opts, 'transductive') and opts.transductive:
            z_unl = z_ab_que                                        # (bs, N_unlabel, C)
        elif hasattr(opts, 'semi') and opts.semi:
            z_unl = z_ab_unl                                        # (bs, N_unlabel, C)
        
        if not using_unlabeled:
            scores, inc_weight = model(flag='inc_forward', z_support=z_b_sup, 
                                       z_query=z_ab_que, y_b=None)              # (bs, N_unlabel, base_way+n_way)
        else:
            # maske predictions on unlabeled data
            scores_unl, inc_weight = model(flag='inc_forward', z_support=z_b_sup, 
                                           z_query=z_unl, y_b=None)             # (bs, N_unlabel, base_way+n_way), (bs, base_way+n_way, C)
            if hasattr(opts, 'no_softmax') and opts.no_softmax:
                pass
            else:
                scores_unl = F.softmax(scores_unl, dim=-1)                      # (bs, N_unlabel, base_way+n_way)
            # make semi-supervised/transductive predictions
            if opts.MAP:
                vals_base, _ = torch.max(scores_unl[:, :, :base_way], dim=-1)   # (bs, N_unlabel)
                vals_novel, _= torch.max(scores_unl[:, :, base_way:], dim=-1)   # (bs, N_unlabel)
                mask_novel = (vals_novel >= vals_base + opts.masking_thres)     # (bs, N_unlabel)
                mask_novel = mask_novel.float()                                 # (bs, N_unlabel)
                mask_sup = torch.ones(bs, l_sup.size(1)).cuda()                 # (bs, n_way*n_shot)
                all_mask = torch.cat([mask_novel, mask_sup], dim=1)             # (bs, N_all)
                all_mask = all_mask.unsqueeze(1)                                # (bs, 1, N_all)
                scores_novel = scores_unl[:, :, base_way:]                      # (bs, N_unlabel, n_way)
                if opts.renorm:
                    scores_novel /= scores_novel.sum(dim=-1, keepdim=True)      # (bs, N_unlabel, n_way)
                all_data = torch.cat([z_unl, z_b_sup.reshape(bs, -1, feat_dim)], dim=1) # (bs, N_all, C)
                all_lb = torch.cat([scores_novel, l_sup], dim=1)                # (bs, N_all, n_way)
                all_lb = all_lb.permute(0, 2, 1)                                # (bs, n_way, N_all)
                if opts.masking:
                    all_lb = all_lb * all_mask                                  # (bs, n_way, N_all)
                if opts.norm_first:
                    all_data = F.normalize(all_data, p=2, dim=-1, eps=1e-12)    # (bs, N_all, C)
                novel_weight = torch.bmm(all_lb, all_data)                      # (bs, n_way, N_all) * (bs, N_all, C) -> (bs, n_way, C)
                novel_weight = novel_weight / all_lb.sum(dim=-1, keepdim=True)  # (bs, n_way, C)
                novel_weight = F.normalize(novel_weight, p=2, dim=-1, eps=1e-12)# (bs, n_way, C)
                novel_weight_org = inc_weight[:, base_way:]                     # (bs, base_way+n_way, C) -> (bs, n_way, C)
                novel_weight = opts.meta_lr * (novel_weight - novel_weight_org) + novel_weight_org
            else:
                n_selected_list = [int(x) for x in opts.n_selected.split(',')]
                n_selected = n_selected_list[-1]
                z_sup_new = torch.zeros(bs, n_way, n_shot+n_selected, feat_dim).cuda()
                scores_novel = scores_unl[:, :, base_way:]                      # (bs, N_unlabel, n_way)
                for c in range(n_way):
                    scores_ = scores_novel[:, :, c]                             # (bs, N_unlabel)
                    _, indx = torch.topk(scores_, n_selected, dim=-1)           # (bs, n_selected)
                    indx = indx.unsqueeze(-1).expand(-1, -1, feat_dim)          # (bs, n_selected, C)
                    z_pseudo = torch.gather(z_unl, 1, indx)                     # (bs, N_unlabel, C) -> (bs, n_selected, C)
                    z_pseudo = torch.cat([z_b_sup_org[:, c], z_pseudo], dim=1)  # (bs, n_shot+n_selected, C)
                    z_sup_new[:, c] = z_pseudo                                  # (bs, n_shot+n_selected, C)
                if opts.norm_first:
                    z_sup_new = F.normalize(z_sup_new, p=2, dim=-1, eps=1e-12)  # (bs, n_way, n_shot+n_selected, C)
                novel_weight = z_sup_new.mean(-2)                               # (bs, n_way, C)
            # final predictions  
            base_weight = model.module.base_weight                              # (base_way, C)
            base_weight = base_weight.unsqueeze(0).expand(bs, -1, -1)           # (bs, base_way, C)
            inc_weight = torch.cat([base_weight, novel_weight], dim=1)          # (bs, base_way+n_way, C)
            inc_weight = F.normalize(inc_weight, p=2, dim=-1, eps=1e-12)        # (bs, base_way+n_way, C)   
            querys = F.normalize(z_ab_que, p=2, dim=-1, eps=1e-12)              # (bs, 2*n_way*n_query, C)
            scores = torch.bmm(querys, inc_weight.permute(0, 2, 1))             # (bs, 2*n_way*n_query, C) * (bs, C, base_way+n_way) -> (bs, 2*n_way*n_query, base_way+n_way)
            scores = model.module.scale_cls * scores                            # (bs, 2*n_way*n_query, base_way+n_way)
        
        scores = scores.view(-1, scores.size(-1))                               # (bs*2*n_way*n_query, base_way+n_way)

        if train_mode:
            loss = loss_fn(scores, y_ab_que)

            loss.backward()
            if ((i+1) % update_interval) == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            loss_avg += loss.item()
            tqdm_gen.set_description('e:%d loss = %.4f' %(epoch, loss.item()))
        
        dic = correct_inc(scores, y_ab_que, n_base=base_way)                    # compute accuracy on current batch
        acc_old_new, delta_a, delta_b, acc_old2, acc_new2 = dic['acc_old_new'], dic['delta_a'], dic['delta_b'], dic['acc_old2'], dic['acc_new2']
        acc_all.append(acc_old_new * 100)
        acc_a2a.append(acc_old2 * 100)
        acc_b2b.append(acc_new2 * 100)
        acc_a2ab.append((acc_old2+delta_a) * 100)
        acc_b2ab.append((acc_new2+delta_b) * 100)
        delta_a_all.append(delta_a * 100)
        delta_b_all.append(delta_b * 100)

    record = summarize_inc(acc_all, acc_a2a, acc_b2b, acc_a2ab, acc_b2ab, delta_a_all, delta_b_all, train_mode, loss_avg, epoch)
    return record


def fakenovel_loop(model, jointloader, opts=None, train_mode=True, optimizer=None, epoch=0, update_interval=1):
    '''
    base and Fake novel data are used for few-shot incremental training
    '''
    assert train_mode == True
    loss_fn = nn.CrossEntropyLoss()
    loss_avg = 0   
    acc_all = []
    iter_num = len(jointloader)
    tqdm_gen = tqdm.tqdm(jointloader)

    n_way = model.n_way 
    n_shot = model.n_support
    n_query = model.n_query
    feat_dim = model.feat_dim
    base_way = model.base_way
    using_unlabeled = ((hasattr(opts, 'transductive') and opts.transductive) \
                        or (hasattr(opts, 'semi') and opts.semi))
    
    if hasattr(opts, 'n_unlabel') and opts.n_unlabel > 0:
        n_unlabel = opts.n_unlabel
    else:
        n_unlabel = 0
    if (hasattr(opts, 'semi') and opts.semi):
        assert n_unlabel > 0

    for i, X in enumerate(tqdm_gen):
        # process images
        x_ab, label = X                                             # (bs*[n_way*(n_query+n_unlabel)+n_way*(n_shot+n_query+n_unlabel)], 3, H, W)
        x_ab = x_ab.cuda()
        bs = x_ab.size(0) // (n_way*(n_query+n_unlabel)+n_way*(n_shot+n_query+n_unlabel))
        if train_mode: assert bs == opts.task_per_batch

        if hasattr(opts, 'add_rotation') and opts.add_rotation:
            x_ab = x_ab.view(bs, -1, *x_ab.shape[1:])               # (bs, [n_way*(n_query+n_unlabel)+n_way*(n_shot+n_query+n_unlabel)], 3, H, W)
            base_bias = n_way * (n_query + n_unlabel)
            novel_bias = n_shot + n_query + n_unlabel               # (bs, [base_bias + n_way * novel_bias], 3, H, W)
            for i in range(bs):
                for k in range(n_way):
                    rot_list = [0, 90, 180, 270]
                    sel_rot = random.choice(rot_list)
                    if sel_rot == 90:
                        x_ab[i, base_bias+k*novel_bias: base_bias+(k+1)*novel_bias] = x_ab[i, base_bias+k*novel_bias: base_bias+(k+1)*novel_bias].transpose(2, 3).flip(2)
                    elif sel_rot == 180:
                        x_ab[i, base_bias+k*novel_bias: base_bias+(k+1)*novel_bias] = x_ab[i, base_bias+k*novel_bias: base_bias+(k+1)*novel_bias].flip(2).flip(3)
                    elif sel_rot == 270:
                        x_ab[i, base_bias+k*novel_bias: base_bias+(k+1)*novel_bias] = x_ab[i, base_bias+k*novel_bias: base_bias+(k+1)*novel_bias].transpose(2, 3).flip(3)
            x_ab = x_ab.reshape(-1, *x_ab.shape[2:])
        
        # process features
        z_ab = model(flag='embedding', input=x_ab)                  # (bs*[n_way*(n_query+n_unlabel)+n_way*(n_shot+n_query+n_unlabel)], C)
        z_ab = z_ab.view(bs, -1, feat_dim)                          # (bs, n_way*(n_query+n_unlabel)+n_way*(n_shot+n_query+n_unlabel), C)
        z_a  = z_ab[:, :n_way*(n_query+n_unlabel)]                  # (bs, n_way*(n_query+n_unlabel), C)
        z_a_que = z_a[:, :n_way*n_query]                            # (bs, n_way*n_query, C)
        if n_unlabel > 0:
            z_a_unl = z_a[:, n_way*n_query:]                        # (bs, n_way*n_unlabel, C)
        
        z_b  = z_ab[:, n_way*(n_query+n_unlabel):]                  # (bs, n_way*(n_shot+n_query+n_unlabel), C)
        z_b  = z_b.view(bs, n_way, -1, feat_dim)                    # (bs, n_way, n_shot+n_query+n_unlabel, C)
        z_b_sup = z_b[:, :, :n_shot]                                # (bs, n_way, n_shot, C)
        z_b_que = z_b[:, :, n_shot:n_shot+n_query]                  # (bs, n_way, n_query, C)
        z_b_que = z_b_que.contiguous().view(bs, -1, feat_dim)       # (bs, n_way*n_query, C)
        if n_unlabel > 0:
            z_b_unl = z_b[:, :, n_shot+n_query:]                    # (bs, n_way, n_unlabel, C)
            z_b_unl = z_b_unl.contiguous().view(bs, -1, feat_dim)   # (bs, n_way*n_unlabel, C)
            z_ab_unl = torch.cat([z_a_unl, z_b_unl], dim=1)         # (bs, 2*n_way*n_unlabel, C)
        z_ab_que = torch.cat([z_a_que, z_b_que], dim=1)             # (bs, 2*n_way*n_query, C)
        # process labels
        label = label.view(bs, -1)                                  # (bs, n_way*(n_query+n_unlabel)+n_way*(n_shot+n_query+n_unlabel))
        y_a_que = label[:, :n_way*n_query]                          # (bs, n_way*n_query)
        y_b = label[:, n_way*(n_query+n_unlabel):]                  # (bs, n_way*(n_shot+n_query+n_unlabel))
        y_b = y_b.view(bs, n_way, n_shot+n_query+n_unlabel)         # (bs, n_way, n_shot+n_query+n_unlabel)
        y_b_= y_b[:, :, 0].cuda()                                   # (bs, n_way), y_b_ denotes fake novel class indices
        y_b_que = y_b[:, :, n_shot:n_shot+n_query]                  # (bs, n_way, n_query) 
        y_b_que = y_b_que.contiguous().view(bs, -1)                 # (bs, n_way*n_query)
        y_ab_que = torch.cat([y_a_que, y_b_que], dim=1).cuda()      # (bs, 2*n_way*n_query)
        y_ab_que = y_ab_que.view(-1)                                # (bs*2*n_way*n_query)
        # support labels
        t_sup = make_nk_label(n_way, n_shot, 1, 0)                  # (1, n_way, n_shot)
        t_sup = t_sup.view(-1, 1).long()                            # (n_way*n_shot, 1)
        l_sup = torch.zeros(n_way*n_shot, n_way)                    # (n_way*n_shot, n_way)
        l_sup = l_sup.scatter(-1, t_sup, 1).cuda()                  # (n_way*n_shot, n_way)
        l_sup = l_sup.unsqueeze(0).expand(bs, -1, -1)               # (bs, n_way*n_shot, n_way)
        # unlabeled data
        if hasattr(opts, 'transductive') and opts.transductive:
            z_unl = z_ab_que                                        # (bs, N_unlabel, C)
        elif hasattr(opts, 'semi') and opts.semi:
            z_unl = z_ab_unl                                        # (bs, N_unlabel, C)

        if not using_unlabeled:
            scores, inc_weight = model(flag='inc_forward', z_support=z_b_sup, 
                                       z_query=z_ab_que, y_b=y_b_)  # (bs, 2*n_way*n_query, base_way)
        else:
            # maske predictions on unlabeled data
            scores_unl, inc_weight = model(flag='inc_forward', z_support=z_b_sup, 
                                           z_query=z_unl, y_b=y_b_)             # (bs, N_unlabel, base_way)
            if hasattr(opts, 'no_softmax') and opts.no_softmax:
                pass
            else:
                scores_unl = F.softmax(scores_unl, dim=-1)                      # (bs, N_unlabel, base_way+n_way)
            # fake novel indices            
            y_b_list = y_b_.cpu().numpy().tolist()                              # (bs, n_way)
            y_c_ = []                                                           # y_c_ is the complementary set of y_b_
            for x in y_b_:
                set_union  = set(range(base_way)) - set(x)
                list_union = list(set_union)
                list_union.sort()
                y_c_.append(list_union)
            y_c_ = torch.from_numpy(np.array(y_c_)).long().cuda()               # (bs, base_way-n_way)
            y_b_score = y_b_.unsqueeze(1).expand(-1, z_unl.size(1), -1)         # (bs, N_unlabel, n_way)
            y_c_score = y_c_.unsqueeze(1).expand(-1, z_unl.size(1), -1)         # (bs, N_unlabel, base_way-n_way)
            scores_novel= torch.gather(scores_unl, 2, y_b_score)                # (bs, N_unlabel, n_way)
            scores_base = torch.gather(scores_unl, 2, y_c_score)                # (bs, N_unlabel, base_way-n_way)
            index = y_b_.unsqueeze(2).expand(-1, -1, feat_dim)                  # (bs, n_way, C)
            if opts.MAP:
                vals_base, _ = torch.max(scores_base, dim=-1)                   # (bs, N_unlabel)
                vals_novel, _= torch.max(scores_novel, dim=-1)                  # (bs, N_unlabel)
                mask_novel = (vals_novel >= vals_base + opts.masking_thres)     # (bs, N_unlabel)
                mask_novel = mask_novel.float()                                 # (bs, N_unlabel)
                mask_sup = torch.ones(bs, l_sup.size(1)).cuda()                 # (bs, n_way*n_shot)
                all_mask = torch.cat([mask_novel, mask_sup], dim=1)             # (bs, N_all)
                all_mask = all_mask.unsqueeze(1)                                # (bs, 1, N_all)
                if opts.renorm:
                    scores_novel /= scores_novel.sum(dim=-1, keepdim=True)      # (bs, N_unlabel, n_way)
                all_data = torch.cat([z_unl, z_b_sup.reshape(bs, -1, feat_dim)], dim=1) # (bs, N_all, C)
                all_lb = torch.cat([scores_novel, l_sup], dim=1)                # (bs, N_all, n_way)
                all_lb = all_lb.permute(0, 2, 1)                                # (bs, n_way, N_all)
                if opts.masking:
                    all_lb = all_lb * all_mask                                  # (bs, n_way, N_all)
                if opts.norm_first:
                    all_data = F.normalize(all_data, p=2, dim=-1, eps=1e-12)    # (bs, N_all, C)
                novel_weight = torch.bmm(all_lb, all_data)                      # (bs, n_way, N_all) * (bs, N_all, C) -> (bs, n_way, C)
                novel_weight = novel_weight / all_lb.sum(dim=-1, keepdim=True)  # (bs, n_way, C)
                novel_weight = F.normalize(novel_weight, p=2, dim=-1, eps=1e-12)# (bs, n_way, C)
                novel_weight_org = torch.gather(inc_weight, 1, index)           # (bs, base_way, C) -> (bs, n_way, C)
                novel_weight = opts.meta_lr * (novel_weight - novel_weight_org) + novel_weight_org
            else:
                pass
            # final predictions  
            base_weight = model.module.base_weight                              # (base_way, C)
            base_weight = base_weight.unsqueeze(0).expand(bs, -1, -1)           # (bs, base_way, C)
            inc_weight = base_weight.scatter(1, index, novel_weight)            # (bs, base_way, C)
            inc_weight = F.normalize(inc_weight, p=2, dim=-1, eps=1e-12)        # (bs, base_way, C)   
            querys = F.normalize(z_ab_que, p=2, dim=-1, eps=1e-12)              # (bs, 2*n_way*n_query, C)
            scores = torch.bmm(querys, inc_weight.permute(0, 2, 1))             # (bs, 2*n_way*n_query, C) * (bs, C, base_way) -> (bs, 2*n_way*n_query, base_way)
            scores = model.module.scale_cls * scores                            # (bs, 2*n_way*n_query, base_way)
            
        scores = scores.view(-1, scores.size(-1))                               # (bs*2*n_way*n_query, base_way)

        acc = top1(scores, y_ab_que)
        acc_all.append(acc * 100)  
        
        loss = loss_fn(scores, y_ab_que)
        loss_avg += loss.item()
        
        loss.backward()
        if ((i+1) % update_interval) == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        tqdm_gen.set_description('e:%d loss = %.4f' %(epoch, loss.item()))

    record = {}
    acc_all  = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all)
    record['acc']  = acc_mean
    record['loss'] = loss_avg / iter_num

    out_str = 'Train epoch: %d avg loss: %.6f Acc both: %4.2f%% +- %4.2f%%' \
               %(epoch, loss_avg/iter_num, acc_mean, 1.96*acc_std/np.sqrt(iter_num))
    print(out_str)
    logging.info(out_str)
    
    return record


def summarize_inc(acc_all, acc_a2a, acc_b2b, acc_a2ab, acc_b2ab, delta_a_all, delta_b_all, 
                  train_mode=False, loss_avg=0, epoch=0):
    '''
    compute mean accuracy and delta metrics over all incremental few-shot episodes
    '''
    iter_num = len(acc_all)
    loss_avg = loss_avg / iter_num
    acc_all  = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all)

    delta_a_all  = np.asarray(delta_a_all)
    delta_b_all  = np.asarray(delta_b_all)
    delta_a_mean = np.mean(delta_a_all)
    delta_b_mean = np.mean(delta_b_all)
    delta_mean   = 0.5 * (delta_a_mean + delta_b_mean)

    acc_a2a    = np.asarray(acc_a2a)
    acc_b2b    = np.asarray(acc_b2b)
    acc_a_mean = np.mean(acc_a2a)
    acc_b_mean = np.mean(acc_b2b)
    acc_a2ab   = np.asarray(acc_a2ab)
    acc_b2ab   = np.asarray(acc_b2ab)
    acc_a2ab_mean = np.mean(acc_a2ab)
    acc_b2ab_mean = np.mean(acc_b2ab)
    
    if train_mode == False:
        out_str = '%d Test Acc = %4.2f%% +- %4.2f%% Delta = %4.2f%% delta_a = %4.2f%% delta_b = %4.2f%% Base Acc = %4.2f%% Novel Acc = %4.2f%%' \
                %(iter_num, acc_mean, 1.96*acc_std/np.sqrt(iter_num), delta_mean, delta_a_mean, delta_b_mean, acc_a_mean, acc_b_mean)
    else:
        out_str = 'Train epoch: %d avg loss: %.6f Acc = %4.2f%% +- %4.2f%% Delta = %4.2f%% delta_a = %4.2f%% delta_b = %4.2f%% Base Acc = %4.2f%% Test Acc = %4.2f%%' \
                  %(epoch, loss_avg, acc_mean, 1.96* acc_std/np.sqrt(iter_num), delta_mean, delta_a_mean, delta_b_mean, acc_a_mean, acc_b_mean)
    print(out_str)
    logging.info(out_str)

    dic = {}
    dic['acc'] = acc_mean
    dic['delta']   = delta_mean
    dic['delta_a'] = delta_a_mean
    dic['delta_b'] = delta_b_mean
    dic['base2base']   = acc_a_mean
    dic['novel2novel'] = acc_b_mean
    dic['base2all']    = acc_a2ab_mean
    dic['novel2all']   = acc_b2ab_mean
    if train_mode: 
        dic['loss'] = loss_avg
    return dic


def correct_inc(scores, y, n_base=64):
    '''
    compute accuracy and delta metrics for the current incremental few-shot episode
    if train_mode == True, return a classification loss
    scores.shape = (N, (base_way+n_way))
    y.shape = (N,)
    '''  
    acc_old_new = top1(scores, y)

    old_idx = y < n_base
    new_idx = y >= n_base
    pred_new = scores[:, n_base:]
    pred_old = scores[:, :n_base]
    y_new = y - n_base
    acc_old  = top1(scores[old_idx], y[old_idx])
    acc_old2 = top1(pred_old[old_idx], y[old_idx])
    acc_new  = top1(scores[new_idx], y[new_idx])
    acc_new2 = top1(pred_new[new_idx], y_new[new_idx])
    delta_a  = acc_old - acc_old2
    delta_b  = acc_new - acc_new2

    dic = {} 
    dic['acc_old_new'] = acc_old_new    # acc_all
    dic['delta_a']  = delta_a           # delta_base
    dic['delta_b']  = delta_b           # delta_novel
    dic['acc_old2'] = acc_old2          # base2base
    dic['acc_new2'] = acc_new2          # novel2novel   
    return dic


def top1(pred, label):
    '''
    compute top1 accuarcy
    pred.shape  = (n_batch, n_class)
    label.shape = (n_batch,)
    '''
    label = label.cpu().numpy()
    topk_scores, topk_labels = pred.data.topk(1, 1, True, True) # topk: Returns the k largest elements of the given input tensor along a given dimension.
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = np.sum(topk_ind[:,0] == label)

    acc = float(top1_correct)/len(label)
    return acc


def make_nk_label(n, k, ep_per_batch=1, base_way=0):
    '''
    n-way, k-shot labels
    return (ep_per_batch, n_way, k_shot)
    '''
    label = torch.arange(n).unsqueeze(1).expand(n, k) + base_way
    label = label.unsqueeze(0).expand(ep_per_batch, -1, -1)
    return label.contiguous()


def one_hot(self, label, num_class):
    '''return one-hot label of shape: (B, n_class)'''
    return torch.zeros((len(label), num_class)).to(label.device).scatter_(1, label.unsqueeze(1), 1)