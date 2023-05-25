import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from PIL import Image
import numpy as np
import random
import tqdm
import logging
from copy import deepcopy
from utils.utils import * 
from utils.fsl_inc import * 
from dataloader.data_utils import *
from dataloader.samplers import *
from dataloader.exemplar_set import ExemplarSet
from models.resnet18_encoder import *
from models.resnet20_cifar import *
from models.binary_classifier import BinaryNet
from models.mlp_models import ScalarNet


class CosClassifier(nn.Module):
    def __init__(self, args, phase='pre_train', needs_finetune=False):
        super(CosClassifier, self).__init__()
        self.args = args
        self.phase = phase
        self.dataset = args.dataset
        self.method = args.method
        self.norm_first = args.norm_first
        self.base_mode = args.base_mode
        self.base_way = args.base_class
        self.joint_way = args.num_classes
        self.n_sessions = args.n_sessions
        self.needs_finetune = needs_finetune # needs finetune during testing
        self.loss_fn = nn.CrossEntropyLoss()
        self.dic_wfpg = {}

        # feature extractor
        if self.dataset == 'mini_imagenet':
            self.feature = resnet18(False, args)
        self.feat_dim = self.feature.final_feat_dim
        
        # meta-training hyper-parameters
        if self.phase == 'meta_train':
            print('loading meta training parameters')
            self.n_way = args.n_way_train
            self.n_query = args.n_query_train 
            self.n_support = args.n_shot_train
        assert self.phase in {'pre_train', 'meta_train', 'meta_test'}
          
        # base weight parameters       
        base_weight = torch.FloatTensor(self.base_way, self.feat_dim).normal_(0.0, np.sqrt(2.0 / self.feat_dim))
        self.base_weight = nn.Parameter(base_weight, requires_grad=True)
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10.0), requires_grad=True)
        self.joint_weight = None                                        # joint weight is used only in the incremental testing phase
        # LwoF parameters
        self.scale_att = nn.Parameter(torch.FloatTensor(1).fill_(10.0), requires_grad=True)
        w_avg = torch.FloatTensor(self.feat_dim).fill_(1)               # initialize to the identity transform
        self.w_avg = nn.Parameter(w_avg, requires_grad=True)
        w_att = torch.FloatTensor(self.feat_dim).fill_(1)               # initialize to the identity transform
        self.w_att = nn.Parameter(w_att, requires_grad=True)
        w_keys = torch.FloatTensor(self.base_way, self.feat_dim).normal_(0.0, np.sqrt(2.0 / self.feat_dim))
        self.w_keys = nn.Parameter(w_keys, requires_grad=True)
        w_q = torch.eye(self.feat_dim, self.feat_dim) + torch.randn(self.feat_dim, self.feat_dim) * 0.001
        self.w_q = nn.Parameter(w_q, requires_grad=True)
    

    def reset_n_way(self, n_way=5):
        '''reset the --n_way argument in Cosine Classifier'''
        self.n_way = n_way

    
    def load_trained_model(self, model_trained):
        '''
        init current (t) feature and weights
        define previous (t-1) and original (fixed t=0) features and weights
        '''
        # init the student model using the teacher model 
        self.load_state_dict(model_trained.state_dict(), strict=False)
        
        # the following modules are only used in finetuning for distillation
        if self.needs_finetune == True:
            # the teacher model has the whole classification weights which will be used in finetuning without data leakage
            assert model_trained.joint_weight.shape[0] == self.args.num_classes
            # define/init features
            self.feature_ema   = resnet18(False, self.args).cuda()  # model with EMA parameters
            self.feature_prev  = resnet18(False, self.args).cuda()  # model t-1
            self.feature_fixed = resnet18(False, self.args).cuda()  # model t=0
            self.feature_ema.load_state_dict(model_trained.feature.state_dict(), strict=True)
            self.feature_prev.load_state_dict(model_trained.feature.state_dict(), strict=True) 
            self.feature_fixed.load_state_dict(model_trained.feature.state_dict(), strict=True)
            self.feature_ema.eval()
            self.feature_prev.eval()
            self.feature_fixed.eval()
            for p in self.feature_ema.parameters():
                p.requires_grad = False
            for p in self.feature_prev.parameters():
                p.requires_grad = False
            for p in self.feature_fixed.parameters():
                p.requires_grad = False
            # define/init weights
            self.weights_ema   = model_trained.joint_weight[:self.base_way].clone().data
            self.weights_prev  = model_trained.joint_weight[:self.base_way].clone().data
            self.weights_fixed = model_trained.joint_weight.clone().data
            self.weights_ema.requires_grad = False
            self.weights_prev.requires_grad = False
            self.weights_fixed.requires_grad = False

            if hasattr(self.args, 'EMA_logits'):
                weights_tmp = F.normalize(self.weights_fixed, p=2, dim=1, eps=1e-12) 
                sim_matrix = torch.mm(weights_tmp, weights_tmp.t())                     # (n_way_all, n_way_all)
                self.sim_n2all   = sim_matrix[self.base_way:, :]                        # (n_way_novel, n_way_all)
                self.sim_n2all   = self.sim_n2all.cuda()                                # (n_way_novel, n_way_all)
                self.sim_n2base  = self.sim_n2all[:, :self.base_way]                    # (n_way_novel, n_way_base)
                self.sim_n2novel = self.sim_n2all[:, self.base_way:]                    # (n_way_novel, n_way_novel)
                # print(self.sim_n2all.shape)
                if self.args.EMA_type in {'learnable_s', 'learnable_c'}:
                    sim_matrix = sim_matrix[self.base_way:, :self.base_way]             # (n_way_novel, n_way_base)
                    vals, indxs = torch.topk(sim_matrix, self.args.EMA_top_k, dim=1)    # (n_way_novel, n_top_k)
                    self.sim_n2b = torch.mean(vals, 1).cuda()                           # (n_way_novel, )
                    self.EMA_FC_K = nn.Parameter(torch.FloatTensor(1).fill_(self.args.EMA_FC_K).cuda(), requires_grad=True)
                    self.EMA_FC_b = nn.Parameter(torch.FloatTensor(1).fill_(self.args.EMA_FC_b).cuda(), requires_grad=True)
                    # print(self.sim_n2b)

            # save intermediate models at the end of each finetune session
            if hasattr(self.args, 'vis_logits') and self.args.vis_logits:
                self.feature_list = []
                self.weights_list = []
                # base class pre-trained backbone (t=0)
                feature_tmp = resnet18(False, self.args).cuda()
                feature_tmp.load_state_dict(model_trained.feature.state_dict(), strict=True)
                self.feature_list.append(feature_tmp)
                # base class pre-trained weights (t=0)
                weights_tmp = model_trained.joint_weight[:self.base_way].clone().data
                self.weights_list.append(weights_tmp)
                # set grad to False
                for i in range(len(self.feature_list)):
                    for p in self.feature_list[i].parameters():
                        p.requires_grad = False
                    self.weights_list[i].requires_grad = False
        
    
    def forward(self, **kwargs):
        flag = kwargs['flag']
        if flag == 'embedding':         # return feature embeddings
            return self.feature(kwargs['input'])
        elif flag == 'base_forward':    # return base class logits
            return self.base_forward(kwargs['input'])
        elif flag == 'inc_forward':     # return base+novel class logits and generated weights
            return self.inc_forward(kwargs['z_support'], kwargs['z_query'], kwargs['y_b'])
        elif flag == 'set_forward':     # return novel class logits and generated weights
            return self.set_forward(kwargs['z_support'], kwargs['z_query'])
        else:
            Exception('Undefined forward type') 


    def base_forward(self, x):
        '''
        base forward, used in 1st standard supervised training
        '''
        f = self.feature.forward(x)                                     # (B, 3, H, W) -> (B, feat_dim)
        f = F.normalize(f, p=2, dim=1, eps=1e-12)                       # (B, feat_dim)
        weight = F.normalize(self.base_weight, p=2, dim=1, eps=1e-12)   # (C, feat_dim)
        pred = self.scale_cls * torch.mm(f, weight.t())                 # (B, C)
        return pred

    
    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader
    

    def save_all_samples_novel(self, trainset, session=1):
        '''
        save 5 exemplars per novel class
        '''
        assert session > 0
        if session == 1:
            samples_path, samples_label = trainset.data, trainset.targets
            if self.dataset == 'cifar100':
                samples_path = [x for x in samples_path]
                samples_label = samples_label.tolist()
            self.novel_exemplar_set = ExemplarSet(samples_path, samples_label, dataset=self.args.dataset, train=True)
        else:
            samples_path, samples_label = trainset.data, trainset.targets
            if self.dataset == 'cifar100':
                samples_path = [x for x in samples_path]
                samples_label = samples_label.tolist()
            samples_path  = self.novel_exemplar_set.data + samples_path
            samples_label = self.novel_exemplar_set.targets + samples_label
            self.novel_exemplar_set.update(samples_path, samples_label)
    

    def save_all_samples_base(self, trainset, backbone, weight, session=0, transform_test=None):
        '''
        save 5 exemplars per base class  
        '''
        assert session == 0
        self.class_max_sim  = [-1] * self.args.base_class
        self.exemplar_idxs  = [-1] * self.args.base_class
        self.exemplar_path  = ['None'] * self.args.base_class
        self.exemplar_label = list(range(len(self.exemplar_path)))
        batch_size_tmp = 256
        
        class_tmp = list(get_session_classes(self.args, session))
        trainset.return_idx = True
        if transform_test is not None:
            trainset.transform = transform_test
        trainloader = DataLoader(dataset=trainset, batch_size=batch_size_tmp, shuffle=False,
                                 num_workers=8, pin_memory=True)
        backbone.eval()
        with torch.no_grad():
            for batch in trainloader:
                data, labels, idxs = [_.cuda() for _ in batch]      # (bs, 3, H, W), (bs, ), (bs, )
                pred = self.get_logits(backbone, weight, 1, data)   # (bs, n_way_tmp)
                for c in class_tmp:
                    pred_c = pred[labels == c]                      # (bs_c, n_way_tmp)
                    idxs_c = idxs[labels == c]                      # (bs_c, )
                    if pred_c.shape[0] > 0:
                        pred_c = pred_c[:, c]                       # (bs_c, )
                        sim_c, idx_c = torch.max(pred_c, dim=0)
                        if sim_c > self.class_max_sim[c]:
                            self.class_max_sim[c] = sim_c.item() 
                            self.exemplar_idxs[c] = idxs_c[idx_c] 
                            self.exemplar_path[c] = trainset.data[idxs_c[idx_c]] 
        trainset.return_idx = False

        # sample the remaining exemplars
        labels_all = deepcopy(trainset.targets)
        if self.dataset != 'cifar100': 
            labels_all = np.array(labels_all)
        idxs_all = np.arange(len(labels_all))
        n_exemplar_per_class = 5
        self.exemplar_path_all  = [] 
        self.exemplar_label_all = np.repeat(np.array(self.exemplar_label), n_exemplar_per_class)
        self.exemplar_label_all = self.exemplar_label_all.tolist() 
        for c in class_tmp:
            self.exemplar_path_all.append(self.exemplar_path[c])
            idxs_all_c = list(idxs_all[labels_all == c])
            assert self.exemplar_idxs[c] in idxs_all_c
            idxs_all_c.remove(self.exemplar_idxs[c])
            idxs_sampled_c = random.sample(idxs_all_c, n_exemplar_per_class - 1)
            for idx_tmp in idxs_sampled_c:
                self.exemplar_path_all.append(trainset.data[idx_tmp])
        assert len(self.exemplar_path_all) == len(self.exemplar_label_all)
            
        self.base_exemplar_set  = ExemplarSet(deepcopy(self.exemplar_path_all), deepcopy(self.exemplar_label_all), dataset=self.args.dataset, train=True)
    

    def save_exemplars(self, trainset, backbone, weight, session=0, transform_test=None):
        '''
        only ONE exemplar per class will be saved
        Note: Temporarily, we only store one exemplar for each base class   
        '''
        if session == 0:
            self.class_max_sim  = [-1] * self.args.base_class
            self.exemplar_path  = ['None'] * self.args.base_class
            self.exemplar_label = list(range(len(self.exemplar_path)))
            batch_size_tmp = 256
        else:
            self.class_max_sim = self.class_max_sim + [-1] * self.args.n_way
            self.exemplar_path = self.exemplar_path + ['None'] * self.args.n_way
            self.exemplar_label = list(range(len(self.exemplar_path)))
            batch_size_tmp = trainset.__len__()
        
        class_tmp = list(get_session_classes(self.args, session))
        trainset.return_idx = True
        if transform_test is not None:
            trainset.transform = transform_test
        trainloader = DataLoader(dataset=trainset, batch_size=batch_size_tmp, shuffle=False,
                                 num_workers=8, pin_memory=True)
        backbone.eval()
        with torch.no_grad():
            for batch in trainloader:
                data, labels, idxs = [_.cuda() for _ in batch]      # (bs, 3, H, W), (bs, ), (bs, )
                pred = self.get_logits(backbone, weight, 1, data)   # (bs, n_way_tmp)
                for c in class_tmp:
                    pred_c = pred[labels == c]                      # (bs_c, n_way_tmp)
                    idxs_c = idxs[labels == c]                      # (bs_c, )
                    if pred_c.shape[0] > 0:
                        pred_c = pred_c[:, c]                       # (bs_c, )
                        sim_c, idx_c = torch.max(pred_c, dim=0)
                        if sim_c > self.class_max_sim[c]:
                            self.class_max_sim[c] = sim_c.item()  
                            self.exemplar_path[c] = trainset.data[idxs_c[idx_c]] 
        trainset.return_idx = False

        if session == 0:
            self.base_exemplar_set  = ExemplarSet(deepcopy(self.exemplar_path), deepcopy(self.exemplar_label), dataset=self.args.dataset, train=True)
        elif session == 1:
            self.novel_exemplar_set = ExemplarSet(deepcopy(self.exemplar_path[self.base_way:]), deepcopy(self.exemplar_label[self.base_way:]), 
                                                  dataset=self.args.dataset, train=True)
        else:
            self.novel_exemplar_set.update(deepcopy(self.exemplar_path[self.base_way:]), deepcopy(self.exemplar_label[self.base_way:]))

    
    def test_inc_loop(self, epoch=0, eval_base_only=False):
        '''
        the main function of testing on all incremental sessions
        Loop: {get dataset at session t -> imprint/init new weights -> 
        (finetune at session t) -> (save exemplars at session t) ->
        evaluate on session t} -> summarize
        '''
        acc_session_list = []
        self.acc_session_list_single = []
        for session in range(self.args.n_sessions):
            # get dataloaders
            train_set, trainloader, testloader = self.get_dataloader(session)
            transform_train = trainloader.dataset.transform
            transform_test  = testloader.dataset.transform

            # init the model
            trainloader.dataset.transform = transform_test                                      # Training augmentation is not used in Imprint
            if session == 0:
                if self.base_mode == 'avg_cos':
                    self.joint_weight = self.get_class_mean(self.feature, trainloader, train_set, session)
                else:
                    self.joint_weight = self.base_weight.data                                   # (base_way, C)
            else:
                new_fc = self.get_class_mean(self.feature, trainloader, train_set, session)     # (n_way, C)
                self.joint_weight = torch.cat([self.joint_weight, new_fc.data], dim=0)          # init current weights
                
                # extend teacher model's weights
                if (self.needs_finetune == True):
                    new_fc_prev = self.get_class_mean(self.feature_prev, trainloader, train_set, session) 
                    self.weights_prev = torch.cat([self.weights_prev, new_fc_prev.data], dim=0)
                    self.weights_prev.requires_grad = False

                # extend EMA parameters model's weights
                if (self.needs_finetune == True):
                    new_fc_ema = self.get_class_mean(self.feature_ema, trainloader, train_set, session) 
                    self.weights_ema = torch.cat([self.weights_ema, new_fc_ema.data], dim=0)
                    self.weights_ema.requires_grad = False
                
                # extend all the recorded models' weights
                if (self.needs_finetune == True) and hasattr(self.args, 'vis_logits') and self.args.vis_logits:
                    for i in range(len(self.feature_list)):
                        new_fc_t = self.get_class_mean(self.feature_list[i], trainloader, train_set, session)
                        self.weights_list[i] = torch.cat([self.weights_list[i], new_fc_t.data], dim=0)
                        self.weights_list[i].requires_grad = False
            
            # finetune the model
            loss_records = None
            if (self.needs_finetune == True) and (session != 0):
                trainloader.dataset.transform = transform_train
                loss_records = self.update_model_ft(trainloader, session, train_set, transform_test) 
            
            # save exemplars (can be placed before finetuning)
            if (self.needs_finetune == True) and (self.args.using_exemplars == True):
                # Note: Temporarily, we only store one exemplar for each base class  
                if self.args.save_all_data_base and (session == 0): 
                    self.save_all_samples_base(train_set, self.feature, self.joint_weight, session, transform_test)
                elif self.args.save_all_data_novel and (session > 0): 
                    self.save_all_samples_novel(train_set, session)
                else:
                    self.save_exemplars(train_set, self.feature, self.joint_weight, session, transform_test)
            
            # evaluate on this session (main branch)
            report_acc_tmp = False if (session > 0) and self.needs_finetune and self.args.bilateral else True
            if (not hasattr(self.args, 'main_branch')) or self.args.main_branch == 'current':
                acc_tmp, scores, feats, labels_y = self.test_one_session(testloader, self.feature, self.joint_weight, self.args, session, 
                                                                         loss_records, report_acc=report_acc_tmp, report_binary=self.args.report_binary, main_branch=True)
            elif self.args.main_branch == 'ema':
                acc_tmp, scores, feats, labels_y = self.test_one_session(testloader, self.feature_ema, self.weights_ema[:self.joint_weight.shape[0]], self.args, session, 
                                                                         loss_records, report_acc=report_acc_tmp, report_binary=self.args.report_binary, main_branch=True)
            else:
                Exception('Undefined main branch type')
                
            if self.needs_finetune and self.args.bilateral:
                self.acc_session_list_single.append(acc_tmp)
            
            # evaluate on this session (bilateral)
            if (session > 0) and self.needs_finetune and self.args.bilateral:
                if self.args.second_branch == 'fixed':
                    acc_org, scores_org, feats_org, _ = self.test_one_session(testloader, self.feature_fixed, self.weights_fixed[:self.joint_weight.shape[0]], 
                                                                              self.args, session, loss_records=None, report_acc=False, report_binary=False, main_branch=False)
                elif self.args.second_branch == 'ema':
                    acc_org, scores_org, feats_org, _ = self.test_one_session(testloader, self.feature_ema, self.weights_ema[:self.joint_weight.shape[0]], 
                                                                              self.args, session, loss_records=None, report_acc=False, report_binary=False, main_branch=False)
                else:
                    Exception('Undefined second branch type')
                acc_tmp, scores_new, _ = self.test_one_session_bilateral(testloader, scores, scores_org, feats, feats_org, labels_y, self.args, session)
            
            # record results on this session
            acc_session_list.append(acc_tmp)
            if eval_base_only: return acc_session_list
        
        # summarize all the evaluation results
        if self.needs_finetune:
            str_out = 'Epoch=%d, (Finetune) Evaluation on all sessions: ['%(epoch)
        else:
            str_out = 'Epoch=%d, Evaluation on all sessions: ['%(epoch)
        for i, acc_tmp in enumerate(acc_session_list):
            str_out += '%.2f%%'%(100*acc_tmp)
            if i < len(acc_session_list) - 1: str_out += ' '
        str_out += ']'
        print(str_out)
        logging.info(str_out)
   
        if self.needs_finetune and self.args.using_exemplars and self.args.vis_exemplars:
            imgs_all = []
            for exe_set in [self.base_exemplar_set, self.novel_exemplar_set]:
                for img in exe_set.data:
                    if self.dataset == 'cifar100':
                        img = Image.fromarray(img)
                    else:
                        img = Image.open(img).convert('RGB')
                        img = transforms.Resize([84, 84])(img)
                    img = transforms.ToTensor()(img)
                    imgs_all.append(img)
            imgs_all = torch.stack(imgs_all, dim=0)
            print(imgs_all.shape)
            save_image(imgs_all, os.path.join('tmp', '%s.jpg'%(self.dataset)), nrow=self.args.vis_exemplars_nrow, padding=2)

        # save output logits
        if (self.needs_finetune == True) and hasattr(self.args, 'vis_logits') and self.args.vis_logits:
            BE_set = ExemplarSet(self.base_exemplar_set.data, self.base_exemplar_set.targets, dataset=self.args.dataset, train=False)
            BE_loader = torch.utils.data.DataLoader(dataset=BE_set, batch_size=BE_set.__len__(), shuffle=False, num_workers=0, pin_memory=True)
            NE_set = ExemplarSet(self.novel_exemplar_set.data, self.novel_exemplar_set.targets, dataset=self.args.dataset, train=False)
            NE_loader = torch.utils.data.DataLoader(dataset=NE_set, batch_size=NE_set.__len__(), shuffle=False, num_workers=0, pin_memory=True)
            X1, Y1 = next(iter(BE_loader))
            X2, Y2 = next(iter(NE_loader))
            X = torch.cat([X1, X2], dim=0).cuda()
            Y = torch.cat([Y1, Y2], dim=0).cuda()
            print(X.shape, Y.shape, Y)
            logits_list = []
            with torch.no_grad():
                for i in range(len(self.feature_list)):
                    model_tmp = self.feature_list[i]
                    model_tmp.eval()
                    weights_tmp = self.weights_list[i]
                    logits_tmp = self.get_logits(model_tmp, weights_tmp, self.scale_cls.data, X)
                    logits_tmp = torch.softmax(logits_tmp/self.args.ft_T, dim=1)
                    logits_list.append(logits_tmp)
                logits_list = torch.stack(logits_list, dim=0).cpu().numpy()
                Y = Y.cpu().numpy()
                print(logits_list.shape)
                np.savez('tmp/%s_%s.npz'%(self.args.logits_tag, self.dataset), logits=logits_list, labels=Y)

        # if self.needs_finetune and self.args.using_exemplars:
        #     print(len(self.base_exemplar_set.data), len(self.base_exemplar_set.targets))
        #     print(len(self.novel_exemplar_set.data), len(self.novel_exemplar_set.targets))
        
        return acc_session_list
    

    def test_one_session(self, testloader, backbone, weight, args, session, loss_records=None, report_acc=True, report_binary=False, main_branch=False):
        '''
        evalute on the current session t based on the given feature extractor and classification weights
        '''
        tqdm_gen = tqdm.tqdm(testloader)
        backbone.eval()
        y = []
        feats = []
        scores = []
        assert (args.base_class + session * args.n_way) == weight.shape[0]
        with torch.no_grad():
            for i, batch in enumerate(tqdm_gen):
                data, test_label = [_.cuda() for _ in batch]
                pred, feat_tmp = self.get_logits(backbone, weight, self.scale_cls.data, data, return_feature=True)
                scores.append(pred)  
                y.append(test_label) 
                feats.append(feat_tmp)         
        y = torch.cat(y, dim=0)
        feats = torch.cat(feats, dim=0)
        scores = torch.cat(scores, dim=0)
        
        acc = top1_acc(scores, y)

        if report_acc == True:
            str_out = '' if loss_records is None else '(CE=%.3f KD=%.3f EX=%.3f) '%(loss_records[0].avg, loss_records[1].avg, loss_records[2].avg)
            str_out += 'Session %d testing accuracy = %.2f%%'%(session, 100*acc)
            print(str_out)
            logging.info(str_out) 
        
        return acc, scores, feats, y
    

    def test_one_session_bilateral(self, testloader, scores, scores_org, feats, feats_org, y, args, session):
        '''
        evalute on the current session t by merging two branches
        '''
        assert session > 0
        scores_tmp = []
        self.feature.eval()
        self.feature_ema.eval()
        self.feature_fixed.eval()
        
        if self.args.merge_strategy == 'routing':
            # select which branch to forward
            if self.args.branch_selector == 'oracle':
                pred_mask_base = (y < self.base_way)
            elif self.args.branch_selector == 'logits_org':
                _, pred_tmp = torch.max(scores_org, dim=1)
                pred_mask_base = (pred_tmp < self.base_way)
            elif self.args.branch_selector == 'logits_current':
                _, pred_tmp = torch.max(scores, dim=1)
                pred_mask_base = (pred_tmp < self.base_way)
            else:
                Exception('Undefined routing strategy')
            # ensemble two branches
            if self.args.masking_novel:
                scores_org[:, self.base_way:] = 0.1 * scores_org[:, self.base_way:]
            scores[pred_mask_base] = scores_org[pred_mask_base]
        
        elif self.args.merge_strategy == 'avg':
            # ensemble two branches using weighted average
            scores = self.args.branch_weights * scores + (1 - self.args.branch_weights) * scores_org

        elif 'attn' in self.args.merge_strategy:
            self.binary_classifier.eval()
            if self.args.merge_strategy == 'attn_s':
                self.fc_b.eval()
                self.fc_n.eval()
            with torch.no_grad():
                pred_b = torch.softmax(scores_org.clone(), dim=-1)                  # (bs, C)
                pred_n = torch.softmax(scores.clone(), dim=-1)                      # (bs, C)
                if self.args.BC_flatten == 'org':
                    fc_input = torch.cat([pred_b, pred_n], dim=-1)                  # (bs, 2C)
                elif self.args.BC_flatten == 'sum':
                    pred_b_1 = torch.sum(pred_b[:, :self.base_way], dim=1, keepdim=True)
                    pred_b_2 = torch.sum(pred_b[:, self.base_way:], dim=1, keepdim=True)
                    pred_n_1 = torch.sum(pred_n[:, :self.base_way], dim=1, keepdim=True)
                    pred_n_2 = torch.sum(pred_n[:, self.base_way:], dim=1, keepdim=True)
                    fc_input = torch.cat([pred_b_1, pred_b_2, pred_n_1, pred_n_2], dim=-1) # (bs, 4)
                elif self.args.BC_flatten == 'max':
                    dim_tmp = (self.joint_weight.shape[0] - self.base_way)
                    pred_b_1 = torch.topk(pred_b[:, :self.base_way], dim_tmp, dim=1)[0]
                    pred_b_2 = torch.topk(pred_b[:, self.base_way:], dim_tmp, dim=1)[0]
                    pred_n_1 = torch.topk(pred_n[:, :self.base_way], dim_tmp, dim=1)[0]
                    pred_n_2 = torch.topk(pred_n[:, self.base_way:], dim_tmp, dim=1)[0]
                    fc_input = torch.cat([pred_b_1, pred_b_2, pred_n_1, pred_n_2], dim=-1) # (bs, 4)
                else:
                    Exception()
                attn_logits  = self.binary_classifier(fc_input)                     # (bs, 2)
                attn_weights = torch.softmax(attn_logits, dim=-1)                   # (bs, 2)
                if self.args.merge_strategy == 'attn':
                    scores_b = scores_org.clone()
                    scores_n = scores.clone()
                    if self.args.masking_novel:
                        scores_b[:, self.base_way:] = 0.1 * scores_b[:, self.base_way:] # base branch only handles base class samples
                    scores = attn_weights[:, 0:1] * scores_b + attn_weights[:, 1:2] * scores_n                 
                elif self.args.merge_strategy == 'attn_f':
                    feat_b = feats_org.clone() 
                    feat_n = feats.clone()
                    if self.norm_first:
                        feat_b = F.normalize(feat_b, p=2, dim=1, eps=1e-12)
                        feat_n = F.normalize(feat_n, p=2, dim=1, eps=1e-12)
                    feat_attn = attn_weights[:, 0:1] * feat_b + attn_weights[:, 1:2] * feat_n
                    feat_attn = F.normalize(feat_attn, p=2, dim=1, eps=1e-12)
                    w_attn = F.normalize(self.ft_weight_attn, p=2, dim=1, eps=1e-12)                
                    scores = self.scale_cls.data * torch.mm(feat_attn, w_attn.t())  
                elif self.args.merge_strategy == 'attn_s':
                    scores_b = scores_org.clone()
                    scores_n = scores.clone()
                    scalar_b = self.fc_b(pred_b)
                    scalar_n = self.fc_n(pred_n)
                    attn_logits = torch.cat([scalar_b, scalar_n], dim=1)
                    attn_weights = torch.softmax(attn_logits, dim=-1)
                    if self.args.masking_novel:
                        scores_b[:, self.base_way:] = 0.1 * scores_b[:, self.base_way:] # base branch only handles base class samples
                    scores = attn_weights[:, 0:1] * scores_b + attn_weights[:, 1:2] * scores_n
                else:
                    Exception() 
                if session == (self.args.n_sessions - 1):
                    self.binary_coeffs = attn_weights 
                    assert (self.binary_coeffs.shape[0] == scores.shape[0]) and (self.binary_coeffs.shape[1] == 2)       
        else:
            Exception('Undefined bilateral branch strategy')
        
        acc = top1_acc(scores, y)

        str_out = '( Bilateral Branch Results ) Session %d testing accuracy = %.2f%%'%(session, 100*acc)
        print(str_out)
        logging.info(str_out)  
        return acc, scores, y

    
    def get_binary_result(self, scores, y, session, n_base):
        mask_base  = (y <  n_base)
        mask_novel = (y >= n_base)
        pred_b = scores[mask_base]              # (N_b, n_way_all)
        pred_n = scores[mask_novel]             # (N_n, n_way_all)
        # joint accuracy
        _, pred_a = torch.max(scores, dim=1)    # (N, )
        pred_a = (pred_a < n_base)
        acc_joint = (pred_a == mask_base).float().mean().item()
        # base class samples            
        _, pred_b = torch.max(pred_b, dim=1)    # (N_b, )
        pred_b2b = (pred_b <  n_base).float()
        pred_b2n = (pred_b >= n_base).float()
        acc_b2b, N_b2b, N_b2n = pred_b2b.mean().item(), pred_b2b.sum().item(), pred_b2n.sum().item()
        # novel class samples
        _, pred_n = torch.max(pred_n, dim=1)    # (N_n, )
        pred_n2n = (pred_n >= n_base).float()
        pred_n2b = (pred_n <  n_base).float()
        acc_n2n, N_n2n, N_n2b = pred_n2n.mean().item(), pred_n2n.sum().item(), pred_n2b.sum().item()
        str_out = '[Binary CLS Results: %.2f%%] Session %d acc:b2b=%.2f%% num: b2b=%d b2n=%d; acc: n2n=%.2f%%, num: n2n=%d, n2b=%d'%(100*acc_joint, session, 
                                100*acc_b2b, N_b2b, N_b2n, 100*acc_n2n, N_n2n, N_n2b)
        return str_out


    def get_binary_result_v2(self, scores, y, session, n_base):
        pred_a = (scores[:, 0] >= scores[:, 1]).long()
        mask_base  = (y <  n_base)
        mask_novel = (y >= n_base)
        pred_b = pred_a[mask_base]              # (N_b, 2)
        pred_n = pred_a[mask_novel]             # (N_n, 2)
        # joint accuracy
        acc_joint = (pred_a == mask_base).float().mean().item()
        # base class samples            
        pred_b2b = (pred_b == 1).float()
        pred_b2n = (pred_b == 0).float()
        acc_b2b, N_b2b, N_b2n = pred_b2b.mean().item(), pred_b2b.sum().item(), pred_b2n.sum().item()
        # novel class samples
        pred_n2n = (pred_n == 0).float()
        pred_n2b = (pred_n == 1).float()
        acc_n2n, N_n2n, N_n2b = pred_n2n.mean().item(), pred_n2n.sum().item(), pred_n2b.sum().item()
        str_out = '[Binary New Results: %.2f%%] Session %d acc:b2b=%.2f%% num: b2b=%d b2n=%d; acc: n2n=%.2f%%, num: n2n=%d, n2b=%d'%(100*acc_joint, session, 
                                100*acc_b2b, N_b2b, N_b2n, 100*acc_n2n, N_n2n, N_n2b)
        return str_out
    
    
    def get_detailed_result(self, scores, y, session, str_out=''):
        assert session > 0
        c2a_list = []
        c2c_list = []
        for t in range(session + 1):
            if t == 0:
                c_start, c_end = 0, self.base_way - 1
            else:
                c_start = self.base_way + (t - 1) * self.args.n_way
                c_end   = self.base_way + t * self.args.n_way - 1
            idx_tmp  = (y >= c_start) & (y <= c_end)      
            y_all    = y[idx_tmp]                       # (n_test, )
            y_tmp    = y_all - c_start                  # (n_test, )
            pred_all = scores[idx_tmp]                  # (n_test, n_way_all)
            pred_tmp = pred_all[:, c_start: c_end+1]    # (n_test, n_way)
            c2a = top1_acc(pred_all, y_all)
            c2c = top1_acc(pred_tmp, y_tmp)
            c2a_list.append(c2a)
            c2c_list.append(c2c)
            str_out += ' |%.2f%%, %.2f%%|'%(100*c2c, 100*c2a)
        return str_out, c2a_list, c2c_list


    def get_class_mean(self, backbone, trainloader, train_set, session):
        '''
        get the ordered class prototypes from trainloader based on the given feature extractor
        '''
        backbone.eval()
        data = []
        label = []
        class_list = np.unique(train_set.targets).tolist()
        class_list.sort()
        # if session > 0: assert len(trainloader) == 1 and len(class_list) == self.args.n_way
        with torch.no_grad():
            for batch in trainloader:
                data_tmp, label_tmp = [_.cuda() for _ in batch]
                data_tmp = backbone.forward(data_tmp)       
                data.append(data_tmp)
                label.append(label_tmp)
            data = torch.cat(data, dim=0)
            label = torch.cat(label, dim=0)
            new_fc = []
            for class_index in class_list:
                data_index = (label == class_index).nonzero().squeeze(-1)
                embedding = data[data_index]                    
                if self.norm_first:
                    embedding = F.normalize(embedding, p=2, dim=1, eps=1e-12)
                proto = embedding.mean(0)                   
                new_fc.append(proto)
            new_fc = torch.stack(new_fc, dim=0)
        return new_fc
    

    def get_class_mean_v2(self, backbone, data, label, session):
        '''
        get the ordered class prototypes from data (loaded images) based on the given feature extractor
        '''
        backbone.eval()
        class_list = np.unique(label.cpu().numpy()).tolist()
        class_list.sort()
        # if session > 0: assert len(trainloader) == 1 and len(class_list) == self.args.n_way
        with torch.no_grad():
            data = backbone.forward(data)       
            new_fc = []
            for class_index in class_list:
                data_index = (label == class_index).nonzero().squeeze(-1)
                embedding = data[data_index]                    
                if self.norm_first:
                    embedding = F.normalize(embedding, p=2, dim=1, eps=1e-12)
                proto = embedding.mean(0)                   
                new_fc.append(proto)
            new_fc = torch.stack(new_fc, dim=0)
        return new_fc
    

    def get_logits(self, backbone, weight, scalar, x, return_feature=False):
        '''
        get the output logits based on the given feature extractor, classification weights and scalar factor
        '''
        f0 = backbone.forward(x)                                                # (B, 3, H, W)
        f = F.normalize(f0, p=2, dim=1, eps=1e-12)                              # (B, C)
        weight = F.normalize(weight, p=2, dim=1, eps=1e-12)                     # (classes_seen, C)
        pred = scalar * torch.mm(f, weight.t())                                 # (B, classes_seen)
        if return_feature:
            return pred, f0
        else:
            return pred


    def update_model_ft(self, trainloader, session, trainset=None, transform_test=None):
        '''
        update the feature extractor and the classification weights based on finetuning
        '''
        # define teacher/stduent feature and weights
        assert session >= 1
        EMA_prob = False
        tmp_all_way = self.joint_weight.shape[0]
        tmp_base_way = tmp_all_way - self.args.n_way
        if hasattr(self.args, 'ft_reinit') and self.args.ft_reinit:
            self.feature.load_state_dict(self.feature_fixed.state_dict(), strict=True) 
            self.joint_weight = self.weights_fixed[:tmp_all_way].clone().data
        self.feature_ema.eval()
        self.feature_prev.eval()
        self.feature_fixed.eval()
        ft_weight = nn.Parameter(self.joint_weight.clone().data, requires_grad=True)    # (tmp_all_way, C)

        if hasattr(self.args, 'imprint_ft_weight') and self.args.imprint_ft_weight:
            assert self.args.using_exemplars
            # get all the available images
            imgs_all, labels_all = [], []
            # base class training data
            self.base_exemplar_set.train = False
            loader_exe_0 = DataLoader(dataset=self.base_exemplar_set, batch_size=self.base_exemplar_set.__len__(), 
                                      shuffle=False, num_workers=8, pin_memory=True)
            imgs_exe_0, labels_exe_0 = next(iter(loader_exe_0))
            imgs_all.append(imgs_exe_0)
            labels_all.append(labels_exe_0)
            self.base_exemplar_set.train = True
            # previous novel classs training data
            if session > 1:
                self.novel_exemplar_set.train = False
                loader_exe_1 = DataLoader(dataset=self.novel_exemplar_set, batch_size=self.novel_exemplar_set.__len__(), 
                                          shuffle=False, num_workers=8, pin_memory=True)
                imgs_exe_1, labels_exe_1 = next(iter(loader_exe_1))
                imgs_all.append(imgs_exe_1)
                labels_all.append(labels_exe_1)
                self.novel_exemplar_set.train = True
            # novel class training data at the current session
            transform_org = trainset.transform
            trainset.transform = transform_test
            loader_trn = DataLoader(dataset=trainset, batch_size=trainset.__len__(),  
                                    shuffle=False, num_workers=8, pin_memory=True)
            imgs_trn, labels_trn = next(iter(loader_trn))
            imgs_all.append(imgs_trn)
            labels_all.append(labels_trn)
            trainset.transform = transform_org
            imgs_all = torch.cat(imgs_all, dim=0).cuda()
            labels_all = torch.cat(labels_all, dim=0).cuda()
            # imprint novel weights using all the available images
            imprinted_weights = self.get_class_mean_v2(self.feature, imgs_all, labels_all, session)
            ft_weight = nn.Parameter(imprinted_weights.clone().data, requires_grad=False)
            assert ft_weight.shape == self.joint_weight.shape
        
        # define parameters to be optimized
        if self.args.part_frozen == False:
            self.feature.train()
            for p in self.feature.parameters():
                p.requires_grad = True
            set_bn_state(self.feature, set_eval=self.args.bn_eval)
            params_list = [{'params': self.feature.parameters(), 'lr': self.args.ft_factor * self.args.ft_lr},
                           {'params': ft_weight}]
        else:
            # freeze all previous layers ...
            self.feature.eval()
            for p in self.feature.parameters():
                p.requires_grad = False
            # .. except the last residual block
            self.feature.layer4.train()
            for p in self.feature.layer4.parameters():
                p.requires_grad = True
            set_bn_state(self.feature.layer4, set_eval=self.args.bn_eval)
            params_list = [{'params': self.feature.layer4.parameters(), 'lr': self.args.ft_factor * self.args.ft_lr},
                           {'params': ft_weight}]
        
        if (session > 1) and hasattr(self.args, 'EMA_logits') and self.args.EMA_logits and self.args.EMA_type == 'learnable':
            EMA_scalar = torch.FloatTensor(1).fill_(self.args.EMA_scalar).cuda()
            EMA_scalar = nn.Parameter(EMA_scalar.clone().data, requires_grad=True)
            params_list.append({'params': EMA_scalar, 'lr': self.args.EMA_scalar_lr})
        
        if (session > 1) and hasattr(self.args, 'EMA_logits') and self.args.EMA_logits and self.args.EMA_type == 'learnable_v':
            EMA_vector = torch.ones(session - 1, 1).float().cuda() * self.args.EMA_scalar # session=2 -> 1, session=3 -> 2 ... (or can be class-dim)
            EMA_vector = nn.Parameter(EMA_vector.clone().data, requires_grad=True)
            params_list.append({'params': EMA_vector, 'lr': self.args.EMA_scalar_lr})
        
        if (session > 1) and hasattr(self.args, 'EMA_logits') and self.args.EMA_logits and self.args.EMA_type == 'learnable_mlp_c':
            self.vector_table = self.sim_n2all[:, :tmp_base_way]                                            # (n_way_novel, all_classes_t-1)
            self.coeff_mlp = BinaryNet(self.vector_table.shape[-1], self.args.EMA_FC_dim).cuda()            
            params_list.append({'params': self.coeff_mlp.parameters(), 'lr': self.args.EMA_FC_lr})
        
        if (session > 1) and hasattr(self.args, 'EMA_logits') and self.args.EMA_logits and self.args.EMA_type == 'learnable_mlp_b':
            self.vector_table = self.sim_n2base                                                             # (n_way_novel, base_way)
            self.coeff_mlp = BinaryNet(self.vector_table.shape[-1], self.args.EMA_FC_dim).cuda()            
            params_list.append({'params': self.coeff_mlp.parameters(), 'lr': self.args.EMA_FC_lr})
        
        if (session > 1) and hasattr(self.args, 'EMA_logits') and self.args.EMA_logits and self.args.EMA_type == 'learnable_mlp_v':
            # base part
            vector_table_base, _ = torch.topk(self.sim_n2base, 1, dim=-1)                                   # (n_way_novel, 1)
            vector_table_base  = vector_table_base.reshape(self.args.n_sessions - 1, self.args.n_way, 1)    # (n_session, n_way, 1)
            vector_table_base  = vector_table_base.mean(1)                                                  # (n_session, 1)
            # novel part
            vector_table_novel = self.sim_n2novel[:, :(tmp_base_way - self.base_way)]                       # (n_way_novel, novel_classes_t-1)
            vector_table_novel = vector_table_novel.reshape(self.args.n_sessions - 1, 
                                                            self.args.n_way, 
                                                            vector_table_novel.shape[-1])                   # (n_session, n_way, novel_classes_t-1)
            vector_table_novel = vector_table_novel.reshape(vector_table_novel.shape[0], 
                                                            vector_table_novel.shape[1], 
                                                            session - 1, 
                                                            self.args.n_way)                                # (n_session, n_way, t-1, n_way)
            vector_table_novel, _ = torch.topk(vector_table_novel, 1, dim=-1)                               # (n_session, n_way, t-1, 1)
            vector_table_novel = vector_table_novel.squeeze(-1)                                             # (n_session, n_way, t-1)
            vector_table_novel = vector_table_novel.mean(1)                                                 # (n_session, t-1)
            self.vector_table = torch.cat([vector_table_base, vector_table_novel], dim=1)                   # (n_session, t)
            self.coeff_mlp = BinaryNet(self.vector_table.shape[-1], self.args.EMA_FC_dim).cuda()            
            params_list.append({'params': self.coeff_mlp.parameters(), 'lr': self.args.EMA_FC_lr})
        
        if (session > 1) and hasattr(self.args, 'EMA_logits') and self.args.EMA_logits and self.args.EMA_type == 'learnable_c':
            EMA_vector = torch.ones(tmp_base_way - self.base_way, 1).float().cuda() * self.args.EMA_scalar 
            EMA_vector = nn.Parameter(EMA_vector.clone().data, requires_grad=True)
            params_list.append({'params': EMA_vector, 'lr': self.args.EMA_scalar_lr})
        
        if (session > 1) and hasattr(self.args, 'EMA_logits') and self.args.EMA_logits and self.args.EMA_type == 'learnable_s':
            EMA_vector = torch.ones(session - 1, 1).float().cuda() * self.args.EMA_scalar 
            EMA_vector = nn.Parameter(EMA_vector.clone().data, requires_grad=True)
            params_list.append({'params': EMA_vector, 'lr': self.args.EMA_scalar_lr})
            if self.args.EMA_reinit:
                self.EMA_FC_K = nn.Parameter(torch.FloatTensor(1).fill_(self.args.EMA_FC_K).cuda(), requires_grad=True)
                self.EMA_FC_b = nn.Parameter(torch.FloatTensor(1).fill_(self.args.EMA_FC_b).cuda(), requires_grad=True)
            params_list.append({'params': self.EMA_FC_K, 'lr': self.args.EMA_FC_lr})
            params_list.append({'params': self.EMA_FC_b, 'lr': self.args.EMA_FC_lr})
        
        if 'attn' in self.args.merge_strategy:
            if self.args.BC_flatten == 'org':
                self.binary_classifier = BinaryNet(2*tmp_all_way, self.args.BC_hidden_dim).cuda()
            elif self.args.BC_flatten == 'sum':
                self.binary_classifier = BinaryNet(4, self.args.BC_hidden_dim).cuda()
            elif self.args.BC_flatten == 'max':
                self.binary_classifier = BinaryNet(4*(tmp_all_way - self.base_way), self.args.BC_hidden_dim).cuda()
            else:
                Exception()
            self.binary_classifier.train()
            params_list.append({'params': self.binary_classifier.parameters(), 'lr': self.args.BC_lr})
        
        if self.args.merge_strategy == 'attn_f':
            w1 = self.weights_fixed[:tmp_all_way].clone().data
            w2 = self.joint_weight.clone().data
            assert w1.shape == w2.shape
            if self.norm_first:
                w1 = F.normalize(w1, p=2, dim=1, eps=1e-12)
                w2 = F.normalize(w2, p=2, dim=1, eps=1e-12)
            w_a = 0.5 * w1 + 0.5 * w2
            self.ft_weight_attn = nn.Parameter(w_a.clone().data, requires_grad=True)
            params_list.append({'params': self.ft_weight_attn, 'lr': self.args.ft_lr})
        
        if self.args.merge_strategy == 'attn_s':
            self.fc_b = ScalarNet(tmp_all_way, self.args.BC_hidden_dim).cuda()
            self.fc_n = ScalarNet(tmp_all_way, self.args.BC_hidden_dim).cuda()
            self.fc_b.train()
            self.fc_n.train()
            params_list.append({'params': self.fc_b.parameters(), 'lr': self.args.BC_lr})
            params_list.append({'params': self.fc_n.parameters(), 'lr': self.args.BC_lr})

        # init optimizer
        if self.args.ft_optimizer.lower() == 'sgd':
            ft_optimizer = torch.optim.SGD(params_list, lr=self.args.ft_lr, 
                                           momentum=0.9, weight_decay=5e-4, nesterov=True)
        else:
            ft_optimizer = torch.optim.Adam(params_list, lr=self.args.ft_lr)

        # loss recorders
        L_ce = AverageMeter()
        L_kd = AverageMeter()
        L_extra = AverageMeter()
        L_norm  = AverageMeter()
        
        # define exemplar dataloader
        assert len(trainloader) == 1
        n_iters = self.args.ft_iters * self.args.ft_n_repeat
        tqdm_train = tqdm.tqdm(range(n_iters))
        data, label = [], []
        if self.args.using_exemplars:
            if session == 1:
                if self.args.batch_size_exemplar > 0:
                    batch_size_base = self.args.batch_size_exemplar
                else:
                    batch_size_base = self.base_way
                # base class exemplars
                base_exemplar_sampler  = Batch_DataSampler(self.base_exemplar_set.targets, 
                                                           self.args.ft_iters, batch_size_base) 
                base_exemplar_loader   = DataLoader(dataset=self.base_exemplar_set, batch_sampler=base_exemplar_sampler, 
                                                    num_workers=8, pin_memory=True)
                base_exemplar_loader   = iter(base_exemplar_loader)
            else:
                if self.args.batch_size_exemplar > 0:
                    batch_size_base  = self.args.batch_size_exemplar//2
                    batch_size_novel = self.args.batch_size_exemplar//2
                else:
                    batch_size_base  = self.base_way
                    batch_size_novel = tmp_base_way - self.base_way
                # base class exemplars
                base_exemplar_sampler  = Batch_DataSampler(self.base_exemplar_set.targets, 
                                                           self.args.ft_iters, batch_size_base) 
                base_exemplar_loader   = DataLoader(dataset=self.base_exemplar_set, batch_sampler=base_exemplar_sampler, 
                                                    num_workers=8, pin_memory=True)
                base_exemplar_loader   = iter(base_exemplar_loader)
                # novel class exemplars
                novel_exemplar_sampler = Batch_DataSampler(self.novel_exemplar_set.targets, 
                                                           self.args.ft_iters, batch_size_novel) 
                novel_exemplar_loader  = DataLoader(dataset=self.novel_exemplar_set, batch_sampler=novel_exemplar_sampler, 
                                                    num_workers=8, pin_memory=True)
                novel_exemplar_loader  = iter(novel_exemplar_loader)
        
        # finetuning
        for i in tqdm_train:
            # load N * K training data
            for batch in trainloader:
                data_tmp, label_tmp = [_.cuda() for _ in batch]
                data.append(data_tmp)
                label.append(label_tmp)
            # formulate a batch
            if (i + 1) % self.args.ft_n_repeat == 0:
                batch_x = torch.cat(data, dim=0)
                label_x = torch.cat(label, dim=0)
                data = []
                label = []
                if self.args.using_exemplars:
                    if session == 1:
                        batch_e, label_e = next(base_exemplar_loader)
                    else:
                        batch_e_b, label_e_b = next(base_exemplar_loader)
                        batch_e_n, label_e_n = next(novel_exemplar_loader)
                        batch_e, label_e = torch.cat([batch_e_b, batch_e_n], dim=0), torch.cat([label_e_b, label_e_n], dim=0)
                    batch_e = batch_e.cuda()
                    label_e = label_e.long().cuda()
                    bs_new = batch_x.shape[0]
                    bs_old = batch_e.shape[0]
                    batch_all = torch.cat([batch_x, batch_e], dim=0)
                    label_all = torch.cat([label_x, label_e], dim=0)
                else:
                    bs_new = batch_x.shape[0]
                    bs_old = 0
                    batch_all = batch_x
                # forward data flow
                logits_all, f_n = self.get_logits(self.feature, ft_weight, self.scale_cls.data, batch_all, return_feature=True)
                logits_x = logits_all[:bs_new]
                with torch.no_grad():
                    assert self.weights_prev.shape[0] == tmp_all_way
                    if self.args.ft_teacher == 'fixed':
                        logits_t, f_b = self.get_logits(self.feature_fixed, self.weights_fixed[:tmp_all_way], self.scale_cls.data, batch_all, return_feature=True)
                    elif self.args.ft_teacher == 'prev':
                        logits_t, f_b = self.get_logits(self.feature_prev, self.weights_prev, self.scale_cls.data, batch_all, return_feature=True)
                    elif self.args.ft_teacher == 'ema':
                        logits_t, f_b = self.get_logits(self.feature_ema, self.weights_ema, self.scale_cls.data, batch_all, return_feature=True)
                    else:
                        Exception()
                
                if self.args.using_exemplars and ('attn' in self.args.merge_strategy):
                    scores_b = logits_t.clone()
                    scores_n = logits_all.clone()
                    if self.args.BC_detach:
                        scores_b = scores_b.data
                        scores_n = scores_n.data
                    pred_b = torch.softmax(scores_b, dim=-1)                            # (bs, C)
                    pred_n = torch.softmax(scores_n, dim=-1)                            # (bs, C)
                    if self.args.BC_flatten == 'org':
                        fc_input = torch.cat([pred_b, pred_n], dim=-1)                  # (bs, 2C)
                    elif self.args.BC_flatten == 'sum':
                        pred_b_1 = torch.sum(pred_b[:, :self.base_way], dim=1, keepdim=True)
                        pred_b_2 = torch.sum(pred_b[:, self.base_way:], dim=1, keepdim=True)
                        pred_n_1 = torch.sum(pred_n[:, :self.base_way], dim=1, keepdim=True)
                        pred_n_2 = torch.sum(pred_n[:, self.base_way:], dim=1, keepdim=True)
                        fc_input = torch.cat([pred_b_1, pred_b_2, pred_n_1, pred_n_2], dim=-1) # (bs, 4)
                    elif self.args.BC_flatten == 'max':
                        dim_tmp = (tmp_all_way - self.base_way)
                        pred_b_1 = torch.topk(pred_b[:, :self.base_way], dim_tmp, dim=1)[0]
                        pred_b_2 = torch.topk(pred_b[:, self.base_way:], dim_tmp, dim=1)[0]
                        pred_n_1 = torch.topk(pred_n[:, :self.base_way], dim_tmp, dim=1)[0]
                        pred_n_2 = torch.topk(pred_n[:, self.base_way:], dim_tmp, dim=1)[0]
                        fc_input = torch.cat([pred_b_1, pred_b_2, pred_n_1, pred_n_2], dim=-1) # (bs, 4)
                    else:
                        Exception()
                    attn_logits  = self.binary_classifier(fc_input)                     # (bs, 2)
                    attn_weights = torch.softmax(attn_logits, dim=-1)                   # (bs, 2)

                    if self.args.merge_strategy == 'attn':
                        if self.args.masking_novel:
                            scores_b[:, self.base_way:] = 0.1 * scores_b[:, self.base_way:] # base branch only handles base class samples
                        scores_attn = attn_weights[:, 0:1] * scores_b + attn_weights[:, 1:2] * scores_n
                        loss_attn = self.args.w_BC_cls * F.cross_entropy(scores_attn, label_all, reduction='mean')
                        label_binary_all = (label_all >= self.base_way).long()
                        if hasattr(self.args, 'BC_binary_factor') and (self.args.BC_binary_factor != 1):
                            loss_binary_b = F.cross_entropy(attn_logits[label_all < self.base_way],
                                                            label_binary_all[label_all < self.base_way], 
                                                            reduction='mean') 
                            loss_binary_n = F.cross_entropy(attn_logits[label_all >= self.base_way],
                                                            label_binary_all[label_all >= self.base_way], 
                                                            reduction='mean')
                            loss_binary = self.args.w_BC_binary * 0.5 * (loss_binary_b + self.args.BC_binary_factor * loss_binary_n)
                        else:
                            loss_binary = self.args.w_BC_binary * F.cross_entropy(attn_logits, label_binary_all, reduction='mean')                    
                    elif self.args.merge_strategy == 'attn_f':
                        feat_b = f_b.clone()
                        feat_n = f_n.clone()
                        if self.args.BC_detach_f:
                            feat_b = feat_b.data
                            feat_n = feat_n.data
                        if self.norm_first:
                            feat_b = F.normalize(feat_b, p=2, dim=1, eps=1e-12)
                            feat_n = F.normalize(feat_n, p=2, dim=1, eps=1e-12)
                        feat_attn = attn_weights[:, 0:1] * feat_b + attn_weights[:, 1:2] * feat_n
                        feat_attn = F.normalize(feat_attn, p=2, dim=1, eps=1e-12)
                        w_attn = F.normalize(self.ft_weight_attn, p=2, dim=1, eps=1e-12)                
                        scores_attn = self.scale_cls.data * torch.mm(feat_attn, w_attn.t()) 
                        loss_attn = self.args.w_BC_cls * F.cross_entropy(scores_attn, label_all, reduction='mean')
                        label_binary_all = (label_all >= self.base_way).long()
                        loss_binary = self.args.w_BC_binary * F.cross_entropy(attn_logits, label_binary_all, reduction='mean')  
                    elif self.args.merge_strategy == 'attn_s':
                        scalar_b = self.fc_b(pred_b)
                        scalar_n = self.fc_n(pred_n)
                        attn_logits = torch.cat([scalar_b, scalar_n], dim=1)
                        attn_weights = torch.softmax(attn_logits, dim=-1)
                        if self.args.masking_novel:
                            scores_b[:, self.base_way:] = 0.1 * scores_b[:, self.base_way:] # base branch only handles base class samples
                        scores_attn = attn_weights[:, 0:1] * scores_b + attn_weights[:, 1:2] * scores_n
                        loss_attn = self.args.w_BC_cls * F.cross_entropy(scores_attn, label_all, reduction='mean')
                        label_binary_all = (label_all >= self.base_way).long()
                        if hasattr(self.args, 'BC_binary_factor') and (self.args.BC_binary_factor != 1):
                            loss_binary_b = F.cross_entropy(attn_logits[label_all < self.base_way],
                                                            label_binary_all[label_all < self.base_way], 
                                                            reduction='mean') 
                            loss_binary_n = F.cross_entropy(attn_logits[label_all >= self.base_way],
                                                            label_binary_all[label_all >= self.base_way], 
                                                            reduction='mean')
                            loss_binary = self.args.w_BC_binary * 0.5 * (loss_binary_b + self.args.BC_binary_factor * loss_binary_n)
                        else:
                            loss_binary = self.args.w_BC_binary * F.cross_entropy(attn_logits, label_binary_all, reduction='mean') 
                    else:
                        Exception()                            

                if self.args.using_exemplars: 
                    logits_t = logits_t[(bs_new - self.args.batch_size_current):]           # (bs_exe, base_way+n_way)
                    if (session > 1) and hasattr(self.args, 'EMA_logits') and self.args.EMA_logits:
                        assert self.args.ft_teacher == 'fixed'
                        assert self.args.EMA_factor_b_1 == self.args.EMA_factor_b_2
                        with torch.no_grad():
                            logits_t_prev = self.get_logits(self.feature_prev, self.weights_prev, self.scale_cls.data, batch_all)
                            logits_t_prev = logits_t_prev[(bs_new - self.args.batch_size_current):]
                        t_prev = session - 1
                        coeff_list = torch.ones(logits_t.shape[0], 1).float().cuda()        # (bs_exe, )
                        label_tmp = label_all[(bs_new - self.args.batch_size_current):]
                        sess_e = label2session(label_tmp, self.base_way, self.args.n_way)   # (bs_exe, )
                        mask_novel = (sess_e > 0)                                           # (bs_exe, )
                        mask_base  = (sess_e <= 0)
                        coeff_list[mask_base] = self.args.EMA_factor_b_1
                        sess_e_novel = sess_e[mask_novel]                                   # (bs_exe_n, ) e.g. [1, 2, 3 ...]
                        label_e_novel = label_tmp[mask_novel]                               # (bs_exe_n, )
                        coeff_novel = torch.ones(sess_e_novel.shape[0], 1).float().cuda()   # (bs_exe_n, )
                        if self.args.EMA_type == 'linear':
                            for idx_tmp, x_tmp in enumerate(sess_e_novel):
                                y_tmp = (self.args.EMA_factor_n_2 - self.args.EMA_factor_n_1) / (t_prev - 1 + 1e-8) * (x_tmp - 1) + self.args.EMA_factor_n_1
                                coeff_novel[idx_tmp] = y_tmp
                            coeff_list[mask_novel] = coeff_novel
                        elif self.args.EMA_type == 'window':
                            for idx_tmp, x_tmp in enumerate(sess_e_novel):
                                if max(1, t_prev - self.args.EMA_w_size) < x_tmp <= t_prev:
                                    y_tmp = self.args.EMA_factor_n_2 
                                else:
                                    y_tmp = self.args.EMA_factor_n_1
                                coeff_novel[idx_tmp] = y_tmp
                            coeff_list[mask_novel] = coeff_novel
                        elif self.args.EMA_type == 'linear_t':
                            y_tmp = (self.args.EMA_factor_n_2 - self.args.EMA_factor_n_1) / (self.args.n_sessions - 2 + 1e-8) * (session - 1) + self.args.EMA_factor_n_1
                            coeff_novel = y_tmp * coeff_novel
                            coeff_list[mask_novel] = coeff_novel
                        elif self.args.EMA_type == 'learnable':
                            y_tmp = torch.sigmoid(EMA_scalar)
                            coeff_novel = y_tmp * coeff_novel
                            coeff_list[mask_novel] = coeff_novel
                            if hasattr(self.args, 'w_l_order') and self.args.w_l_order == 1:
                                loss_norm = self.args.w_l * (1.0 - coeff_novel).mean()
                            else:
                                loss_norm = self.args.w_l * (torch.sqrt(((1.0 - coeff_novel)**2).sum()) / coeff_novel.shape[0])
                            L_norm.update(loss_norm.item())
                        elif self.args.EMA_type == 'learnable_v':
                            assert t_prev == EMA_vector.shape[0]
                            for idx_tmp, x_tmp in enumerate(sess_e_novel):              # x_tmp=1 -> idx=0, x_tmp=2 -> idx=1
                                y_tmp = torch.sigmoid(EMA_vector[x_tmp - 1])
                                coeff_novel[idx_tmp] = y_tmp
                            coeff_list[mask_novel] = coeff_novel
                            if hasattr(self.args, 'w_l_order') and self.args.w_l_order == 1:
                                loss_norm = self.args.w_l * (1.0 - coeff_novel).mean()
                            else:
                                loss_norm = self.args.w_l * (torch.sqrt(((1.0 - coeff_novel)**2).sum()) / coeff_novel.shape[0])
                            L_norm.update(loss_norm.item())
                        elif self.args.EMA_type == 'learnable_mlp_c':
                            inputs = self.vector_table[label_e_novel - self.base_way]   # (n_way_novel, all_classes_t-1) -> (n_exe, all_classes_t-1)
                            coeff_novel = torch.sigmoid(self.coeff_mlp(inputs)[:, 0:1]) # (n_exe, 1) 
                            coeff_list[mask_novel] = coeff_novel
                            if hasattr(self.args, 'w_l_order') and self.args.w_l_order == 1:
                                loss_norm = self.args.w_l * (1.0 - coeff_novel).mean()
                            else:
                                loss_norm = self.args.w_l * (torch.sqrt(((1.0 - coeff_novel)**2).sum()) / coeff_novel.shape[0])
                            L_norm.update(loss_norm.item())
                        elif self.args.EMA_type == 'learnable_mlp_b':
                            inputs = self.vector_table[label_e_novel - self.base_way]   # (n_way_novel, all_classes_t-1) -> (n_exe, all_classes_t-1)
                            coeff_novel = torch.sigmoid(self.coeff_mlp(inputs)[:, 0:1]) # (n_exe, 1) 
                            coeff_list[mask_novel] = coeff_novel
                            if hasattr(self.args, 'w_l_order') and self.args.w_l_order == 1:
                                loss_norm = self.args.w_l * (1.0 - coeff_novel).mean()
                            else:
                                loss_norm = self.args.w_l * (torch.sqrt(((1.0 - coeff_novel)**2).sum()) / coeff_novel.shape[0])
                            L_norm.update(loss_norm.item())
                        elif self.args.EMA_type == 'learnable_mlp_v':
                            inputs = self.vector_table[sess_e_novel - 1]                # (n_session, t) -> (n_exe, t)
                            coeff_novel = torch.sigmoid(self.coeff_mlp(inputs)[:, 0:1]) # (n_exe, t)
                            coeff_list[mask_novel] = coeff_novel
                            if hasattr(self.args, 'w_l_order') and self.args.w_l_order == 1:
                                loss_norm = self.args.w_l * (1.0 - coeff_novel).mean()
                            else:
                                loss_norm = self.args.w_l * (torch.sqrt(((1.0 - coeff_novel)**2).sum()) / coeff_novel.shape[0])
                            L_norm.update(loss_norm.item())
                        elif self.args.EMA_type == 'learnable_c':
                            assert (tmp_all_way - self.base_way - self.args.n_way) == EMA_vector.shape[0]
                            for idx_tmp, x_tmp in enumerate(label_e_novel):             # x_tmp=1 -> idx=0, x_tmp=2 -> idx=1
                                y_tmp = torch.sigmoid(EMA_vector[x_tmp - self.base_way])
                                coeff_novel[idx_tmp] = y_tmp
                            coeff_list[mask_novel] = coeff_novel
                            if hasattr(self.args, 'w_l_order') and self.args.w_l_order == 1:
                                loss_norm = self.args.w_l * (1.0 - coeff_novel).mean()
                            else:
                                loss_norm = self.args.w_l * (torch.sqrt(((1.0 - coeff_novel)**2).sum()) / coeff_novel.shape[0])
                            L_norm.update(loss_norm.item())
                        elif self.args.EMA_type == 'learnable_s':
                            assert t_prev == EMA_vector.shape[0]
                            for idx_tmp, x_tmp in enumerate(sess_e_novel): # x_tmp=1 -> idx=0, x_tmp=2 -> idx=1
                                if self.args.EMA_s_type == 0:
                                    y_tmp = EMA_vector[x_tmp - 1] * (self.EMA_FC_K * self.sim_n2b[label_e_novel[idx_tmp] - self.base_way] + self.EMA_FC_b)
                                    y_tmp = torch.sigmoid(y_tmp)
                                elif self.args.EMA_s_type == 1:
                                    y_tmp = EMA_vector[x_tmp - 1] * (self.EMA_FC_K * self.sim_n2b[label_e_novel[idx_tmp] - self.base_way])
                                    y_tmp = torch.sigmoid(y_tmp)
                                elif self.args.EMA_s_type == 2:
                                    y_tmp = torch.sigmoid(EMA_vector[x_tmp - 1]) * (self.sim_n2b[label_e_novel[idx_tmp] - self.base_way])
                                elif self.args.EMA_s_type == 3:
                                    y_tmp = (self.EMA_FC_K * self.sim_n2b[label_e_novel[idx_tmp] - self.base_way] + self.EMA_FC_b)
                                    y_tmp = torch.sigmoid(y_tmp)
                                elif self.args.EMA_s_type == 4:
                                    y_tmp = (self.EMA_FC_K * self.sim_n2b[label_e_novel[idx_tmp] - self.base_way])
                                    y_tmp = torch.clamp(y_tmp, min=0.0, max=1.0)
                                else:
                                    Exception()
                                coeff_novel[idx_tmp] = y_tmp
                            coeff_list[mask_novel] = coeff_novel
                            if hasattr(self.args, 'w_l_order') and self.args.w_l_order == 1:
                                loss_norm = self.args.w_l * (1.0 - coeff_novel).mean()
                            else:
                                loss_norm = self.args.w_l * (torch.sqrt(((1.0 - coeff_novel)**2).sum()) / coeff_novel.shape[0])
                            L_norm.update(loss_norm.item())
                        else:
                            Exception()
                        if hasattr(self.args, 'EMA_prob') and self.args.EMA_prob:
                            EMA_prob = True
                            logits_t = coeff_list * torch.softmax(logits_t/self.args.ft_T, dim=-1) + \
                                       (1 - coeff_list) * torch.softmax(logits_t_prev/self.args.ft_T, dim=-1)
                        else:
                            logits_t = coeff_list * logits_t + (1 - coeff_list) * logits_t_prev
                    if hasattr(self.args, 'KD_rectified') and self.args.KD_rectified:
                        logits_t_rcf = self.args.KD_rectified_factor * logits_t
                        rcf_mask = convert2one_hot(label_e, logits_t.shape[1])      # (bs_exe, base_way+n_way)
                        logits_t = torch.where(rcf_mask.byte(), logits_t, logits_t_rcf)
                
                # compute loss functions
                ### classification loss (cross-entropy/triplet loss)
                if hasattr(self.args, 'triplet') and self.args.triplet and self.args.triplet_gap != 0:
                    cos_matrix = logits_x / self.scale_cls.data                     # (bs_lb, base_way+n_way)
                    sim_positive = cos_matrix[range(cos_matrix.shape[0]), label_x]  # (bs_lb,)
                    margin_mask = convert2one_hot(label_x, logits_x.shape[1])       # (bs_lb, base_way+n_way)
                    neg_matrix = torch.where(margin_mask.byte(), 
                        -1 * torch.ones_like(cos_matrix).cuda(), cos_matrix)        # (bs_lb, base_way+n_way)
                    sim_negative, _ = torch.max(neg_matrix, dim=1)                  # (bs_lb,)
                    loss_ce = torch.clamp((sim_negative - sim_positive + self.args.triplet_gap), min=0)
                    loss_ce = loss_ce.mean()
                else:
                    if hasattr(self.args, 'margin') and self.args.margin != 0:
                        phi = logits_x - self.scale_cls.data * self.args.margin
                        margin_mask = convert2one_hot(label_x, logits_x.shape[1])
                        margin_mask[:, :tmp_base_way] = 0
                        logits_x = torch.where(margin_mask.byte(), phi, logits_x)
                    loss_ce = F.cross_entropy(logits_x, label_x, reduction='mean')
                loss_ce = self.args.w_cls * loss_ce
                L_ce.update(loss_ce.item())
                
                ### distillation loss
                logits_s = logits_all[(-1 * logits_t.shape[0]):]
                assert logits_s.shape[0] == logits_t.shape[0]
                if hasattr(self.args, 'weighted_kd') and self.args.weighted_kd:
                    KD_weights = torch.ones(logits_s.shape[0]).cuda()
                    KD_weights[label_e >= self.base_way] = self.args.w_kd_novel
                else:
                    KD_weights = torch.ones(logits_s.shape[0]).cuda()
                if hasattr(self.args, 'ft_KD_all') and self.args.ft_KD_all:
                    loss_kd = w_KDloss(logits_s[:, :tmp_all_way],  logits_t[:, :tmp_all_way],  self.args.ft_T, KD_weights, EMA_prob)
                else:
                    loss_kd = w_KDloss(logits_s[:, :tmp_base_way], logits_t[:, :tmp_base_way], self.args.ft_T, KD_weights, EMA_prob)
                loss_kd = self.args.w_d * loss_kd
                L_kd.update(loss_kd.item())

                ### extra loss
                loss_extra = None
                if self.args.using_exemplars:
                    loss_extra = self.args.w_e * F.cross_entropy(logits_all[bs_new:], label_e, reduction='mean')
                    L_extra.update(loss_extra.item())

                if loss_extra is not None:
                    loss = loss_ce + loss_kd + loss_extra
                else:
                    loss = loss_ce + loss_kd

                if (session > 1) and hasattr(self.args, 'EMA_logits') and self.args.EMA_logits and ('learnable' in self.args.EMA_type):
                    loss += loss_norm
                
                if self.args.using_exemplars and ('attn' in self.args.merge_strategy):
                    loss += (loss_attn + loss_binary)
                
                # update
                ft_optimizer.zero_grad()
                loss.backward()
                ft_optimizer.step()
                str_tqdm = 'Session %d finetuning iter %d/%d: loss_ce=%.4f loss_kd=%.4f loss_extra=%.4f'%(session, (i+1)/self.args.ft_n_repeat,
                            self.args.ft_iters, loss_ce.item(), loss_kd.item(), (loss_extra.item() if loss_extra is not None else 0))
                if (session > 1) and hasattr(self.args, 'EMA_logits') and self.args.EMA_logits and ('learnable' in self.args.EMA_type):
                    str_tqdm += ' loss_norm=%.4f'%(loss_norm.item())
                if self.args.using_exemplars and ('attn' in self.args.merge_strategy):
                    str_tqdm += ' loss_attn=%.4f loss_binary=%.4f'%(loss_attn.item(), loss_binary.item())
                tqdm_train.set_description(str_tqdm)

                # re-imprint ft_weight
                if hasattr(self.args, 'imprint_ft_weight') and self.args.imprint_ft_weight:
                    imprinted_weights = self.get_class_mean_v2(self.feature, imgs_all, labels_all, session)
                    ft_weight = nn.Parameter(imprinted_weights.clone().data, requires_grad=False)
                    assert ft_weight.shape == self.joint_weight.shape
                    if self.args.part_frozen == False:
                        self.feature.train()
                        for p in self.feature.parameters():
                            p.requires_grad = True
                    else:
                        self.feature.eval()
                        for p in self.feature.parameters():
                            p.requires_grad = False
                        self.feature.layer4.train()
                        for p in self.feature.layer4.parameters():
                            p.requires_grad = True
        
        # update classification weights to the finetuned ones
        self.joint_weight.data = ft_weight.clone().data
        
        # update model t-1
        self.feature_prev.load_state_dict(self.feature.state_dict(), strict=True) 
        self.weights_prev = self.joint_weight.clone().data
        for p in self.feature_prev.parameters():
            p.requires_grad = False
        self.weights_prev.requires_grad = False

        # momentum update model
        if self.args.ft_momentum_type == 0:
            for param_ema, param_fixed, param_tmp in zip(self.feature_ema.parameters(), self.feature_fixed.parameters(), self.feature.parameters()):
                param_ema.data = self.args.ft_momentum * param_fixed.data + (1 - self.args.ft_momentum) * param_tmp.data
            weight_1 = self.weights_fixed[:self.joint_weight.shape[0]].clone().data
            weight_2 = self.joint_weight.clone().data
        elif self.args.ft_momentum_type == 1:
            for param_ema, param_tmp in zip(self.feature_ema.parameters(), self.feature.parameters()):
                param_ema.data = self.args.ft_momentum * param_ema.data + (1 - self.args.ft_momentum) * param_tmp.data
            weight_1 = self.weights_ema.clone().data
            weight_2 = self.joint_weight.clone().data
        else:
            Exception()
        if self.norm_first:
            weight_1 = F.normalize(weight_1, p=2, dim=1, eps=1e-12)
            weight_2 = F.normalize(weight_2, p=2, dim=1, eps=1e-12)
        self.weights_ema = self.args.ft_momentum * weight_1 + (1 - self.args.ft_momentum) * weight_2
        for p in self.feature_ema.parameters():
            p.requires_grad = False
        self.weights_ema.requires_grad = False
        
        if hasattr(self.args, 'vis_logits') and self.args.vis_logits:
            # append feature after finetuning at session t
            feature_tmp = resnet18(False, self.args).cuda()
            feature_tmp.load_state_dict(self.feature.state_dict(), strict=True)
            self.feature_list.append(feature_tmp)
            # append classification weights after finetuning at session t
            weights_tmp = self.joint_weight.clone().data
            self.weights_list.append(weights_tmp)
            # set grad to False
            for i in range(len(self.feature_list)):
                for p in self.feature_list[i].parameters():
                    p.requires_grad = False
                self.weights_list[i].requires_grad = False
        
        return [L_ce, L_kd, L_extra]


    def inc_forward(self, z_support, z_query, y_b=None):
        '''
        input:
            z_support.shape = (bs, n_way, n_shot, C) the support samples are only from novel classes
            z_query.shape = (bs, 2*n_way*n_query, C) the query samples are from base and novel classes
            y_b.shape = (bs, n_way) fake novel class label (if provided as in fake novel training)
        output:
            prediction and generated weights
        '''
        if self.norm_first:
            z_support = F.normalize(z_support, p=2, dim=-1, eps=1e-12)          # (bs, n_way, n_shot, feat_dim)
        z_proto = z_support.mean(-2)                                            # (bs, n_way, feat_dim)
        z_query = F.normalize(z_query, p=2, dim=-1, eps=1e-12)                  # (bs, 2*n_way*n_query, feat_dim)

        if self.phase == 'pre_train':
            assert self.method == 'imprint'
        if self.method == 'LwoF':
            assert self.norm_first == True
        
        inc_weight = self.weight_generation(z_proto, gen_type=self.method, 
                                            z_support=z_support, y_b=y_b)       # (bs, base_way+n_way, feat_dim) or (bs, base_way, feat_dim)
        scores  = self.scale_cls * torch.bmm(z_query, inc_weight.permute(0,2,1))# (bs, 2*n_way*n_query, base_way+n_way) or (bs, 2*n_way*n_query, base_way)
       
        return scores, inc_weight
    

    def set_forward(self, z_support, z_query):
        '''
        input:
            z_support.shape = (bs, n_way, n_shot, C) the support samples are from novel classes
            z_query.shape = (bs, n_way*n_query, C) the query samples are from novel classes
        output:
            prediction and generated weights
        '''
        if self.norm_first:
            z_support = F.normalize(z_support, p=2, dim=-1, eps=1e-12)          # (bs, n_way, n_shot, feat_dim)
        z_proto = z_support.mean(-2)                                            # (bs, n_way, feat_dim)
        z_query = F.normalize(z_query, p=2, dim=-1, eps=1e-12)                  # (bs, n_way*n_query, feat_dim)

        if self.phase == 'pre_train':
            assert self.method == 'imprint'
        if self.method == 'LwoF':
            assert self.norm_first == True
        
        # inc_weight = self.weight_generation(z_proto, gen_type=self.method, 
        #                                     z_support=z_support)
        # novel_weight = inc_weight[:, self.base_way:]                            # (bs, base_way+n_way, feat_dim) -> (bs, n_way, feat_dim)
        assert self.method == 'imprint'
        novel_weight = F.normalize(z_proto, p=2, dim=-1, eps=1e-12)             # (bs, n_way, feat_dim)
        scores  = self.scale_cls * torch.bmm(z_query, 
                                             novel_weight.permute(0, 2, 1))     # (bs, n_way*n_query, n_way)
       
        return scores, novel_weight


    def weight_generation(self, z_proto, gen_type='imprint', z_support=None, y_b=None):
        '''
        input:
            z_proto is the novel class prototypes i.e. w'_avg of {n_way} novel category in the paper 
            z_proto.shape = (bs, n_way, feat_dim) (not l2 normalized)
            z_support is {n_support} training examples of {n_way} novel category 
            z_support.shape = (bs, n_way, n_support, feat_dim) (l2 normalized)
            y_b.shape = (bs, n_way) is the indices of the Fake Novel classes
        output:
            generated weights
        '''
        bs = z_proto.size(0)
        flag_FakeNovel = False
        if y_b is not None:
            flag_FakeNovel = True
            if gen_type =='LwoF':
                y_b_ = y_b.cpu().numpy().tolist()                                   # (bs, n_way)
                y_c = []                                                            # y_c is the complementary set of y_b
                for x in y_b_:
                    set_union  = set(range(self.base_way)) - set(x)
                    list_union = list(set_union)
                    list_union.sort()
                    y_c.append(list_union)
                y_c = torch.from_numpy(np.array(y_c)).long().cuda()                 # (bs, base_way-n_way)
                y_c = y_c.unsqueeze(2)                                              # (bs, base_way-n_way, 1)
                index_c = y_c.expand(-1, -1, self.feat_dim)                         # (bs, base_way-n_way, C)
            y_b = y_b.unsqueeze(2)                                                  # (bs, n_way, 1)
            index = y_b.expand(-1, -1, self.feat_dim)                               # (bs, n_way, C)  

        base_weight = F.normalize(self.base_weight, p=2, dim=-1, eps=1e-12)         # (base_way, C)
        base_weight = base_weight.unsqueeze(0).expand(bs, -1, -1)                   # (bs, base_way, C)
        
        if gen_type == 'imprint':
            novel_weight= F.normalize(z_proto, p=2, dim=-1, eps=1e-12)              # (bs, n_way, C)
            if not flag_FakeNovel:
                inc_weight = torch.cat([base_weight, novel_weight], dim=1)          # (bs, base_way+n_way, C)
            else:
                inc_weight = base_weight.scatter(1, index, novel_weight)            # (bs, base_way, C)
        
        elif gen_type =='LwoF':
            bs, n_novel_way, n_support, feat_dim = z_support.shape                  # (bs, n_way, n_support, C)           
            weight_avg = self.w_avg.unsqueeze(0).unsqueeze(0) * z_proto             # (bs, n_way, C)
            querys = torch.bmm(z_support.view(bs, -1, feat_dim),                    # (bs, n_way*n_shot, C) * (bs, C, C) ->
                               self.w_q.unsqueeze(0).expand(bs, -1, -1))            # (bs, n_way*n_shot, C) {\phi_q*z_i'}
            querys = F.normalize(querys, p=2, dim=-1, eps=1e-12)                    # (bs, n_way*n_shot, C)
            keys = F.normalize(self.w_keys, p=2, dim=1, eps=1e-12)                  # (base_way, C) {k_b}
            keys = keys.unsqueeze(0).expand(bs, -1, -1)                             # (bs, base_way, C)
            if not flag_FakeNovel:
                att_coeff =  torch.bmm(querys, keys.permute(0, 2, 1))               # (bs, n_way*n_shot, C) * (bs, C, base_way) -> (bs, n_way*n_shot, base_way)
                att_coeff = F.softmax(self.scale_att * att_coeff, dim=-1)           # (bs, n_way*n_shot, base_way) {Att(\phi_q*z_i', k_b)}
                weight_att = torch.bmm(att_coeff, base_weight)                      # (bs, n_way*n_shot, base_way) * (bs, base_way, C) -> (bs, n_way*n_shot, C)
                weight_att = weight_att.view(bs, n_novel_way, 
                                             n_support, feat_dim).mean(-2)          # (bs, n_way, n_shot, C) -> (bs, n_way, C)
                weight_att = self.w_att.unsqueeze(0).unsqueeze(0) * weight_att      # (bs, n_way, C)
                novel_weight = weight_avg + weight_att                              # (bs, n_way, C)
                novel_weight = F.normalize(novel_weight, p=2, dim=-1, eps=1e-12)    # (bs, n_way, C)
                inc_weight  = torch.cat([base_weight, novel_weight], dim=1)         # (bs, base_way+n_way, C)
            else:
                # filter fake novel classes  
                fake_base_keys = torch.gather(keys, 1, index_c)                     # (bs, base_way-n_way, C)
                fake_base_weight = torch.gather(base_weight, 1, index_c)            # (bs, base_way-n_way, C)
                # attention mechanism
                att_coeff = torch.bmm(querys, fake_base_keys.permute(0, 2, 1))      # (bs, n_way*n_shot, C) * (bs, C, base_way-n_way) -> (bs, n_way*n_shot, base_way-n_way)
                att_coeff = F.softmax(self.scale_att * att_coeff, dim=-1)           # (bs, n_way*n_shot, base_way-n_way) 
                weight_att = torch.bmm(att_coeff, fake_base_weight)                 # (bs, n_way*n_shot, base_way-n_way) * (bs, base_way-n_way, C) -> (bs, n_way*n_shot, C)
                weight_att = weight_att.view(bs, n_novel_way,
                                             n_support,feat_dim).mean(-2)           # (bs, n_way, n_shot, C) -> (bs, n_way, C)
                weight_att = self.w_att.unsqueeze(0).unsqueeze(0) * weight_att      # (bs, n_way, C)
                novel_weight = weight_avg + weight_att                              # (bs, n_way, C)
                novel_weight = F.normalize(novel_weight, p=2, dim=-1, eps=1e-12)    # (bs, n_way, C)
                inc_weight = base_weight.clone()                                    # (bs, base_way, C)
                inc_weight = base_weight.scatter(1, index, novel_weight)            # (bs, base_way, C)
        
        else:
            raise Exception('Unsupported weight generation type!')
        
        return inc_weight   