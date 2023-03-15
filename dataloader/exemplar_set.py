import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from dataloader.autoaugment_mini import AutoAugImageNetPolicy


class ExemplarSet(Dataset):

    def __init__(self, exemplar_path, exemplar_label, dataset='mini_imagenet', train=True):
        self.dataset = dataset
        self.train = train  # training or testing stage
        self.return_idx = False
        if self.dataset == 'mini_imagenet':
            image_size = 84
            # transform used in training
            self.transform_train = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                AutoAugImageNetPolicy(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            # transform used in testing
            self.transform_test = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.update(exemplar_path, exemplar_label)
    
    def update(self, exemplar_path, exemplar_label):
        self.data = exemplar_path
        self.targets = exemplar_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img, targets = self.data[i], self.targets[i]
        if self.train:
            if self.dataset == 'cifar100':
                image = self.transform_train(Image.fromarray(img))
            else:
                image = self.transform_train(Image.open(img).convert('RGB'))
        else:
            if self.dataset == 'cifar100':
                image = self.transform_test(Image.fromarray(img))
            else:
                image = self.transform_test(Image.open(img).convert('RGB'))
        
        if self.return_idx:
            return image, targets, i
        else: 
            return image, targets