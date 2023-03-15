import torch
import random
import numpy as np
from IPython import embed


class MetaSampler():
    '''
    sample a few-shot learning episode of shape (n_cls * (n_shot+n_query))
    '''
    def __init__(self, label, n_batch, n_cls, n_shot, n_query):
        self.n_batch = n_batch          # number of episode
        self.n_cls = n_cls              # number of class to sample
        self.n_shot = n_shot            # number of support samples per class
        self.n_query = n_query          # number of query samples per class

        label = np.array(label)         # e.g. a list of samples in {0,1,2,...63} for mini-imagenet, local label
        self.m_ind = []
        for i in range(max(label) + 1): # loop over all the classes
            ind = np.argwhere(label == i).reshape(-1) # np.array of positions where label = i, i = 0,1,2...
            self.m_ind.append(ind)      # a list of numpy arrays, the i-th element are positions where label = i

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []                  # a list of Tensor lists of image idxs, where idxs belong to the same class in each sub-list
            classes = np.random.choice(len(self.m_ind), self.n_cls, replace=False)

            for c in classes:
                l = np.random.choice(self.m_ind[c], self.n_shot+self.n_query, replace=False)
                batch.append(torch.from_numpy(l))

            batch = torch.stack(batch).reshape(-1) # (n_cls, n_shot+n_query) -> (n_cls * (n_shot+n_query))
            yield batch


class BaseSampler():
    '''
    sample n_query samples from ALL base classes
    e.g. n_query = 5 * 15
    '''
    def __init__(self, label, n_batch, n_query):
        self.n_batch = n_batch  # number of episode
        self.n_query = n_query  # number of query samples to sample from base classes
        self.label = np.array(label) 

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            l = np.random.choice(len(self.label), self.n_query, replace=False)
            batch = torch.from_numpy(l)
            yield batch


class FakeNovelSampler():
    '''
    Fake Novel sampling for fake novel training, which returns:
    fake base [query] + fake novel [support + query]: ([n_way*n_query]+[n_way*(n_shot+n_query)], )
    '''
    def __init__(self, label, n_batch, n_cls, n_shot, n_query):
        self.n_batch = n_batch          # number of episode
        self.n_cls = n_cls              # number of class to sample (novel class)
        self.n_shot = n_shot            # number of support samples per class
        self.n_query = n_query          # number of query samples per class

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1): # loop over all the classes
            ind = np.argwhere(label == i).reshape(-1) 
            self.m_ind.append(ind)                 

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            # sample fake novel classes
            batch_novel = []
            classes = np.random.choice(len(self.m_ind), self.n_cls, replace=False)
            for c in classes:
                l = np.random.choice(self.m_ind[c], self.n_shot+self.n_query, replace=False)
                batch_novel.append(torch.from_numpy(l))
            batch_novel = torch.stack(batch_novel).reshape(-1) # (n_cls, n_shot+n_query) -> (n_cls * (n_shot+n_query))
            # sample fake base classes
            base_idx = []
            for i in range(len(self.m_ind)):
                if i not in classes:
                    base_idx += self.m_ind[i].tolist()
            base_idx = np.array(base_idx)
            batch_base = np.random.choice(base_idx, self.n_cls*self.n_query, replace=False)
            batch_base = torch.from_numpy(batch_base)

            batch = torch.cat([batch_base, batch_novel], dim=0) # ([n_way*n_query+n_way*(n_shot+n_query)], ) 
            yield batch


class Batch_MetaSampler():
    '''
    sample a batch of few-shot learning episodes of shape (bs * n_cls * (n_shot+n_query))
    '''
    def __init__(self, label, n_batch, n_cls, n_shot, n_query, ep_per_batch=1):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_shot + n_query
        self.ep_per_batch = ep_per_batch

        label = np.array(label)
        self.catlocs = []
        for c in range(max(label) + 1):
            self.catlocs.append(np.argwhere(label == c).reshape(-1))

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for i_ep in range(self.ep_per_batch):
                episode = []
                classes = np.random.choice(len(self.catlocs), self.n_cls, replace=False)
                for c in classes:
                    l = np.random.choice(self.catlocs[c], self.n_per, replace=False)
                    episode.append(torch.from_numpy(l))
                episode = torch.stack(episode)
                batch.append(episode)
            batch = torch.stack(batch) # bs * n_cls * n_per
            yield batch.view(-1)


class Batch_BaseSampler():
    def __init__(self, label, n_batch, n_query, ep_per_batch=1):
        self.n_batch = n_batch  # number of episode
        self.n_query = n_query  # number of query samples to sample from base classes
        self.ep_per_batch = ep_per_batch
        self.label = np.array(label) 

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for i_ep in range(self.ep_per_batch):
                l = np.random.choice(len(self.label), self.n_query, replace=False)
                batch.append(torch.from_numpy(l))
            batch = torch.stack(batch) # bs * n_query 
            yield batch.view(-1)


class Batch_DataSampler():
    def __init__(self, label, n_batch, n_query, ep_per_batch=1):
        self.n_batch = n_batch  # number of episode
        self.n_query = n_query  # number of query samples to sample from base classes
        self.ep_per_batch = ep_per_batch
        self.label = np.array(label) 

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for i_ep in range(self.ep_per_batch):
                l = np.random.choice(len(self.label), self.n_query, replace=True)
                batch.append(torch.from_numpy(l))
            batch = torch.stack(batch) # bs * n_query 
            yield batch.view(-1)


class Batch_FakeNovelSampler():
    def __init__(self, label, n_batch, n_cls, n_shot, n_query, ep_per_batch=1):
        self.n_batch = n_batch          # number of episode
        self.n_cls = n_cls              # number of class to sample (novel class)
        self.n_shot = n_shot            # number of support samples per class
        self.n_query = n_query          # number of query samples per class
        self.ep_per_batch = ep_per_batch

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1): # loop over all the classes
            ind = np.argwhere(label == i).reshape(-1) 
            self.m_ind.append(ind)                 

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch_all = []
            for i_ep in range(self.ep_per_batch):
                # sample fake novel classes
                batch_novel = []
                classes = np.random.choice(len(self.m_ind), self.n_cls, replace=False)
                for c in classes:
                    l = np.random.choice(self.m_ind[c], self.n_shot+self.n_query, replace=False)
                    batch_novel.append(torch.from_numpy(l))
                batch_novel = torch.stack(batch_novel).reshape(-1) # (n_cls, n_shot+n_query) -> (n_cls * (n_shot+n_query))
                # sample fake base classes
                base_idx = []
                for i in range(len(self.m_ind)):
                    if i not in classes:
                        base_idx += self.m_ind[i].tolist()
                base_idx = np.array(base_idx)
                batch_base = np.random.choice(base_idx, self.n_cls*self.n_query, replace=False)
                batch_base = torch.from_numpy(batch_base)

                batch = torch.cat([batch_base, batch_novel], dim=0) # ([n_way*n_query+n_way*(n_shot+n_query)], ) 
                batch_all.append(batch)
            batch_all = torch.stack(batch_all)
            yield batch_all.view(-1)


class Batch_PairedSampler():
    '''
    sample a batch of pairs of samples of shape (bs*2)
    '''
    def __init__(self, label, n_batch, n_pairs, np_random=False):
        label = np.array(label)
        self.num_class = max(label) + 1
        self.n_batch = n_batch
        self.n_pairs = n_pairs # bs
        self.np_random = np_random
        self.catlocs = []
        for c in range(self.num_class):
            self.catlocs.append(np.argwhere(label == c).reshape(-1))

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            if self.np_random:
                classes = np.random.choice(len(self.catlocs), self.n_pairs, replace=True)
            else:
                classes = random.choices(list(range(self.num_class)), k=self.n_pairs)
            for c in classes:
                if self.np_random:
                    l = np.random.choice(self.catlocs[c], 2, replace=False)
                else:
                    l = np.array(random.sample(self.catlocs[c].tolist(), 2))
                batch.append(torch.from_numpy(l))
            batch = torch.stack(batch)           
            yield batch.view(-1) # bs * 2


class Batch_FakeNovelSampler_2():
    def __init__(self, label, n_batch, n_cls, n_query, n_unlabel, ep_per_batch=1):
        self.n_batch = n_batch          # number of episode
        self.n_cls = n_cls              # number of class to sample (novel class)
        self.n_query = n_query          # number of support samples per class
        self.n_unlabel = n_unlabel      # number of query samples per class
        self.ep_per_batch = ep_per_batch

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1): # loop over all the classes
            ind = np.argwhere(label == i).reshape(-1) 
            self.m_ind.append(ind)                 

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch_all = []
            for i_ep in range(self.ep_per_batch):
                # sample fake novel classes
                batch_novel = []
                classes = np.random.choice(len(self.m_ind), self.n_cls, replace=False)
                for c in classes:
                    l = np.random.choice(self.m_ind[c], self.n_query, replace=False)
                    batch_novel.append(torch.from_numpy(l))
                batch_novel = torch.stack(batch_novel).reshape(-1) # (n_cls, n_query) -> (n_cls * n_query)
                # sample fake base classes
                base_idx = []
                for i in range(len(self.m_ind)):
                    base_idx += self.m_ind[i].tolist()
                base_idx = np.array(base_idx)
                batch_base = np.random.choice(base_idx, self.n_cls*self.n_unlabel, replace=False)
                batch_base = torch.from_numpy(batch_base)

                batch = torch.cat([batch_novel, batch_base], dim=0) # ([n_way*n_query+n_way*n_unlabel], ) 
                batch_all.append(batch)
            batch_all = torch.stack(batch_all)
            yield batch_all.view(-1)