import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryNet(nn.Module):
    '''
    simple binary classification network with 1 hidden layer
    '''
    def __init__(self, input_dim, hidden_dim=64):
        super(BinaryNet, self).__init__()   
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim    

        self.fc_layer1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc_layer2 = nn.Linear(self.hidden_dim, 2)       
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.fc_layer1(x)
        x = self.relu(x)
        x = self.fc_layer2(x)
        return x


class ScalarNet(nn.Module):
    '''
    simple fully-connected network with 1 hidden layer which outputs a single scalar
    '''
    def __init__(self, input_dim, hidden_dim=64):
        super(ScalarNet, self).__init__()   
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim    

        self.fc_layer1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc_layer2 = nn.Linear(self.hidden_dim, 1)       
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.fc_layer1(x)
        x = self.relu(x)
        x = self.fc_layer2(x)
        return x


class BinaryNet0(nn.Module):
    '''
    binary classification network without hidden layer
    '''
    def __init__(self, feat_dim, args):
        super(BinaryNet0, self).__init__()
        assert (args.using_BC) and (args.BC_n_hidden == 0)
        self.BC_norm_first = args.BC_norm_first        
        self.fc_layer1 = nn.Linear(feat_dim, 2)
    
    def forward(self, x):
        if self.BC_norm_first: x = F.normalize(x, p=2, dim=1, eps=1e-12)        
        x = self.fc_layer1(x)
        return x


class BinaryNet1(nn.Module):
    '''
    binary classification network with 1 hidden layer
    '''
    def __init__(self, feat_dim, args):
        super(BinaryNet1, self).__init__()
        assert (args.using_BC) and (args.BC_n_hidden >= 1)
        self.BC_norm_first = args.BC_norm_first
        self.BC_dropout = args.BC_dropout
        self.BC_batchnorm = args.BC_batchnorm
        
        self.fc_layer1 = nn.Linear(feat_dim, args.BC_hidden_dim)
        self.fc_layer2 = nn.Linear(args.BC_hidden_dim, 2)
        
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(args.BC_hidden_dim)
        self.dropout = nn.Dropout(p=args.BC_dropout_p)
    
    def forward(self, x):
        if self.BC_norm_first: x = F.normalize(x, p=2, dim=1, eps=1e-12)        
        x = self.fc_layer1(x)
        if self.BC_batchnorm: x = self.bn(x)
        x = self.relu(x)
        if self.BC_dropout: x = self.dropout(x)
        x = self.fc_layer2(x)
        return x


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--using_BC', action='store_true', help='use a binary classifier to determine the corresponding branch')
    parser.add_argument('--BC_norm_first', action='store_true', help='l2 norm the input features')
    parser.add_argument('--BC_dropout', action='store_true', help='use dropout layers in FC layers')
    parser.add_argument('--BC_dropout_p', type=float, default=0.5, help='the dropout probability')
    parser.add_argument('--BC_batchnorm', action='store_true', help='use batch norm in FC layers')
    parser.add_argument('--BC_n_hidden', type=int, default=0, help='number of hidden layers in the binary classifier (up to 1)')
    parser.add_argument('--BC_hidden_dim', type=int, default=128, help='the hidden dim in the hidden layer')
    args = parser.parse_args()
    model = BinaryNet0(512, args) if args.BC_n_hidden == 0 else BinaryNet1(512, args)
    model = model.cuda()
    x = torch.randn(64, 512).cuda()
    pred = model(x)
    print(pred.shape)