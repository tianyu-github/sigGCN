#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 21:15:43 2020

@author: tianyu
"""
import torch
#from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import sys
sys.path.insert(0, 'lib/')


if torch.cuda.is_available():
    print('cuda available')
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
    torch.cuda.manual_seed(1)
else:
    print('cuda not available')
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor
    torch.manual_seed(1)

from coarsening import lmax_L
from coarsening import rescale_L
#from utilsdata import sparse_mx_to_torch_sparse_tensor

class my_sparse_mm(torch.autograd.Function):
    """
    Implementation of a new autograd function for sparse variables,
    called "my_sparse_mm", by subclassing torch.autograd.Function
    and implementing the forward and backward passes.
    """

    def forward(self, W, x):  # W is SPARSE
        self.save_for_backward(W, x)
        y = torch.mm(W, x)
        return y

    def backward(self, grad_output):
        W, x = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input_dL_dW = torch.mm(grad_input, x.t())
        grad_input_dL_dx = torch.mm(W.t(), grad_input )
        return grad_input_dL_dW, grad_input_dL_dx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    
#    L = L.tocoo()
#    indices = np.column_stack((L.row, L.col)).T
#    indices = indices.astype(np.int64)
#    indices = torch.from_numpy(indices)
#    indices = indices.type(torch.LongTensor)
#    L_data = L.data.astype(np.float32)
#    L_data = torch.from_numpy(L_data)
#    L_data = L_data.type(torch.FloatTensor)
#    L = torch.sparse.FloatTensor(indices, L_data, torch.Size(L.shape))
    
    return torch.sparse.FloatTensor(indices, values, shape)#.requires_grad_()


#########################################################################################################
class Graph_GCN(nn.Module):

    def __init__(self, net_parameters):

        print('Graph ConvNet: GCN')

        super(Graph_GCN, self).__init__()

        # parameters
        D_g, D_nn, CL1_F, CL1_K, CL2_F, CL2_K, CNN1_F, CNN1_K, FC1_F, FC2_F,NN_FC1, NN_FC2, out_dim, flag = net_parameters
        self.flag = flag
        self.in_dim = D_g
        self.out_dim = out_dim
        self.FC2_F = FC2_F
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gene = D_nn
        self.initScale = initScale = 6
        self.poolsize = 8
        FC1Fin = CL1_F*(D_g//self.poolsize)
        self.FC1Fin = FC1Fin
        self.CL1_K = CL1_K; self.CL1_F = CL1_F; 
        
        # Feature_H, Feature_W = (Input_Height - filter_H + 2P)/S + 1, (Input_Width - filter_W + 2P)/S + 1
        height = int(np.ceil(np.sqrt(int(D_nn))))
        FC2Fin = int(CNN1_F * (height//2) ** 2)
        self.FC2Fin = FC2Fin;
        
        # graph CL1
        self.cl1 = nn.Linear(CL1_K, CL1_F)
#        # graph CL2
#        self.cl2 = nn.Linear(CL2_K*CL1_F, CL2_F)
#        #FC gcnpure
#        self.fc_gcnpure = nn.Linear(FC1Fin, self.out_dim)
        # FC 1
        self.fc1 = nn.Linear(FC1Fin, FC1_F)
        # FC 2
        if self.FC2_F == 0:
            FC2_F = self.num_gene
            print('---------',FC2_F)
        self.fc2 = nn.Linear(FC1_F, FC2_F)
        # FC 3
        self.fc3 = nn.Linear(FC2_F, D_g)
        # CNN_FC1
        self.cnn_fc1 = nn.Linear(FC2Fin, FC1_F)
        #FC_concat with CNN
        Fin = FC1Fin + FC2Fin; Fout = self.out_dim;
        self.FC_concat = nn.Linear(Fin, self.out_dim)             
        #FC_sum2 with NN
        Fin = FC1_F + NN_FC2; Fout = self.out_dim;
        self.FC_sum2 = nn.Linear(Fin, Fout)                  
        #FC_sum1 with CNN
        Fin = FC1_F + FC1_F; Fout = self.out_dim;
        self.FC_sum1 = nn.Linear(Fin, Fout)             
        # NN_FC1
        self.nn_fc1 = nn.Linear(self.in_dim, NN_FC1)
        # NN_FC2
        self.nn_fc2 = nn.Linear(NN_FC1, NN_FC2)
        # NN_FC3_decode
        self.nn_fc3 = nn.Linear(NN_FC2, NN_FC1)
        # NN_FC4_decode
        Fin = NN_FC2; Fout = self.in_dim;
        self.nn_fc4 = nn.Linear(Fin, Fout)        

        
        # nb of parameters
        nb_param = CL1_K* CL1_F + CL1_F          # CL1
#        nb_param += CL2_K* CL1_F* CL2_F + CL2_F  # CL2
        nb_param += FC1Fin* FC1_F + FC1_F        # FC1
#        nb_param += FC1_F* FC2_F + FC2_F         # FC2
        print('nb of parameters=',nb_param,'\n')


    def init_weights(self, W, Fin, Fout):

        scale = np.sqrt( self.initScale / (Fin+Fout) )
        W.uniform_(-scale, scale)
        
        return W


    def graph_conv_cheby(self, x, cl,L, Fout, K):

        # parameters
        # B = batch size
        # V = nb vertices
        # Fin = nb input features
        # Fout = nb output features
        # K = Chebyshev order & support size
        B, V, Fin = x.size(); B, V, Fin = int(B), int(V), int(Fin)

        # rescale Laplacian
        lmax = lmax_L(L)
        L = rescale_L(L, lmax)

        # convert scipy sparse matric L to pytorch
        L = sparse_mx_to_torch_sparse_tensor(L)
        if torch.cuda.is_available():
            L = L.cuda()

        # transform to Chebyshev basis
        x0 = x.permute(1,2,0).contiguous()  # V x Fin x B
        x0 = x0.view([V, Fin*B])            # V x Fin*B
        x = x0.unsqueeze(0)                 # 1 x V x Fin*B

        def concat(x, x_):
            x_ = x_.unsqueeze(0)            # 1 x V x Fin*B
            return torch.cat((x, x_), 0)    # K x V x Fin*B

        if K > 1:
            x1 = my_sparse_mm()(L,x0)              # V x Fin*B
            x = torch.cat((x, x1.unsqueeze(0)),0)  # 2 x V x Fin*B
        for k in range(2, K):
            x2 = 2 * my_sparse_mm()(L,x1) - x0
            x = torch.cat((x, x2.unsqueeze(0)),0)  # M x Fin*B --> K x V x Fin*B
            x0, x1 = x1, x2

        x = x.view([K, V, Fin, B])           # K x V x Fin x B
        x = x.permute(3,1,2,0).contiguous()  # B x V x Fin x K
        x = x.view([B*V, Fin*K])             # B*V x Fin*K

        # Compose linearly Fin features to get Fout features
        x = cl(x)                            # B*V x Fout
        x = x.view([B, V, Fout])             # B x V x Fout

        return x


    # Max pooling of size p. Must be a power of 2.
    def graph_max_pool(self, x, p):
        if p > 1:
            x = x.permute(0,2,1).contiguous()  # x = B x F x V
            x = nn.MaxPool1d(p)(x)             # B x F x V/p
            x = x.permute(0,2,1).contiguous()  # x = B x V/p x F
            return x
        else:
            return x


    def forward(self, x_in, d, L):
        
        x = x_in#[:,:self.num_gene]
        x_nn = x_in#[:,self.num_gene:]
        #print('111',x_in.shape, x.shape, x_nn.shape)
        x = x.unsqueeze(2) # B x V x Fin=1
        x = self.graph_conv_cheby(x, self.cl1, L[0], self.CL1_F, self.CL1_K)

        x = F.relu(x)
        x = self.graph_max_pool(x, self.poolsize)

        
        # flatten()
        x = x.view(-1, self.FC1Fin)

        ##############################################
        ##                  GCNPURE                 ##
        ##############################################        
        if self.flag == 'gcnpure':
            x = self.fc_gcnpure(x)
            x = F.log_softmax(x) 
            return x

        
        ##############################################
        ##                  GAE_re                  ##
        ##############################################
        x_reAdj = 0 #torch.stack([F.sigmoid(torch.mm(z_i, z_i.t())) for z_i in x_reAdj])

        
        ##############################################
        ##                  GAE                     ##
        ##############################################
        #if self.flag == 'gae':
        x = self.fc1(x)
        x = F.relu(x)
        x_hidden_gae = x

        x_decode_gae = self.fc2(x_hidden_gae)
#        x_decode_gae = F.relu(x_decode_gae)
        if self.FC2_F != 0:                
            x_decode_gae = F.relu(x_decode_gae)
            x_decode_gae  = nn.Dropout(d)(x_decode_gae)            
            x_decode_gae = self.fc3(x_decode_gae)            

        


        ##############################################
        ##                  GCNCNN                  ##
        ##############################################
        
      
        # NN
        x_nn = self.nn_fc1(x_nn) # B x V
        x_nn = F.relu(x_nn)
        x_nn = self.nn_fc2(x_nn)
        x_nn = F.relu(x_nn)

#        x_hidden_ae = x_nn
#        x_decode_ae = self.nn_fc3(x_hidden_ae)
#        x_decode_ae = F.relu(x_decode_ae)
#        x_decode_ae = self.nn_fc4(x_hidden_ae)
#        x_decode_ae = F.relu(x_decode_ae)       

        # concatenate layer  
        x = torch.cat((x_hidden_gae, x_nn),1) 
#        x_decode_gae = self.fc2(x)        
        x = self.FC_sum2(x)

        x = F.log_softmax(x)        


        return x_decode_gae, x_hidden_gae, x, x_reAdj


    def loss(self, y1, y_target1,y2, y_target2,l2_regularization):
        if self.flag == 'gcnpure':
            loss1 = 0
            loss2 = nn.NLLLoss()(y2,y_target2)
        
        else:
       
            loss1 = nn.MSELoss()(y1, y_target1)
#            loss1 = F.binary_cross_entropy_with_logits(y1, y_target1)
            #loss2 = nn.CrossEntropyLoss()(y2,y_target2)
            loss2 = nn.NLLLoss()(y2,y_target2)
            

        loss = 1 * loss1 + 1 * loss2 
        #print(loss1,loss2)
        l2_loss = 0.0
        for param in self.parameters():
            data = param* param
            l2_loss += data.sum()
            #print('------------', l2_loss)

        loss += 0.2* l2_regularization* l2_loss

        return loss

#
#    def update(self, lr):
#
#        update = torch.optim.SGD( self.parameters(), lr=lr, momentum=0.9 )
#
#        return update
#
#
#    def update_learning_rate(self, optimizer, lr):
#
#        for param_group in optimizer.param_groups:
#            param_group['lr'] = lr
#
#        return optimizer
#
#
#    def evaluation(self, y_predicted, test_l):
#
#        _, class_predicted = torch.max(y_predicted.data, 1)
#        return 100.0* (class_predicted == test_l).sum()/ y_predicted.size(0)
#########################################################################################################
class Graph_GCNorg(nn.Module):

    def __init__(self, net_parameters):

        print('Graph ConvNet: GCN')

        super(Graph_GCNorg, self).__init__()

        # parameters
        D, CL1_F, CL1_K, CNN1_F, CNN1_K, FC1_F, FC2_F,out_dim, flag = net_parameters
        self.flag = flag
        self.in_dim = D
        self.out_dim = out_dim
        self.FC2_F = FC2_F
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
#        FC1Fin = CL2_F*(D//16)
        FC1Fin = CL1_F*(D//4)
        # Feature_H, Feature_W = (Input_Height - filter_H + 2P)/S + 1, (Input_Width - filter_W + 2P)/S + 1
        height = int(np.ceil(np.sqrt(int(D))))
        FC2Fin = int(CNN1_F * (height//2) ** 2)


        # graph CL1
        self.cl1 = nn.Linear(CL1_K, CL1_F)
        Fin = CL1_K; Fout = CL1_F;
        scale = np.sqrt( 2.0/ (Fin+Fout) )
        self.cl1.weight.data.uniform_(-scale, scale)
        self.cl1.bias.data.fill_(0.0)
        self.CL1_K = CL1_K; self.CL1_F = CL1_F;

        ##CNN
#        self.conv1 = nn.Sequential( # (1,70,70))
#                     nn.Conv2d(in_channels = 1, # 输入维度，rgb为3
#                               out_channels = CNN1_F,  #输出filter个数
#                               kernel_size= CNN1_K,
#                               stride = 1,  # 1为默认值
#                               padding = 1  # 0为默认值; 若与原尺寸相同，if stride=1,padding=(kernnalsize-1)/2 
#                               ), #-->(32,68,68) 
#                     nn.ReLU(),
##                     nn.BatchNorm2d(32),
#                     nn.MaxPool2d(2,1) #-->(32,34,34) 13
#                     )
#        # CNN 1
        self.conv1 = nn.Conv2d(in_channels = 1, # 输入维度，rgb为3
                               out_channels = CNN1_F,  #输出filter个数
                               kernel_size= CNN1_K, 
                               padding = (2,2))
        
        Fin = CNN1_K**2; Fout = CNN1_F;
        scale = np.sqrt( 2.0/ (Fin+Fout) )
        self.conv1.weight.data.uniform_(-scale, scale)
        self.conv1.bias.data.fill_(0.0) 
             
        self.pool = nn.MaxPool2d(2)            

        # FC1
        self.fc1 = nn.Linear(FC1Fin, FC1_F)
        Fin = FC1Fin; Fout = FC1_F;
        scale = np.sqrt( 2.0/ (Fin+Fout) )
        self.fc1.weight.data.uniform_(-scale, scale)
        self.fc1.bias.data.fill_(0.0)
        self.FC1Fin = FC1Fin


        # FC2
        if self.FC2_F == 0:
            FC2_F = D
            print('---------',FC2_F)
        self.fc2 = nn.Linear(FC1_F, FC2_F)
        Fin = FC1_F; Fout = FC2_F;
        scale = np.sqrt( 2.0/ (Fin+Fout) )
        self.fc2.weight.data.uniform_(-scale, scale)
        self.fc2.bias.data.fill_(0.0)


        # FC3
        self.fc3 = nn.Linear(FC2_F, D)
        Fin = FC2_F; Fout = D;
        scale = np.sqrt( 2.0/ (Fin+Fout) )
        self.fc3.weight.data.uniform_(-scale, scale)
        self.fc3.bias.data.fill_(0.0)

        # CNN_FC1
        self.cnn_fc1 = nn.Linear(FC2Fin, FC1_F)
        Fin = FC2Fin; Fout = FC1_F;
        scale = np.sqrt( 2.0/ (Fin+Fout) )
        self.cnn_fc1.weight.data.uniform_(-scale, scale)
        self.cnn_fc1.bias.data.fill_(0.0)
        self.FC2Fin = FC2Fin
        
        #FC_concat with CNN
        Fin = FC1Fin + FC2Fin; Fout = self.out_dim;
        print('**Fin4FC_concat',Fin)
        self.FC_concat = nn.Linear(Fin, self.out_dim)     #        
        scale = np.sqrt( 2.0/ (Fin+Fout) )
        self.FC_concat.weight.data.uniform_(-scale, scale)
        self.FC_concat.bias.data.fill_(0.0)

        #FC_sum with CNN
        Fin = FC1_F*2; Fout = self.out_dim;
        print('**Fin4FC_concat',Fin)
        self.FC_sum = nn.Linear(Fin, self.out_dim)     #        
        scale = np.sqrt( 2.0/ (Fin+Fout) )
        self.FC_sum.weight.data.uniform_(-scale, scale)
        self.FC_sum.bias.data.fill_(0.0)


        # nb of parameters
        nb_param = CL1_K* CL1_F + CL1_F          # CL1
#        nb_param += CL2_K* CL1_F* CL2_F + CL2_F  # CL2
        nb_param += FC1Fin* FC1_F + FC1_F        # FC1
#        nb_param += FC1_F* FC2_F + FC2_F         # FC2
        print('nb of parameters=',nb_param,'\n')


    def init_weights(self, W, Fin, Fout):

        scale = np.sqrt( 2.0/ (Fin+Fout) )
        W.uniform_(-scale, scale)

        return W


    def graph_conv_cheby(self, x, cl,L, lmax, Fout, K):

        # parameters
        # B = batch size
        # V = nb vertices
        # Fin = nb input features
        # Fout = nb output features
        # K = Chebyshev order & support size
        B, V, Fin = x.size(); B, V, Fin = int(B), int(V), int(Fin)

        # rescale Laplacian
        lmax = lmax_L(L)
        L = rescale_L(L, lmax)


        # convert scipy sparse matric L to pytorch
        L = sparse_mx_to_torch_sparse_tensor(L)
        if torch.cuda.is_available():
            L = L.cuda()

        # transform to Chebyshev basis
        x0 = x.permute(1,2,0).contiguous()  # V x Fin x B
        x0 = x0.view([V, Fin*B])            # V x Fin*B
        x = x0.unsqueeze(0)                 # 1 x V x Fin*B

        def concat(x, x_):
            x_ = x_.unsqueeze(0)            # 1 x V x Fin*B
            return torch.cat((x, x_), 0)    # K x V x Fin*B

        if K > 1:
            x1 = my_sparse_mm()(L,x0)              # V x Fin*B
            x = torch.cat((x, x1.unsqueeze(0)),0)  # 2 x V x Fin*B
        for k in range(2, K):
            x2 = 2 * my_sparse_mm()(L,x1) - x0
            x = torch.cat((x, x2.unsqueeze(0)),0)  # M x Fin*B --> K x V x Fin*B
            x0, x1 = x1, x2

        x = x.view([K, V, Fin, B])           # K x V x Fin x B
        x = x.permute(3,1,2,0).contiguous()  # B x V x Fin x K
        x = x.view([B*V, Fin*K])             # B*V x Fin*K

        # Compose linearly Fin features to get Fout features
        x = cl(x)                            # B*V x Fout
        x = x.view([B, V, Fout])             # B x V x Fout

        return x


    # Max pooling of size p. Must be a power of 2.
    def graph_max_pool(self, x, p):
        if p > 1:
            x = x.permute(0,2,1).contiguous()  # x = B x F x V
            x = nn.MaxPool1d(p)(x)             # B x F x V/p
            x = x.permute(0,2,1).contiguous()  # x = B x V/p x F
            return x
        else:
            return x


    def forward(self, x, d, L, lmax):

        x_cnn= x        # CNN copy input,shape: B x V
        
        x = x.unsqueeze(2) # B x V x Fin=1
        x = self.graph_conv_cheby(x, self.cl1,L[0], lmax[0], self.CL1_F, self.CL1_K)
        x = F.relu(x)
        x = self.graph_max_pool(x, 4)
        #print('x.shape after gcn:',x.shape) #torch.Size([64, 224, 4])
        
        # flatten()
        x = x.view(-1, self.FC1Fin)
        #print('x.shape',x.shape)  #x.shape torch.Size([64, 896])
        
        ##############################################
        ##                  GAE                     ##
        ##############################################
        if self.flag == 'gae':
            x_gae = self.fc1(x)
            x_gae = F.relu(x_gae)
            x_hidden = x_gae
            
            x_gae  = nn.Dropout(d)(x_gae)            
            x_gae = self.fc2(x_gae)
            if self.FC2_F != 0:                
                x_gae = F.relu(x_gae)
                x_gae  = nn.Dropout(d)(x_gae)            
                x_gae = self.fc3(x_gae)            
        
            return x_gae, x_hidden
        
        ##############################################
        ##                  GCNPURE                 ##
        ##############################################        
        elif self.flag == 'gcnpure':
            x = self.fc1(x)

        ##############################################
        ##                  GCNCNN                  ##
        ##############################################
        elif self.flag == 'gcncnn':
            ## cnn
            Bsize, Fsize = x_cnn.shape
            height = width = int(np.ceil(np.sqrt(int(Fsize))))
            #print('cnn:height and width', Bsize, Fsize, height, width) cnn:height and width 64 896 30 30
            padzeros = torch.zeros((Bsize, height * width - Fsize), device= self.device)
            x_cnn = torch.cat((x_cnn, padzeros), 1)
            
            x_cnn = x_cnn.view(x_cnn.size(0), height, width)
            x_cnn = x_cnn.unsqueeze(1)            
            #print x_cnn: torch.Size([64, 1, 30, 30])
            
            x_cnn = self.conv1(x_cnn)
            x_cnn = F.relu(x_cnn)
    
            x_cnn = self.pool(x_cnn)
            
            x_cnn = x_cnn.permute(0,3,2,1).contiguous()
            
            #print('x_cnn.shape after pool:',x_cnn.shape)
            x_cnn = x_cnn.view(x_cnn.shape[0],-1) # x_cnn.shape torch.Size([64, 32, 15, 15])
                 
            
            x = torch.cat((x, x_cnn), 1) 
            x = self.FC_concat(x)
            ###print(x_hidden.shape, x_cnn_f1.shape) #64*64, 64*64
           


        return x


    def loss(self, y2, y_target2, l2_regularization):

        if self.flag == 'gae':
            loss = nn.MSELoss()(y2, y_target2)
        else:
            loss = nn.CrossEntropyLoss()(y2,y_target2)
        #loss = 1 * loss1 + 1 * loss2
        #print(loss1,loss2)
        l2_loss = 0.0
        for param in self.parameters():
            data = param* param
            l2_loss += data.sum()
            #print('------------', l2_loss)

        loss += 0.5* l2_regularization* l2_loss

        return loss

