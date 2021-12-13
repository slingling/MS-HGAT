# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 22:29:40 2021

@author: Ling Sun
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from graphConstruct import  get_EdgeAttention, get_NodeAttention, normalize

class HGATLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, transfer, concat=True, bias=False, edge = True):
        super(HGATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.edge = edge

        self.transfer = transfer

        if self.transfer:
            self.weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        else:
            self.register_parameter('weight', None)

        self.weight2 = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.weight3 = Parameter(torch.Tensor(self.out_features, self.out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)
        self.weight3.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, root_emb):
        
        if self.transfer:
            x = x.matmul(self.weight)
        else:
            x = x.matmul(self.weight2)
        
        if self.bias is not None:
            x = x + self.bias  
            
        #n2e_att = get_NodeAttention(x, adj.t(), root_emb)

        adjt = F.softmax(adj.T,dim = 1)
        #adj = normalize(adj)
        

        edge = torch.matmul(adjt, x)
        
        edge = F.dropout(edge, self.dropout, training=self.training)
        edge = F.relu(edge,inplace = False)

        e1 = edge.matmul(self.weight3)

        
        adj = F.softmax(adj,dim = 1)
        #adj = get_EdgeAttention(adj)

        node = torch.matmul(adj, e1)
        node = F.dropout(node, self.dropout, training=self.training)
        

        if self.concat:
            node = F.relu(node,inplace = False)
            
        if self.edge:
            edge = torch.matmul(adjt, node)        
            edge = F.dropout(edge, self.dropout, training=self.training)
            edge = F.relu(edge,inplace = False) 
            return node, edge
        else:
            return node

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
