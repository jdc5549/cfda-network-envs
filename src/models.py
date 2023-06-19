import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import ncr

class MLP_Critic(torch.nn.Module):
    def __init__(self,embed_size,hidden_size,output_size,num_node_features=0,num_mlp_layers=2,dropout_rate=0.2):
        super().__init__()
        self.dropout_rate = dropout_rate
        num_embeddings = ncr(output_size,2) #want a unique embedding for every pair of nodes
        self.act_embedding = nn.Embedding(num_embeddings,embed_size)
        self.mlp_layers = nn.ModuleList([nn.Linear(2*embed_size,hidden_size)])# + num_node_features,hidden_size)])#[self.lin1,self.lin2]
        num_mlp_layers = max([num_mlp_layers,1])
        for i in range(num_mlp_layers-1):
            self.mlp_layers.append(nn.Linear(hidden_size,hidden_size))
        self.output_layer = nn.Linear(hidden_size,output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self,actions,node_features=None):
        #GCN Encoding
        #node_features = node_features.to(torch.float32)
        a = actions[...,0]
        a_emb = self.act_embedding(a)
        d = actions[...,1]
        d_emb = self.act_embedding(d)
        x = torch.cat([a_emb,d_emb],-1)#,node_features],-1)
        #MLP for Encoded vector
        for layer in self.mlp_layers:
            x = F.relu(layer(x))
            x = F.dropout(x,p=self.dropout_rate,training=self.training)
        x = self.output_layer(x)
        output = self.sigmoid(x.squeeze(-1))
        return output

class GCN_Critic(torch.nn.Module):
    def __init__(self,embed_size,hidden_size,output_size,num_conv_layers=2,num_mlp_layers=2,dropout=0):
        super().__init__()
        from torch_geometric.nn import GCNConv,Linear
        self.dropout = dropout
        num_embeddings = ncr(output_size,2) #want a unique embedding for every pair of nodes
        self.act_embedding = nn.Embedding(num_embeddings,output_size)
        num_conv_layers = max((2,num_conv_layers))
        conv1 = GCNConv(3,hidden_size)
        self.convs = nn.ModuleList([conv1])
        for i in range(num_conv_layers-2):
            self.convs.append(GCNConv(hidden_size,hidden_size))
        self.convs.append(GCNConv(hidden_size,embed_size))
        self.mlp_layers = nn.ModuleList([Linear(embed_size,hidden_size)])#[self.lin1,self.lin2]
        num_mlp_layers = max([num_mlp_layers,1])
        for i in range(num_mlp_layers-1):
            self.mlp_layers.append(Linear(hidden_size,hidden_size))
        self.output_layer = Linear(hidden_size,1)
        self.sigmoid = nn.Sigmoid()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self,actions,node_features,edge_index):
        #GCN Encoding
        batch_size,num_features,num_nodes = node_features.shape
        batch = torch.zeros(batch_size*num_nodes,dtype=torch.int64)
        for i in range(batch_size):
            batch[i*num_nodes:(i+1)*(num_nodes)] = i
        node_features = node_features.to(torch.float32).squeeze()
        a = actions[...,0]
        a_emb = self.act_embedding(a)
        d = actions[...,1]
        d_emb = self.act_embedding(d)
        x = torch.stack((a_emb,d_emb,node_features),dim=2)
        #x = torch.cat([a_emb,d_emb,node_features],-1)
        x = x.reshape(batch_size*num_nodes,x.shape[2])
        edge_index_batches = edge_index[:]
        for i,b in enumerate(edge_index_batches):
            b += i*num_nodes
        edge_index_batches = edge_index_batches.reshape(2,batch_size*edge_index.shape[2])
        for conv in self.convs[:-1]:
            x = F.relu(conv(x,edge_index_batches))
            x = F.dropout(x,p=self.dropout,training=self.training)
        x = self.convs[-1](x,edge_index_batches)
        x = x.reshape(batch_size,num_nodes,-1)
        # if True:  # center pooling
        #     _, center_indices = np.unique(batch.cpu().numpy(), return_index=True)
        #     x_src = x[center_indices]
        #     x_dst = x[center_indices + 1]
        #     x = (x_src * x_dst)
        #     #x = F.relu(self.lin1(x))
        #     #x = F.dropout(x, p=self.dropout, training=self.training)
        #     x = self.lin2(x)
        # else:  # sum pooling
        #     from torch_geometric.nn.pool import global_add_pool
        #     x = global_add_pool(x, batch)
        #     x = F.relu(self.lin1(x))
        #     x = F.dropout(x, p=self.dropout, training=self.training)
        #     x = self.lin2(x)
        #MLP for Encoded vector
        for layer in self.mlp_layers:
            x = F.relu(layer(x))
        x = self.output_layer(x).squeeze(-1)
        output = self.sigmoid(x)
        return output

#Exploiter Models
class Exploiter_MLP_Critic(torch.nn.Module):
    def __init__(self,embed_size,hidden_size,num_nodes,num_mlp_layers=2,dropout_rate=0.2):
        super().__init__()
        self.dropout_rate = dropout_rate
        num_embeddings = ncr(num_nodes,2) #want a unique embedding for every pair of nodes
        self.act_embedding = nn.Embedding(num_embeddings,embed_size)
        self.mlp_layers = nn.ModuleList([nn.Linear(embed_size + num_node_features,hidden_size)])#[self.lin1,self.lin2]
        num_mlp_layers = max([num_mlp_layers,1])
        for i in range(num_mlp_layers-1):
            self.mlp_layers.append(nn.Linear(hidden_size,hidden_size))
        self.output_layer = nn.Linear(hidden_size,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,action):
        #GCN Encoding
        a = action[...,0]
        x = self.act_embedding(a)

        #MLP for Encoded vector
        for layer in self.mlp_layers:
            x = F.relu(layer(x))
            x = F.dropout(x,p=self.dropout_rate,training=self.training)
        x = self.output_layer(x)
        output = x.squeeze(-1)
        return output