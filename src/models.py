import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import ncr

class MLP_Critic(torch.nn.Module):
    def __init__(self,embed_size,hidden_size,num_nodes,p=2,num_node_features=0,num_mlp_layers=2,dropout_rate=0.5):
        super().__init__()
        self.dropout_rate = dropout_rate
        num_embeddings = ncr(num_nodes,p) #want a unique embedding for every pair of nodes
        self.act_embedding = nn.Embedding(num_embeddings,embed_size)
        #self.mlp_layers = nn.ModuleList([nn.Linear(2*embed_size,hidden_size)])# + num_node_features,hidden_size)])#[self.lin1,self.lin2]
        self.mlp_layers = nn.ModuleList([nn.Linear(2*embed_size + num_nodes,hidden_size)])#[self.lin1,self.lin2]
        num_mlp_layers = max([num_mlp_layers,1])
        for i in range(num_mlp_layers-1):
            self.mlp_layers.append(nn.Linear(hidden_size,hidden_size))
        # self.mlp_layers = self.mlp_layers.to('cuda:1')
        # for layer in self.mlp_layers:
        #     layer.to('cuda:1')
        self.output_layer = nn.Linear(hidden_size,num_nodes)
        self.sigmoid = nn.Sigmoid()

    def forward(self,actions,node_features=None):
        #GCN Encoding
        node_features = node_features.squeeze()
        a = actions[...,0]
        a_emb = self.act_embedding(a)
        d = actions[...,1]
        d_emb = self.act_embedding(d)
        a_emb = a_emb.to(node_features.dtype)
        d_emb = d_emb.to(node_features.dtype)

        if node_features.dtype == torch.float16:
            move = True
            a_emb = a_emb.to('cuda:2')
            d_emb = d_emb.to('cuda:2')
            node_features = node_features.to('cuda:2')
            x = torch.cat([a_emb,d_emb,node_features],-1)
            x = x.to('cuda:1')
        else:
            move = False
            x = torch.cat([a_emb,d_emb,node_features],-1)
        #MLP for Encoded vector

        for layer in self.mlp_layers:
            if move:
                layer.to('cuda:1').half()
            x = F.relu(layer(x))
            x = F.dropout(x,p=self.dropout_rate,training=self.training)
        if move:
            self.output_layer.to('cuda:3').half()
            x = x.to('cuda:3')
        x = self.output_layer(x)
        output = self.sigmoid(x.squeeze(-1))
        output = output.to('cuda:0')
        if move:
            for layer in self.mlp_layers:
                layer.to('cuda:0').to(torch.float32)
            self.output_layer.to('cuda:0').to(torch.float32)
        return output

class GCN_Critic(torch.nn.Module):
    def __init__(self,embed_size,hidden_size,num_nodes,num_conv_layers=3,num_mlp_layers=2,dropout=0.4):
        super().__init__()
        from torch_geometric.nn import GCNConv,Linear
        from torch_geometric.nn.pool import global_add_pool
        self.global_sum_pool = global_add_pool
        self.dropout = dropout
        # num_embeddings = ncr(num_nodes,2) #want a unique embedding for every pair of nodes
        # self.act_embedding = nn.Embedding(num_embeddings,num_nodes)
        num_conv_layers = max([1,num_conv_layers])
        if num_conv_layers == 1:
            conv1 = GCNConv(3,embed_size)
            self.convs = nn.ModuleList([conv1])
        elif num_conv_layers > 1:
            conv1 = GCNConv(3,hidden_size)
            self.convs = nn.ModuleList([conv1])
            for i in range(1,num_conv_layers-1):
                self.convs.append(GCNConv(hidden_size,hidden_size))
            self.convs.append(GCNConv(hidden_size,embed_size))
        self.mlp_layers = nn.ModuleList([Linear(embed_size,hidden_size)])#[self.lin1,self.lin2]
        num_mlp_layers = max([num_mlp_layers,1])
        for i in range(num_mlp_layers-1):
            self.mlp_layers.append(Linear(hidden_size,hidden_size))
        self.output_layer = Linear(hidden_size,num_nodes)
        self.sigmoid = nn.Sigmoid()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self,actions,node_features,edge_index):
        #GCN Encoding
        size = len(node_features)
        if size < 3:
            node_features = node_features.unsqueeze(1)
        batch_size,num_features,num_nodes = node_features.shape
        batch = torch.zeros(batch_size*num_nodes,dtype=torch.int64)
        batch = batch.to(node_features.device)
        for i in range(batch_size):
            batch[i*num_nodes:(i+1)*(num_nodes)] = i
        node_features = node_features.to(torch.float32).squeeze()
        if size == 2:
            print(actions.shape)
            x = torch.stack((actions[:,0],actions[:,1],node_features))
            x = x.reshape(batch_size*num_nodes,x.shape[2])
        else:    
            x = torch.stack((actions[:,:,0],actions[:,:,1],node_features),dim=2)
            x = x.reshape(batch_size*num_nodes,x.shape[2])
        edge_index_batches = edge_index[:]
        for i,b in enumerate(edge_index_batches):
            b += i*num_nodes
        edge_index_batches = edge_index_batches.reshape(2,batch_size*edge_index.shape[2])
        for conv in self.convs:
            x = F.relu(conv(x,edge_index_batches))
            x = F.dropout(x,p=self.dropout,training=self.training)
        # #MLP Layers
        # x = x.reshape(batch_size,num_nodes*x.shape[-1])
        # for layer in self.mlp_layers:
        #     x = F.relu(layer(x))            
        #     x = F.dropout(x, p=self.dropout, training=self.training)
        # #Pooling and MLP layers
        # if False:  # center pooling
        #     _, center_indices = np.unique(batch.cpu().numpy(), return_index=True)
        #     x_src = x[center_indices]
        #     x_dst = x[center_indices + 1]
        #     x = (x_src * x_dst)
        #     for layer in self.mlp_layers:
        #         x = F.relu(layer(x))
        #         x = F.dropout(x, p=self.dropout, training=self.training)
        # else:  # sum pooling
        x = self.global_sum_pool(x, batch)
        for layer in self.mlp_layers:
            x = F.relu(layer(x))            
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.output_layer(x)
        output = self.sigmoid(x)
        return output

class GIN_Critic(nn.Module):
    def __init__(self, embed_size, hidden_size, num_nodes, num_conv_layers=2, num_mlp_layers=1, dropout=0.4):
        super().__init__()
        from torch_geometric.nn import GINConv,Linear,global_sum_pool
        self.global_sum_pool = global_sum_pool

        self.dropout = dropout
        num_conv_layers = max([1, num_conv_layers])

        nn1 = nn.Sequential(nn.Linear(3, hidden_size), nn.ReLU(), nn.Linear(hidden_size, embed_size))
        self.convs = nn.ModuleList([GINConv(nn1)])

        for i in range(1, num_conv_layers):
            nn2 = nn.Sequential(nn.Linear(embed_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, embed_size))
            self.convs.append(GINConv(nn2))

        self.mlp_layers = nn.ModuleList([Linear(embed_size, hidden_size)])
        num_mlp_layers = max([num_mlp_layers, 1])
        for i in range(num_mlp_layers - 1):
            self.mlp_layers.append(Linear(hidden_size, hidden_size))

        self.output_layer = Linear(hidden_size, num_nodes)
        self.sigmoid = nn.Sigmoid()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, actions, node_features, edge_index):
        batch_size, num_features, num_nodes = node_features.shape
        batch = torch.zeros(batch_size * num_nodes, dtype=torch.int64)
        batch = batch.to(node_features.device)
        for i in range(batch_size):
            batch[i * num_nodes:(i + 1) * num_nodes] = i
        node_features = node_features.to(torch.float32).squeeze()
        x = torch.stack((actions[:, :, 0], actions[:, :, 1], node_features), dim=2)
        x = x.reshape(batch_size * num_nodes, x.shape[2])
        edge_index_batches = edge_index[:]
        for i, b in enumerate(edge_index_batches):
            b += i * num_nodes
        edge_index_batches = edge_index_batches.reshape(2, batch_size * edge_index.shape[2])
        for conv in self.convs:
            x = F.relu(conv(x, edge_index_batches))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.global_sum_pool(x, batch)
        for layer in self.mlp_layers:
            x = F.relu(layer(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.output_layer(x)
        output = self.sigmoid(x)
        return output

class GAT_Critic(torch.nn.Module):
    def __init__(self,embed_size,hidden_size,num_nodes,num_conv_layers=2,num_mlp_layers=2,dropout=0.6):
        super(GAT_Critic, self).__init__()
        from torch_geometric.nn import GATConv,Linear
        self.dropout = dropout
        self.conv1 = GATConv(3, hidden_size, heads=8, dropout=dropout)
        self.conv2 = GATConv(hidden_size*8, embed_size, heads=1, concat=False,dropout=dropout)
        self.mlp_layers = nn.ModuleList([Linear(num_nodes*embed_size,hidden_size)])#[self.lin1,self.lin2]
        num_mlp_layers = max([num_mlp_layers,1])
        for i in range(num_mlp_layers-1):
            self.mlp_layers.append(Linear(hidden_size,hidden_size))
        self.output_layer = Linear(hidden_size,num_nodes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, actions,node_features,edge_index):
        batch_size,num_features,num_nodes = node_features.shape
        batch = torch.zeros(batch_size*num_nodes,dtype=torch.int64)
        batch = batch.to(node_features.device)
        for i in range(batch_size):
            batch[i*num_nodes:(i+1)*(num_nodes)] = i
        node_features = node_features.to(torch.float32).squeeze()
        x = torch.stack((actions[:,:,0],actions[:,:,1],node_features),dim=2)
        x = x.reshape(batch_size*num_nodes,x.shape[2])
        edge_index_batches = edge_index[:]
        for i,b in enumerate(edge_index_batches):
            b += i*num_nodes
        edge_index_batches = edge_index_batches.reshape(2,batch_size*edge_index.shape[2])

        x = F.elu(self.conv1(x, edge_index_batches))
        x = torch.nn.functional.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv2(x, edge_index_batches))
        x = torch.nn.functional.dropout(x, p=0.6, training=self.training)
        x = x.reshape(batch_size,num_nodes*x.shape[-1])
        for layer in self.mlp_layers:
            x = F.relu(layer(x))            
            x = F.dropout(x, p=self.dropout/2, training=self.training)
        x = self.output_layer(x)
        return self.sigmoid(x)

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