import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP_Critic(nn.Module):
    def __init__(self,obs_embed_size,act_embed_size,output_size,hidden_size=64,depth=2,drop_rate=0):
        super(SubActMultiCriticMlp, self).__init__()
        hidden = hidden_size
        self.dropout = nn.Dropout(drop_rate)
        self.depth = depth if depth >= 1 else 1
        #self.obs_input_layer = nn.Linear(obs_embed_size,hidden)
        #self.p1_input_layer = nn.Linear(act_embed_size,hidden)
        #self.p2_input_layer = nn.Linear(act_embed_size,hidden)
        #self.all_hidden_layer = nn.Linear(3*hidden,hidden)
        self.input_layer = nn.Linear(obs_embed_size + 2*act_embed_size,hidden)
        self.hlayers = nn.ModuleList()
        for i in range(self.depth-1):
            self.hlayers.append(nn.Linear(hidden,hidden))
        #num_resnet_blocks = int((self.depth-2)/2)
        #self.resnet_blocks = ResNet1D(num_resnet_blocks,hidden_size)
        self.output_layer = nn.Linear(hidden,output_size)
        self.sigmoid = nn.Sigmoid()

    def reset_parameters(self):
        for lay in [self.p1_hidden_layer,self.p2_hidden_layer]:
            if isinstance(lay, nn.Linear):
                lay.weight.data.uniform_(*hidden_init(lay))
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)
         
    def forward(self,obs,p1_act,p2_act):
        obs = obs.to(torch.float32)
        p1_act = p1_act.to(torch.float32)
        p2_act = p2_act.to(torch.float32)
        #obs_h = self.dropout(self.obs_input_layer(obs))
        #p1_h = self.dropout(self.p1_input_layer(p1_act))
        #p2_h = self.dropout(self.p2_input_layer(p2_act))
        # Concat obs and actions
        #x = torch.cat((obs_h,p1_h,p2_h),-1)
        x = torch.cat((obs,p1_act,p2_act),-1)
        #x = F.relu(x)
        #x = self.all_hidden_layer(x)
        x = self.input_layer(x)
        x = self.dropout(F.relu(x))
        for layer in self.hlayers:
            x = layer(x)
            x = self.dropout(x)
            x = F.relu(x)   
        #x = self.resnet_blocks(x)     
        x = self.output_layer(x)
        #print(x)
        outputs = self.sigmoid(x)
        return outputs

class GCN_Critic(torch.nn.Module):
    def __init__(self,embed_size,hidden_size,num_features,output_size,num_conv_layers=1,num_mlp_layers=2,max_z=2000,dropout=0.25):
        super().__init__()
        from torch_geometric.nn import GCNConv,Linear
        self.dropout = dropout
        self.max_z = max_z
        self.z_embedding = nn.Embedding(self.max_z,hidden_size-num_features)
        #initial_channels = num_features + hidden_size
        self.conv1 = GCNConv(hidden_size, hidden_size)
        self.convs = [self.conv1]
        for i in range(num_conv_layers-1):
            self.convs.append(GCNConv(hidden_size,hidden_size))
        #self.lin1 = Linear(hidden_size,hidden_size)
        self.lin2 = Linear(hidden_size,embed_size)
        self.mlp_layers = [Linear(embed_size,hidden_size)]#[self.lin1,self.lin2]
        num_mlp_layers = max([num_mlp_layers,2])
        for i in range(num_mlp_layers-2):
            self.mlp_layers.append(Linear(hidden_size,hidden_size))
        self.output_layer = Linear(hidden_size,1)
        self.sigmoid = nn.Sigmoid()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self,z,node_features,edge_index):
        #GCN Encoding
        batch_size = node_features.shape[0]
        num_nodes = node_features.shape[2]
        batch = torch.zeros(batch_size*num_nodes,dtype=torch.int64)
        for i in range(batch_size):
            batch[i*num_nodes:(i+1)*(num_nodes)] = i
        node_features = node_features.to(torch.float32).permute(0,2,1)
        z_emb = self.z_embedding(z)
        x = torch.cat([z_emb,node_features],2)
        x = x.reshape((batch_size*num_nodes,x.shape[2]))
        edge_index_batches = edge_index[:]
        for i,b in enumerate(edge_index_batches):
            b += i*num_nodes
        edge_index_batches = edge_index_batches.reshape(2,batch_size*edge_index.shape[2])
        for conv in self.convs[:-1]:
            x = F.relu(conv(x,edge_index_batches))
            x = F.dropout(x,p=self.dropout,training=self.training)
        x = self.convs[-1](x,edge_index_batches)
        x = x.reshape((batch_size,num_nodes,-1))
        print(x.shape)
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
        print(x.shape)
        outputs = self.output_layer(x)
        print(x.shape)
        x = self.sigmoid(x.squeeze(-1))
        print(x.shape)
        exit()
        return outputs