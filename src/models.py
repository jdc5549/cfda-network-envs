import torch
import torch.nn as nn
import torch.nn.functional as F


class SubActMultiCriticMlp(nn.Module):
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
        return x

class GCN_Encoder(torch.nn.Module):
    def __init__(self,embed_size,hidden_size,num_features,train,depth=2):
        super().__init__()
        from torch_geometric.nn import GCNConv,Linear
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.train = train
        self.conv1 = GCNConv(num_features, hidden_size)
        self.linear = Linear(self.hidden_size,self.embed_size)
        ref_hconv = GCNConv(hidden_size,hidden_size)
        self.hconvs = []
        for i in range(self.depth-1):
            self.hconvs.append(GCNConv(hidden_size,hidden_size))

    def forward(self,node_features,edge_index):
        node_features = node_features.to(torch.float32)
        x = self.conv1(node_features, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.train)
        for hc in self.hconvs:
            x = hc(x,edge_index)
            x = F.relu(x)
        x = self.linear(x)
        return(torch.transpose(torch.sigmoid(x),0,1))