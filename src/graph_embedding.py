import networkx as nx
import networkx.algorithms.centrality as central
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

def heuristic_action_embedding(net,bounds,action_pair):
    num_nodes = net.number_of_nodes()
    n0 = sorted(net.nodes())[0]
    metrics = []
    degree_centrality = sum([central.degree_centrality(net)[a+n0] for a in action_pair])
    min_c = bounds[0][0]
    max_c = bounds[0][1]
    degree_centrality = 2*(c-min_c)/(max_c-min_c)-1 if (max_c-min_c) != 0 else 0
    metrics.append(torch.tensor(degree_centralities))

    closeness_centrality = sum([central.closeness_centrality(net)[a+n0] for a in action_pair])
    min_c = bounds[1][0]
    max_c = bounds[1][1]
    closeness_centrality = 2*(c-min_c)/(max_c-min_c)-1 if (max_c-min_c) != 0 else 0
    metrics.append(torch.tensor(closeness_centrality))

    betweenness_centrality = sum([central.betweenness_centrality(net)[a+n0] for a in action_pair])
    min_c = bounds[2][0]
    max_c = bounds[2][1]
    betweenness_centrality = 2*(c-min_c)/(max_c-min_c)-1 if (max_c-min_c) != 0 else 0
    metrics.append(torch.tensor(betweenness_centrality))

    eigenvector_centrality = sum([central.eigenvector_centrality(net)[a+n0] for a in action_pair])
    min_c = bounds[3][0]
    max_c = bounds[3][1]
    eigenvector_centrality = 2*(c-min_c)/(max_c-min_c)-1 if (max_c-min_c) != 0 else 0
    metrics.append(torch.tensor(eigenvector_centrality))

    node_distance = nx.shortest_path_length(net,source=n0+action_pair[0],target=n0+action_pair[1])
    min_c = bounds[4][0]
    max_c = bounds[4][1]
    node_distance = 2*(c-min_c)/(max_c-min_c)-1 if (max_c-min_c) != 0 else 0
    metrics.append(torch.tensor(node_distance))

    common_neighbors = nx.common_neighbors(net,n0+action_pair[0],n0+action_pair[1])
    min_c = bounds[5][0]
    max_c = bounds[5][1]
    common_neighbors = 2*(c-min_c)/(max_c-min_c)-1 if (max_c-min_c) != 0 else 0
    metrics.append(torch.tensor(common_neighbors))


def get_featurized_obs(batch_obs,embed_model=None):
    #tic = time.perf_counter()
    feat_obs_list = []
    for obs in batch_obs:
        if obs.shape[0] > obs.shape[1]:
            adj_mat = obs[:-1]
            thresh = obs[-1]
        else:
            adj_mat = obs
        net = nx.from_numpy_array(adj_mat)
        n0 = sorted(net.nodes())[0] #recognize 0 vs 1 indexing of node names
        num_nodes = net.number_of_nodes()
        nodes = [i for i in range(num_nodes)]
        max_node = max(nodes)
        min_node = min(nodes)
        nodes = [2*(node-min_node)/(max_node-min_node)-1 if (max_node-min_node) != 0 else 0 for node in nodes]
        #toc = time.perf_counter()
        #print('Node Names: ', toc - tic)
        feat_t = torch.tensor(nodes)
        if obs.shape[0] > obs.shape[1]:
            #tic = time.perf_counter()
            max_t = max(thresh)
            min_t= min(thresh)
            norm_thresh = [2*(t-min_t)/(max_t-min_t)-1 if (max_t-min_t) != 0 else 0 for t in thresh]
            #toc = time.perf_counter()
            #print('thresholds: ', toc - tic)=
            # A = np.asarray(nx.adjacency_matrix(self.net).todense())
            # for row in A:
            #     metrics.append(row.tolist())
        ######################################Graph Embedding#######################################
        norm_thresh = torch.tensor(norm_thresh)
        if embed_model is None:
            feat_t = torch.stack((feat_t,norm_thresh))
            embed = heuristic_feature_embedding(net) #TODO will need to account for CNs here
        else:
            feat_t = feat_t.unsqueeze(0)
            from torch_geometric.utils import from_networkx
            ptgeo_data = from_networkx(net)
            embed = embed_model(norm_thresh.reshape([-1,1]),ptgeo_data.edge_index)
        feat_obs = torch.cat((feat_t,embed),dim=0)
        feat_obs_list.append(feat_obs)
    batch_obs_t = torch.stack(feat_obs_list)
    batch_obs_t = torch.permute(batch_obs_t,(0,2,1))
    #toc = time.perf_counter()
    #print('Embedding time: ', toc-tic)
    return batch_obs_t

def heuristic_feature_embedding(net):
    num_nodes = net.number_of_nodes()
    n0 = sorted(net.nodes())[0]
    metrics = []
    degree_centralities = [central.degree_centrality(net)[i+n0] for i in range(num_nodes)]
    max_c = max(degree_centralities)
    min_c = min(degree_centralities)
    degree_centralities = [2*(c-min_c)/(max_c-min_c)-1 if (max_c-min_c) != 0 else 0 for c in degree_centralities]
    metrics.append(torch.tensor(degree_centralities))
    #toc = time.perf_counter()
    #print('Degree: ', toc - tic)

    #tic = time.perf_counter()
    # closeness_centralities = [central.closeness_centrality(net)[i+n0] for i in range(num_nodes)]
    # max_c = max(closeness_centralities)
    # min_c = min(closeness_centralities)
    # closeness_centralities = [2*(c-min_c)/(max_c-min_c)-1 if (max_c-min_c) != 0 else 0 for c in closeness_centralities]
    # metrics.append(torch.tensor(closeness_centralities))
    #toc = time.perf_counter()
    #print('Closeness: ', toc - tic)

    #tic = time.perf_counter()
    # harmonic_centralities = [central.harmonic_centrality(net)[i+n0] for i in range(num_nodes)]
    # max_c = max(harmonic_centralities)
    # min_c = min(harmonic_centralities)
    # harmonic_centralities = [2*(c-min_c)/(max_c-min_c)-1 if (max_c-min_c) != 0 else 0 for c in harmonic_centralities]
    # metrics.append(torch.tensor(harmonic_centralities))
    #toc = time.perf_counter()
    #print('Harmonic: ', toc - tic)

    #tic = time.perf_counter()
    betweenness_centralities = [central.betweenness_centrality(net)[i+n0] for i in range(num_nodes)]
    max_c = max(betweenness_centralities)
    min_c = min(betweenness_centralities)
    betweenness_centralities = [2*(c-min_c)/(max_c-min_c)-1 if (max_c-min_c) != 0 else 0 for c in betweenness_centralities]
    metrics.append(torch.tensor(betweenness_centralities))
    #toc = time.perf_counter()
    #print('Betweeness: ', toc - tic)

    #tic = time.perf_counter()
    # eigenvector_centralities = [central.eigenvector_centrality(net)[i+n0] for i in range(num_nodes)]
    # max_c = max(eigenvector_centralities)
    # min_c = min(eigenvector_centralities)
    # eigenvector_centralities = [2*(c-min_c)/(max_c-min_c)-1 if (max_c-min_c) != 0 else 0 for c in eigenvector_centralities]
    # metrics.append(torch.tensor(eigenvector_centralities))
    #toc = time.perf_counter()
    #print('Eigen: ', toc - tic)

    #tic = time.perf_counter()
    # second_order_centralities = [central.second_order_centrality(net)[i+n0] for i in range(num_nodes)]
    # max_c = max(second_order_centralities)
    # min_c = min(second_order_centralities)
    # second_order_centralities = [2*(c-min_c)/(max_c-min_c)-1 if (max_c-min_c) != 0 else 0 for c in second_order_centralities]
    # metrics.append(torch.tensor(second_order_centralities))
    #toc = time.perf_counter()
    #print('Second Order: ', toc - tic)

    # closeness_vitalities = [vital.closeness_vitality(net)[i+n0] for i in range(num_nodes)]
    # max_c = max(closeness_vitalities)
    # min_c = min(closeness_vitalities)
    # closeness_vitalities = [2*(c-min_c)/(max_c-min_c)-1 if (max_c-min_c) != 0 else 0 for c in closeness_vitalities]
    # metrics.append(closeness_vitalities)

    #k = min(net_b.number_of_nodes(),10)
    # tic = time.perf_counter()
    # flow_betweenness_centralities = [central.current_flow_betweenness_centrality(net)[i+n0] for i in range(num_nodes)]
    # max_c = max(flow_betweenness_centralities)
    # min_c = min(flow_betweenness_centralities)
    # flow_betweenness_centralities = [2*(c-min_c)/(max_c-min_c)-1 if (max_c-min_c) != 0 else 0 for c in flow_betweenness_centralities]
    # metrics.append(flow_betweenness_centralities)
    # toc = time.perf_counter()
    # print('Flow Betweeness: ', toc - tic)
    return torch.stack(tuple(metrics))

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

if __name__ == '__main__':
    from utils import create_networks
    from torch_geometric.utils import from_networkx
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.cuda.device(0)
    _,net = create_networks('SF',10)
    model = GCN_Encoder(16,16,net.number_of_nodes(),True)
    geo_data = from_networkx(net)
    out = model(geo_data.edge_index)