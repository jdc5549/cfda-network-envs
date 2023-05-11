import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import networkx as nx
import random

class NetCascDataset(Dataset):
    def __init__(self,data_dir,casc_type,gnn=False):
        self.gnn = gnn
        topo_dir = data_dir + 'topologies/'
        topologies = []
        thresholds = []
        for filename in os.listdir(topo_dir):
            if filename.endswith('.edgelist'):
                topo_path = os.path.join(topo_dir, filename)
                G = nx.read_edgelist(topo_path,nodetype=int)
                thresh_filename = filename[:-9] + f'_thresh.npy'
                thresh_path = os.path.join(topo_dir,thresh_filename)
                if os.path.isfile(thresh_path):
                    thresholds.append(np.load(thresh_path))
                topologies.append(G)

        casc_data = np.load(data_dir + f"{casc_type}casc_trialdata.npy")
        self.action_data = casc_data[:,:,:-1].astype(int)
        self.reward_data = casc_data[:,:,-1]
        if len(thresholds) > 0: 
            thresholds = np.array(thresholds)
            max_t = np.min(thresholds, axis=1, keepdims=True)
            min_t = np.max(thresholds, axis=1, keepdims=True)
            norm_thresh = [2*(t-min_t[i])/(max_t[i]-min_t[i])-1 if (max_t[i]-min_t[i]) != 0 else np.zeros(thresholds.shape[1]) for i,t in enumerate(thresholds)]
            norm_thresh = torch.tensor(np.stack(norm_thresh))
        if self.gnn:
            if len(thresholds) > 0:
                self.net_features = norm_thresh.reshape([-1,1])
            else:
                self.net_features = torch.ones([len(topologies),1])
            from torch_geometric.utils import from_networkx
            self.edges = [from_networkx(topo).edge_index for topo in topologies]
        else:
            from graph_embedding import heuristic_feature_embedding
            num_nodes = topologies[0].number_of_nodes()
            self.feat_topo_data = torch.zeros([len(topologies),7,num_nodes])
            for i,net in enumerate(topologies):
                n0 = sorted(net.nodes())[0] #recognize 0 vs 1 indexing of node names
                nodes = [i for i in range(num_nodes)]
                max_node = max(nodes)
                min_node = min(nodes)
                nodes = [2*(node-min_node)/(max_node-min_node)-1 if (max_node-min_node) != 0 else 0 for node in nodes]
                #toc = time.perf_counter()
                #print('Node Names: ', toc - tic)
                feat_t = torch.tensor(nodes)
                feat_t = torch.stack((feat_t,norm_thresh[i]))
                embed = heuristic_feature_embedding(net) #TODO will need to account for CNs here
                feat_t = torch.cat((feat_t,embed),dim=0)
                self.feat_topo_data[i] = feat_t
            self.feat_topo_data = torch.permute(self.feat_topo_data,(0,2,1))

    def __len__(self):
        # Return the total number of samples in your dataset
        return self.reward_data.size

    def __getitem__(self, index):
        # Retrieve and return a single sample from your dataset at the given index
        # Return a tuple (data, label) or a dictionary {'data': data, 'label': label}
        #separate index into topology index and trial index
        topo_idx = index // self.reward_data.shape[1]
        trial_idx = index % self.reward_data.shape[1]
        #if GNN return net_features and edges to pass through the GNN, else return the heuristic embedding
        if self.gnn: 
            return ((self.net_features[topo_idx],self.edges[topo_idx],self.action_data[topo_idx,trial_idx]),self.reward_data[topo_idx,trial_idx])
        else:
            return ((self.feat_topo_data[topo_idx],self.action_data[topo_idx,trial_idx]),self.reward_data[topo_idx,trial_idx])

if __name__ == '__main__':
    data_dir = './data/5C2/100topo_10trials_thresholdcasc/'
    dataset = NetCascDataset(data_dir)
    data_len = dataset.__len__()
    print(f'Size of Dataset: {data_len}')
    import random
    rand_idx = random.randint(0, data_len-1)
    (feat_topo, actions), reward = dataset.__getitem__(rand_idx)
    print(f"featurized observation shape: ", feat_topo.shape)
    print(f"attack action: ", actions[:2])
    print(f"defense action: ", actions[2:])
    print(f"attacker reward: ", reward)