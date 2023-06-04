import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import networkx as nx
import random
import pickle

class NetCascDataset_Subact(Dataset):
    def __init__(self,ego_data_dir,subact_data_dir,casc_type,gnn=False):
        self.gnn = gnn
        filename = 'net_0.edgelist'
        topo_path = os.path.join(ego_data_dir, filename)
        self.topology = nx.read_edgelist(topo_path,nodetype=int)
        thresh_filename = filename[:-9] + f'_thresh.npy'
        thresh_path = os.path.join(ego_data_dir,thresh_filename)
        if os.path.isfile(thresh_path):
            self.thresholds = np.load(thresh_path)
        else:
            self.thresholds = []
        casc_data = np.load(subact_data_dir + f"subact_{casc_type}casc_trialdata.npy")
        casc_data = np.reshape(casc_data,(-1,5))

        unique_list = []
        kept_idxs = []
        for i,c1 in enumerate(casc_data):
            unique = True
            for c2 in unique_list:
                if np.array_equal(c1,c2):
                    unique = False
                    break
            if unique:
                unique_list.append(c1)
                kept_idxs.append(i)
        casc_data = np.asarray(unique_list)

        self.action_data = casc_data[:,:-1].astype(int)
        self.reward_data = casc_data[:,-1]

        info_fn = subact_data_dir + f"subact_{casc_type}casc_trialinfo.pkl"
        with open(info_fn,'rb') as f:
            info_data = pickle.load(f)

        merged_info = {}
        for i,info in enumerate(info_data):
            offset = i*100
            for key,value in info.items():
                new_key = key + offset
                merged_info[new_key] = value


        self.failset_onehot_data = torch.zeros((casc_data.shape[0],self.thresholds.shape[0]))
        i = 0
        for key,value in merged_info.items():
            if key in kept_idxs:
                self.failset_onehot_data[i,value['fail_set']] = 1
                i += 1
        self.subact_sets = np.load(subact_data_dir + 'subact_sets.npy')
        if len(self.thresholds) > 0:
            max_t = np.max(self.thresholds)
            min_t = np.min(self.thresholds)
            norm_thresh = [2*(t-min_t)/(max_t-min_t)-1 if (max_t-min_t) != 0 else 0 for i,t in enumerate(self.thresholds)]
            norm_thresh = torch.tensor(np.stack(norm_thresh)).unsqueeze(0)
        if self.gnn:
            if len(self.thresholds) > 0:
                self.net_features = norm_thresh#norm_thresh.reshape([-1,1])
            else:
                self.net_features = torch.ones([len(self.topology),1])
            from torch_geometric.utils import from_networkx
            self.edges = [from_networkx(topo).edge_index for topo in self.topology]
        else:
            from graph_embedding import heuristic_feature_embedding
            num_nodes = self.topology.number_of_nodes()
            #n0 = sorted(net.nodes())[0] #recognize 0 vs 1 indexing of node names
            # nodes = [i for i in range(num_nodes)]
            # max_node = max(nodes)
            # min_node = min(nodes)
            #nodes = [2*(node-min_node)/(max_node-min_node)-1 if (max_node-min_node) != 0 else 0 for node in nodes]
            #toc = time.perf_counter()
            #print('Node Names: ', toc - tic)
            #feat_t = torch.tensor(nodes)
            embed = heuristic_feature_embedding(self.topology) #TODO will need to account for CNs here
            self.feat_topo = torch.cat((norm_thresh,embed),dim=0).flatten()#permute(self.feat_topo_data,(0,2,1))    
            #self.feat_topo = feat_t
            #self.feat_topo_data = torch.permute(self.feat_topo_data,(0,2,1))          
    def __len__(self):
        # Return the total number of samples in your dataset
        return self.reward_data.size

    def show_topology(self):
        import matplotlib.pyplot as plt
        nx.draw(self.topology,with_labels=True)
        plt.draw()
        plt.show()

    def __getitem__(self, index):
        # Retrieve and return a single sample from your dataset at the given index
        # Return a tuple (data, label) or a dictionary {'data': data, 'label': label}
        #separate index into topology index and trial index
        # topo_idx = index // self.reward_data.shape[1]
        # trial_idx = index % self.reward_data.shape[1]
        #if GNN return net_features and edges to pass through the GNN, else return the heuristic embedding
        if self.gnn: 
            return ((self.net_features[topo_idx],self.edges[topo_idx],self.action_data[topo_idx,trial_idx]),self.reward_data[topo_idx,trial_idx])
        else:
            return ((self.feat_topo,self.action_data[index]),(self.reward_data[index],self.failset_onehot_data[index]))


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
            max_t = np.max(thresholds, axis=1, keepdims=True)
            min_t = np.min(thresholds, axis=1, keepdims=True)
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
            num_features = 8
            self.feat_topo_data = torch.zeros([len(topologies),num_features,num_nodes])
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
    # from utils import create_random_nets, ncr, get_combinatorial_actions
    # data_dir = './data/5C2/training_data/8000topo_100trials_RandomExpl/' 
    # dataset = NetCascDataset(data_dir,'threshold')
    # data_len = dataset.__len__()
    # print(f'Size of Dataset: {data_len}')
    # all_topos = dataset.feat_topo_data
    # trial_data = np.load(data_dir + 'thresholdcasc_trialdata.npy') 
    # all_actions = get_combinatorial_actions(5,2)
    # print(all_actions)
    # topo_list = []
    # for i in range(all_topos.size(0)):
    #     topo_list.append(all_topos[i].tolist())
    # # import random
    # # rand_idx = random.randint(0, data_len-1)
    # from torch.utils.data import DataLoader
    # data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    # for i, data in enumerate(data_loader):
    #     (feat_topo,actions), reward = data
    #     B = feat_topo.shape[0]
    #     topo_index = [topo_list.index(feat_topo[j].tolist()) for j in range(B)][0]
    #     actions = actions.tolist()[0]
    #     matching_data = None
    #     for element in trial_data[topo_index]:
    #         if np.array_equal(element[:4],actions):
    #             matching_data = element
    #             break
    #     try:
    #         data_rew = matching_data[-1][0]
    #         if data_rew != reward:
    #             print(f'Loader Reward {reward}, Data Reward {data_rew}')
    #     except:
    #         pass
    ego_data_dir = 'data/Ego/7C2/ego_NashEQ/'
    util = np.load(ego_data_dir + 'thresholdcasc_NashEQs/util.npy')
    #print(np.around(util,decimals=3))
    subact_data_dir = ego_data_dir + '5sets_5targets_100trials_RandomCycleExpl/'
    dataset=NetCascDataset_Subact(ego_data_dir,subact_data_dir,'threshold')
    rand_idx = random.randint(0,dataset.__len__())
    (_,action),(reward,multi_hot) = dataset.__getitem__(rand_idx)
    print(action)
    print(reward)
    print(multi_hot)
    #print(dataset.subact_sets)
    #print(dataset.thresholds)
    #dataset.show_topology()
    # from utils import get_combinatorial_actions
    # all_actions = get_combinatorial_actions(7,2)
    # for i,act in enumerate(dataset.action_data):
    #     for j,a in enumerate(act):
    #         atk_idx = all_actions.index(a[:2].tolist())
    #         def_idx = all_actions.index(a[2:].tolist())
    #         if util[atk_idx,def_idx] != dataset.reward_data[i,j]:
    #             print([all_actions[atk_idx],all_actions[def_idx]])