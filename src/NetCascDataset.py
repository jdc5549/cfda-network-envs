import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import networkx as nx
import random
import math
import pickle
from utils import get_combinatorial_actions
import multiprocessing as mp
import ctypes as c
from tqdm import tqdm

class NetCascDataset_Subact(Dataset):
    def __init__(self,ego_data_dir,subact_data_dir,casc_type,gnn=False,topo_features=False,val=False,cfda=True):
        max_cfac_ratio = 10
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
        if val:
            casc_data = np.load(subact_data_dir + f"subact_{casc_type}casc_valdata.npy")
        else:
            casc_data = np.load(subact_data_dir + f"subact_{casc_type}casc_trialdata.npy")
        casc_data = np.reshape(casc_data,(-1,5))
        if cfda and not val:
            cfac_data = np.load(subact_data_dir + f"subact_{casc_type}casc_CFACtrialdata.npy")
            shuffled_indices = np.random.permutation(len(cfac_data))
            limited_data = cfac_data[shuffled_indices]
            limited_data = limited_data[:max_cfac_ratio*len(casc_data)]
            casc_data = np.concatenate((casc_data,limited_data))
            #casc_data = cfac_data
        all_actions = get_combinatorial_actions(self.thresholds.shape[0],2)
        # unique_list = []
        # kept_idxs = []
        # self.casc_keys = set()
        # for i,c in enumerate(casc_data):
        #     c_tup = tuple(c.astype(int))
        #     key = len(all_actions)*all_actions.index(c_tup[:2]) + all_actions.index(c_tup[2:4])
        #     if key not in self.casc_keys:
        #         self.casc_keys.add(key)
        #         kept_idxs.append(i)
        #         unique_list.append(c)
        # casc_data = np.stack(unique_list)
        # if val:
        #     print(f'Val Data: {casc_data}')
        # else:
        #     print(f'Train Data: {casc_data}')

        self.action_data = casc_data[:,:-1].astype(int)
        self.reward_data = casc_data[:,-1]

        # self.z_action_data = torch.zeros((self.action_data.shape[0],2,self.thresholds.shape[0]),dtype=torch.int32)
        # for i,a in enumerate(self.action_data):
        #     self.z_action_data[i][0][a[:2]] = 1
        #     self.z_action_data[i][1][a[2:]] = 1

        num_processes = min((mp.cpu_count()-2,4))
        chunk_size = len(self.action_data) // num_processes
        #rem = len(self.action_data) % chunk_size

        with mp.Pool(processes=num_processes) as pool:
            # mp_array = mp.Array('i', self.action_data.shape[0]*2)
            # result_comb_action_idxs = np.frombuffer(mp_array.get_obj(),c.c_int)
            # result_comb_action_idxs = np.reshape(result_comb_action_idxs,(self.action_data.shape[0],2))
            args = [(i*chunk_size, min((i+1)*chunk_size,len(self.action_data)), self.action_data, all_actions,chunk_size)
                    for i in range((len(self.action_data)+chunk_size -1)//chunk_size)]
            result_comb_action_idxs = np.vstack([result for result in pool.starmap(self.process_chunk, args)])

            # Convert the shared array to a torch tensor
            self.comb_action_idxs_mp = torch.tensor(result_comb_action_idxs, dtype=torch.long).view(-1, 2)
        #self.comb_action_idxs_sp = torch.zeros((self.action_data.shape[0],2),dtype=torch.long)
        # for i,a in enumerate(self.action_data):
        #     atk_act = tuple(a[:2])
        #     comb_idx = all_actions.index(atk_act)
        #     self.comb_action_idxs_sp[i,0] = comb_idx
        #     def_act = tuple(a[2:])
        #     comb_idx = all_actions.index(def_act)
        #     self.comb_action_idxs_sp[i,1] = comb_idx
        self.failset_onehot_data = torch.zeros((casc_data.shape[0],self.thresholds.shape[0]))
        self.comb_action_idxs = self.comb_action_idxs_mp
        if not val:
            info_fn = subact_data_dir + f"subact_{casc_type}casc_trialinfo.pkl"
            with open(info_fn,'rb') as f:
                casc_info = pickle.load(f)
            # merged_info = {}
            # for i,info in enumerate(info_data):
            #     offset = i*100
            #     for key,value in info.items():
            #         new_key = key + offset
            #         merged_info[new_key] = value
            if cfda:
                cfda_info_fn = subact_data_dir + f"subact_{casc_type}casc_CFACtrialinfo.pkl"
                with open(cfda_info_fn,'rb') as f:
                    cfac_info = pickle.load(f)
                limited_info = [cfac_info[i] for i in shuffled_indices]
                limited_info = limited_info[:max_cfac_ratio*len(casc_info)]
                casc_info = casc_info + limited_info
            with tqdm(total=len(casc_info),desc='Processing Info') as pbar:
                for i,info in enumerate(casc_info):
                    self.failset_onehot_data[i,info['fail_set']] = 1
                    pbar.update(1)

            for i,r in enumerate(self.reward_data):
                mh_r = torch.mean(self.failset_onehot_data[i]).cpu().numpy()
                r = np.float32(r)
                try:
                    assert math.isclose(mh_r,r)
                except:
                    print(casc_info[i])
                    print(mh_r)
                    print(r)
                    print(self.failset_onehot_data[i])
                    exit()

        #i = 0
        # for key,value in casc_info.items():
        #     if key in kept_idxs:
        #         self.failset_onehot_data[i,value['fail_set']] = 1
        #         i += 1

        self.subact_sets = np.load(subact_data_dir + 'subact_sets.npy')
        if len(self.thresholds) > 0:
            max_t = np.max(self.thresholds)
            min_t = np.min(self.thresholds)
            norm_thresh = [2*(t-min_t)/(max_t-min_t)-1 if (max_t-min_t) != 0 else 0 for i,t in enumerate(self.thresholds)]
            norm_thresh = torch.tensor(np.stack(norm_thresh)).unsqueeze(0)
        if len(self.thresholds) > 0:
            self.net_features = norm_thresh#norm_thresh.reshape([-1,1])
        else:
            self.net_features = None
        if self.gnn:
            from torch_geometric.utils import from_networkx
            self.edges = from_networkx(self.topology).edge_index
        if topo_features:
            from graph_embedding import heuristic_feature_embedding
            embed = heuristic_feature_embedding(self.topology) #TODO will need to account for CNs here
            if self.net_features is not None:
                self.net_features = torch.cat((self.net_features,embed),dim=0)#permute(self.feat_topo_data,(0,2,1))  
            else:
                self.net_features = embed
        self.net_features = self.net_features.flatten()
            #self.feat_topo = feat_t
            #self.feat_topo_data = torch.permute(self.feat_topo_data,(0,2,1))      

    def process_chunk(self,start, end, action_data, all_actions,chunk_size):
        # if end <= action_data.shape[0]:
        #     result_comb_action_idxs = np.zeros((chunk_size,2))
        # else:
        result_comb_action_idxs = np.zeros((end-start,2))

        if start == 0:
            for i in tqdm(range(len(result_comb_action_idxs)), desc="First Chunk Progress", leave=False):
                a = action_data[i+start]
                atk_act = tuple(sorted(a[:2]))
                comb_idx = all_actions.index(atk_act)
                result_comb_action_idxs[i, 0] = comb_idx
                def_act = tuple(sorted(a[2:]))
                comb_idx = all_actions.index(def_act)
                result_comb_action_idxs[i, 1] = comb_idx                
        else:
            for i in range(len(result_comb_action_idxs)):
                a = action_data[i+start]
                atk_act = tuple(sorted(a[:2]))
                comb_idx = all_actions.index(atk_act)
                result_comb_action_idxs[i, 0] = comb_idx
                def_act = tuple(sorted(a[2:]))
                comb_idx = all_actions.index(def_act)
                result_comb_action_idxs[i, 1] = comb_idx
        return result_comb_action_idxs
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
            return ((self.net_features,self.edges,self.comb_action_idxs[index]),(self.reward_data[index],self.failset_onehot_data[index]))
        else:
            return ((self.net_features,self.comb_action_idxs[index]),(self.reward_data[index],self.failset_onehot_data[index]))


class NetCascDataset(Dataset):
    def __init__(self,data_dir,casc_type,gnn=False,val=True):
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
        if val:
            casc_data = np.load(data_dir + f"{casc_type}casc_valdata.npy")
        else:
            casc_data = np.load(data_dir + f"{casc_type}casc_trialdata.npy")
        self.action_data = casc_data[:,:,:-1].astype(int)
        self.z_action_data = torch.zeros((self.action_data.shape[0],self.action_data.shape[1],self.thresholds.shape[0]),dtype=torch.int32)
        for i,topo in self.action_data:
            for j,a in enumerate(topo):
                self.z_action_data[i][j][a[:2]] += 1
                self.z_action_data[i][j][a[2:]] += 2

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
            return ((self.net_features[topo_idx],self.edges[topo_idx],self.z_action_data[topo_idx,trial_idx]),self.reward_data[topo_idx,trial_idx])
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
    ego_data_dir = 'data/Ego/25C2/ego_NashEQ/'
    util = np.load(ego_data_dir + 'thresholdcasc_NashEQs/util.npy')
    #print(np.around(util,decimals=3))
    subact_data_dir = ego_data_dir + '80sets_5targets_100trials_RandomCycleExpl/'
    dataset=NetCascDataset_Subact(ego_data_dir,subact_data_dir,'threshold',topo_features=True)
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