import numpy as np
import os
import time
import random
import multiprocessing as mp
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

import marl
from .policy import Policy, ModelBasedPolicy
from marl.tools import gymSpace2dim,get_combinatorial_actions

from scipy.optimize import linprog
import networkx as nx


class RandomPolicy(Policy):
    """
    The class of random policies
    
    :param model: (Model or torch.nn.Module) The q-value model 
    :param action_space: (gym.Spaces) The action space
    """
    def __init__(self, action_space,num_actions=1,all_actions=[]):
        self.action_space = action_space
        self.num_actions= num_actions
        self.all_actions = all_actions
        
    def __call__(self, state,num_actions=1):
        """
        Return a random action given the state
        
        :param state: (Tensor) The current state
        """
        actions_list = []
        for j in range(len(state)):
            actions = []
            for i in range(self.num_actions):    
                a = self.action_space.sample()
                while a in actions:
                    a = self.action_space.sample()
                actions.append(self.all_actions[a])
            if len(actions) > 1:
                actions_list.append(actions)
            else:
                actions_list.append(actions[0])
        return actions_list

class TargetedPolicy(Policy):
    """
    The class of Node Degree Heuristic policies
    
    :param model: (Model or torch.nn.Module) The q-value model 
    :param action_space: (gym.Spaces) The action space
    """
    def __init__(self, action_space,node_ranking='degree_centrality',num_actions=1,all_actions=[]):
        self.action_space = action_space
        self.num_actions=num_actions
        self.all_actions = all_actions
        self.node_rank_alg = node_ranking


    def get_TOPSIS_vals(self,G):
        import networkx.algorithms.centrality as central
        num_nodes = G.number_of_nodes()
        degree_centrality = central.degree_centrality(G)
        closeness_centrality = central.closeness_centrality(G)
        betweenness_centrality = central.betweenness_centrality(G)
        eigenvector_centrality = central.eigenvector_centrality(G)
        
        # Combine centrality measures into a matrix
        centrality_matrix = np.array([
            list(degree_centrality.values()),
            list(closeness_centrality.values()),
            list(betweenness_centrality.values()),
            list(eigenvector_centrality.values())
        ]).T
        
        # Normalize the centrality measures
        norm_centrality_matrix = centrality_matrix / np.linalg.norm(centrality_matrix)
        
        # Calculate the weighted normalized decision matrix
        P = norm_centrality_matrix / norm_centrality_matrix.sum(axis=0)
        entropy = -np.sum(P * np.log(P + 1e-9), axis=0) / np.log(num_nodes)
        weights = (1 - entropy) / (4 - np.sum(entropy))        
        #weighted_norm_matrix = norm_centrality_matrix * weights
        
        # Determine positive and negative ideal solutions
        positive_ideal_solution = norm_centrality_matrix.max(axis=0)
        negative_ideal_solution = norm_centrality_matrix.min(axis=0)
        
        # Calculate separation measures
        separation_positive = np.sqrt((weights*(norm_centrality_matrix - positive_ideal_solution) ** 2).sum(axis=1))
        separation_negative = np.sqrt((weights*(norm_centrality_matrix - negative_ideal_solution) ** 2).sum(axis=1))
        
        # Calculate relative closeness to the ideal solution
        relative_closeness = separation_negative / (separation_positive + separation_negative)
        
        # Create a dictionary to map nodes to their relative closeness values
        node_importance = {node: relative_closeness[i] for i, node in enumerate(G.nodes())}
        
        return node_importance

    def __call__(self, state,num_actions=1):
        """
        Return the highest degree nodes
        
        :param state: (Tensor) The current state
        """
        comb_size = len(self.all_actions[0])
        actions_list = []
        for j in range(len(state)):
            G = nx.from_numpy_matrix(state[j][:-1])
            if self.node_rank_alg == 'degree_centrality':
                node_vals = [G.degree(n) for n in G.nodes]
            elif self.node_rank_alg == 'TOPSIS':
                node_vals = self.get_TOPSIS_vals(G)
            else:
                raise ValueError(f'Node Ranking algorithm {self.node_rank_alg} not recognized')

            act_vals = [np.sum([node_vals[act[i]] for i in range(comb_size)]) for act in self.all_actions]
            self.sorted_idx = np.flip(np.argsort(act_vals))

            sorted_acts = [self.all_actions[idx] for idx in self.sorted_idx]
            actions = sorted_acts[:num_actions]
            if len(actions) > 1:
                actions_list.append(actions)
            else:
                actions_list.append(actions[0])
        return actions_list

class RTMixedPolicy(Policy):
    """
    The class of Node Degree Heuristic policies
    
    :param model: (Model or torch.nn.Module) The q-value model 
    :param action_space: (gym.Spaces) The action space
    """
    def __init__(self, action_space,pt,num_actions=1,all_actions=[]):
        self.action_space = action_space
        self.num_actions= num_actions
        self.all_actions = all_actions
        self.pt = pt

    def __call__(self, state):
        """
        """
        rn = random.uniform(0,1)
        actions_list = []
        if rn < self.pt:
            act_degrees = [state[act[0]] + state[act[1]] for act in self.all_actions]
            sorted_idx = np.flip(np.argsort(act_degrees))
            #sorted_acts = [self.all_actions[idx] for idx in sorted_idx]
            a = sorted_idx[0]
            return self.all_actions[a]
        else:
            a = self.action_space.sample()
            return self.all_actions[a]        

class QPolicy(ModelBasedPolicy):
    """
    The class of policies based on a Q function
    
    :param model: (Model or torch.nn.Module) The q-value model 
    :param observation_space: (gym.Spaces) The observation space
    :param action_space: (gym.Spaces) The action space
    """
    def __init__(self, model, observation_space=None, action_space=None):
        self.observation_space = observation_space
        self.action_space = action_space
        self.model = marl.model.make(model, obs_sp=gymSpace2dim(self.observation_space), act_sp=gymSpace2dim(self.action_space))
        
        
    def __call__(self, state,num_actions=1):
        """
        Return an action given the state
        
        :param state: (Tensor) The current state
        """  
        if isinstance(self.Q, nn.Module):
            state = torch.tensor(state).float().unsqueeze(0)
            with torch.no_grad():
                sorted_acts = torch.sort(self.Q(state),descending=True)
                return sorted_acts[1][:num_actions].tolist()
        else:
            sorted_acts = torch.sort(self.Q(state),descending=True)
            return sorted_acts[:num_actions]

    @property
    def Q(self):
        return self.model

class MinimaxQTablePolicy(ModelBasedPolicy):
    """
    The class of policies based on a Q function
    
    :param model: (Model or torch.nn.Module) The q-value model 
    :param observation_space: (gym.Spaces) The observation space
    :param action_space: (gym.Spaces) The action space
    """
    def __init__(self, model, observation_space=None, player=0,action_space=None,all_actions=[],act_degree=1):
        self.observation_space = observation_space
        self.action_space = action_space
        self.model = marl.model.make(model, obs_sp=gymSpace2dim(self.observation_space), act_sp=gymSpace2dim(self.action_space))
        self.player = player
        self.policy = None
        self.degree = act_degree
        self.all_actions = all_actions

    def get_policy(self,observation,dummy=None): #dummy arg where feat_actions goes so it's compatible with generic get_policy method
        num_player_actions = gymSpace2dim(self.action_space) #TODO: figure out how to pass both players action spaces so we can do assymetric games
        num_p1_acts = gymSpace2dim(self.action_space)
        num_p2_acts = gymSpace2dim(self.action_space)
        c = np.zeros(num_player_actions+1)
        c[num_player_actions] = -1
        U = self.Q(state=observation).squeeze().cpu().detach().numpy()
        # print(self.player)
        # if self.player == 0:
        U = U.T #LP package requires the player to be the column player
        A_ub = np.hstack((-U,np.ones((num_player_actions,1))))
        b_ub = np.zeros(num_player_actions)
        A_eq = np.ones((1,num_player_actions+1))
        A_eq[:,num_player_actions] = 0
        b_eq = 1
        bounds = [(0,None) for i in range(num_player_actions+1)]
        bounds[num_player_actions] = (None,None)
        res = linprog(c,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq,bounds=bounds)
        if not res["success"]:
            print("Failed to Optimize LP with exit status ",res["status"])
        value = res["fun"]
        policy = res["x"][0:num_player_actions]
        return policy, U
        
    def __call__(self,observation,num_actions=1,policy=None):
        """
        Return an action given the state
        
        :param state: (Tensor) The current state
        """  
        retries = 0
        retry_lim = 100
        if policy is None:
            policy = self.get_policy(observation)
        try:
            pd = Categorical(torch.tensor(policy))
        except:
            print("Failed with policy:", policy)
            pd = Categorical(1/len(policy)*torch.ones_like(torch.tensor(policy)))
        actions = []
        num_samples = num_actions / self.degree
        for i in range(num_actions):
            retries=0
            a = pd.sample().item()
            while a in actions:
                if retries < retry_lim:
                    a = pd.sample().item()
                    retries += 1
                else:
                    a = self.action_space.sample()
            if num_samples == 1:
                return self.all_actions[a]
            else:  
                if type(self.all_actions[a]) == list:
                    for act in self.all_actions[a]: actions.append(act)          
                else:
                    actions.append(self.all_actions[a])  
        return actions

    @property
    def Q(self):
        return self.model

class QCriticPolicy(ModelBasedPolicy):
    """
    The class of policies based on a Q critic-style function
    
    :param model: (Model or torch.nn.Module) The q-value model 
    :param observation_space: (gym.Spaces) The observation space
    :param action_space: (gym.Spaces) The action space
    """
    def __init__(self, model, observation_space=None, action_space=None,atk_degree=1,all_actions=[]):
        self.observation_space = observation_space
        self.action_space = action_space
        self.model = marl.model.make(model, obs_sp=gymSpace2dim(self.observation_space), act_sp=gymSpace2dim(self.action_space))
        self.all_actions = all_actions
        self.degree = atk_degree

    def get_policy(self,state,featurized_actions):
        num_player_actions = gymSpace2dim(self.action_space)
        value = np.zeros(num_player_actions)
        for i in range(num_player_actions):
            value[i] = self.Q(state,featurized_actions[i])
        policy = F.gumbel_softmax(torch.tensor(value)).numpy()
        return policy, value
        
    def __call__(self, state,num_actions=1,policy=None):
        if self.degree > 1:
            featurized_actions = torch.stack([state[action].flatten() for action in self.all_actions]).float()
        else:
            featurized_actions = torch.tensor(state).float()
        state = torch.tensor(np.mean(state,axis=0)).float()
        retries = 0
        retry_lim = 100
        with torch.no_grad():
            policy = self.get_policy(state,featurized_actions)
            pd = Categorical(torch.tensor(policy))
            actions = []
            num_samples = num_actions / self.degree
            for i in range(num_actions):
                retries=0
                a = pd.sample().item()
                while a in actions:
                    if retries < retry_lim:
                        a = pd.sample().item()
                        retries += 1
                    else:
                        a = self.action_space.sample()
                if num_samples == 1:
                    return self.all_actions[a]
                else:  
                    if type(self.all_actions[a]) == list:
                        for act in self.all_actions[a]: actions.append(act)          
                    else:
                        actions.append(self.all_actions[a])  
            return actions

    @property
    def Q(self):
        return self.model

class MABCriticPolicy(ModelBasedPolicy):
    """
    The class of policies based on a Q critic-style function
    
    :param model: (Model or torch.nn.Module) The q-value model 
    :param action_space: (gym.Spaces) The action space
    """
    def __init__(self, model, action_space,network_size):
        self.action_space = action_space
        self.all_actions = get_combinatorial_actions(network_size,2)
        self.model = marl.model.make(model, act_sp=gymSpace2dim(self.action_space))

    def get_policy(self):
        policy = F.gumbel_softmax(torch.tensor(value)).numpy()
        return policy, self.model.q_table()
        
    def __call__(self):
        policy,_ = self.get_policy(state,featurized_actions)
        pd = Categorical(torch.tensor(policy))
        a = pd.sample().item()
        return self.all_actions[a]

    @property
    def Q(self):
        return self.model

class MinimaxQCriticPolicy(ModelBasedPolicy):
    """
    The class of policies based on a Q critic-style function with two player zero sum reward
    
    :param model: (Model or torch.nn.Module) The q-value model 
    :param observation_space: (gym.Spaces) The observation space
    :param action_space: (gym.Spaces) The action space
    """
    def __init__(self, model, observation_space=None, action_space = None,player=0,all_actions=[],act_degree=1):
        self.observation_space = observation_space
        self.action_space = action_space
        self.model = model
        self.player = player
        self.policy = None
        self.degree = act_degree
        self.all_actions = all_actions

    def get_policy(self,state,featurized_actions):
        #print('t_obs in get_policy:', state)
        num_player_actions = gymSpace2dim(self.action_space)
        num_p1_acts = gymSpace2dim(self.action_space)
        num_p2_acts = gymSpace2dim(self.action_space)
        c = np.zeros(num_player_actions+1)
        c[num_player_actions] = -1
        U = np.zeros([num_p1_acts,num_p2_acts])
        # U = (-2*self.player +1) * np.array([[0.1 for i in range(10)],[0.1 for i in range(10)],
        #     [0.2 for i in range(10)],[0.6 for i in range(10)],[0.1 for i in range(10)],
        #     [0.2 for i in range(10)],[0.2 for i in range(10)],[0.2 for i in range(10)],
        #     [0.1 for i in range(10)],[0.3 for i in range(10)]])
        #np.fill_diagonal(U,0)
        #U_noisy = U + np.random.randn(num_p1_acts,num_p2_acts)*1e-4
        state_t = state.repeat(num_p1_acts,num_p2_acts,1)
        p1_act_t = torch.unsqueeze(featurized_actions,dim=1)#torch.zeros(num_p1_acts,self.num_p2_acts,featurized_actions[0].numel())
        p1_act_t = p1_act_t.expand(num_p1_acts,num_p2_acts,featurized_actions[0].numel())
        p2_act_t = torch.unsqueeze(featurized_actions,dim=0)
        p2_act_t = p2_act_t.expand(num_p1_acts,num_p2_acts,featurized_actions[0].numel())
        U = torch.squeeze(self.Q(state_t,p1_act_t,p2_act_t)).detach().cpu().numpy()*(-2*self.player +1)
        if self.player == 0:
            U_calc = -U.T #LP package requires the player to be the column player
        else:
            U_calc = -U
        A_ub = np.hstack((U_calc,np.ones((num_player_actions,1))))
        b_ub = np.zeros(num_player_actions)
        A_eq = np.ones((1,num_player_actions+1))
        A_eq[:,num_player_actions] = 0
        b_eq = 1
        bounds = [(0,None) for i in range(num_player_actions+1)]
        bounds[num_player_actions] = (None,None)
        res = linprog(c,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq,bounds=bounds,options={'maxiter': 1000,'tol': 1e-6})

        if not res["success"]:
            print("Failed to Optimize LP with exit status ",res["status"])
        value = res["fun"]
        policy = res["x"][0:num_player_actions]
        # print('policy: ', policy)
        # if self.player == 1:
        #     exit()
        return policy,U

    def __call__(self, state,num_actions=1,policy=None):
        if self.degree > 1:
            featurized_actions = torch.stack([state[action].flatten() for action in self.all_actions])
        else:
            featurized_actions = torch.tensor(state)            
        t_state = torch.mean(state,dim=0)
        with torch.no_grad():
            retries = 0
            retry_lim = 100
            if policy is None:
                policy, value = self.get_policy(t_state,featurized_actions)
            try:
                pd = Categorical(torch.tensor(policy))
            except:
                print("Failed util: ",value)
                print("Failed with policy:", policy)
                pd = Categorical(1/len(policy)*torch.ones_like(torch.tensor(policy)))
            actions = []
            num_samples = num_actions / self.degree
            for i in range(num_actions):
                retries=0
                a = pd.sample().item()
                while a in actions:
                    if retries < retry_lim:
                        a = pd.sample().item()
                        retries += 1
                    else:
                        a = self.action_space.sample()
                if num_samples == 1:
                    return self.all_actions[a]
                else:  
                    if type(self.all_actions[a]) == list:
                        for act in self.all_actions[a]: actions.append(act)          
                    else:
                        actions.append(self.all_actions[a])  
            actions.sort()
            return actions

    @property
    def Q(self):
        return self.model

class SubactMinimaxQCriticPolicy(ModelBasedPolicy):
    """
    The class of policies based on a Q critic-style function with two player zero sum reward
    
    :param model: (Model or torch.nn.Module) The q-value model 
    :param observation_space: (gym.Spaces) The observation space
    :param action_space: (gym.Spaces) The action space
    """
    def __init__(self, model, action_space = None,player=0,all_actions=[],eval_mode=False,device='cpu',model_path=None,cluster_map=None):
        self.action_space = action_space
        self.model = model
        self.player = player
        self.all_actions = all_actions
        self.cluster_map = cluster_map
        if self.cluster_map is not None:
            p = len(self.all_actions[0])
            num_clusters = int(max(cluster_map)+1)
            pairs_zero = [[-1,n] for n in range(num_clusters)]
            same_pairs = [[n,n] for n in range(num_clusters)]
            wrong_cluster_combs = pairs_zero + same_pairs
            cluster_combs = get_combinatorial_actions(num_clusters,p)
            self.cluster_actions = cluster_combs + wrong_cluster_combs

        self.max_size_in_mem = 250000
        self.device = device
        #if len(all_actions)*len(all_actions) > self.max_size_in_mem:
        self.action_indices = torch.tensor([i for i in range(len(self.all_actions))]).to(torch.int).to(self.device)
        #else:
        #    self.action_indices = torch.tensor([i for i in range(len(self.all_actions))]).to(torch.long).to(device)
        self.eval_mode = eval_mode
        self.model_path = model_path
        policy_file = f'{self.model_path}/player{self.player}_policy.npy'
        if os.path.exists(policy_file):
            self.policy = np.load(policy_file)
            print(f'Loaded player {self.player} policy from {policy_file}')
        else:
            self.policy = None

    def get_toy_policy(self,node_features=None,edge_index=None): 
        num_player_actions = gymSpace2dim(self.action_space)
        num_p1_acts = gymSpace2dim(self.action_space)
        num_p2_acts = gymSpace2dim(self.action_space)
        if edge_index is not None:
            expanded_features = node_features.repeat(self.action_indices.shape[0]*self.action_indices.shape[0],1).unsqueeze(1)
            edge_index = edge_index.unsqueeze(0)
            edge_index = edge_index.repeat(self.action_indices.shape[0]*self.action_indices.shape[0],1,1)
            #expand actions to multi-hot for GNN
            multi_hot_actions = torch.zeros(self.action_indices.shape[0]*self.action_indices.shape[0],node_features.shape[-1],2)
            for i,aa in enumerate(self.all_actions):
                for j,dd in enumerate(self.all_actions):
                    multi_hot_actions[i*len(self.all_actions)+j,aa,0] = 1
                    multi_hot_actions[i*len(self.all_actions)+j,dd,1] = 1
            multi_hot_actions = multi_hot_actions.to(self.device)
            out = torch.squeeze(self.Q(multi_hot_actions,expanded_features,edge_index)).detach()*(-2*self.player +1)
            out = out.reshape(self.action_indices.shape[0],self.action_indices.shape[0],-1)
        else:
            if node_features is None:
                expanded_features = None
            else:
                expanded_features = node_features.unsqueeze(0)
                expanded_features = expanded_features.half().repeat(self.action_indices.shape[0],self.action_indices.shape[0],1)
                #expanded_features = expanded_features.unsqueeze(2)
            grid_p1,grid_p2 = torch.meshgrid(self.action_indices,self.action_indices)
            actions = torch.stack((grid_p1,grid_p2),dim=-1)
            out = torch.squeeze(self.Q(actions,expanded_features)).detach()*(-2*self.player +1)           
        U_full = torch.mean(out,dim=-1).cpu().numpy()
        num_cluster_acts = len(self.cluster_actions)
        #convert full action space Utility to cluster space Utility
        U_cluster = np.zeros((num_cluster_acts,num_cluster_acts))
        U_counts = np.zeros((num_cluster_acts,num_cluster_acts))

        for i in range(num_player_actions):
            for j in range(num_player_actions):
                aa = self.all_actions[i]
                da = self.all_actions[j]
                aa_cluster = sorted([self.cluster_map[a] for a in aa])
                da_cluster = sorted([self.cluster_map[d] for d in da])
                aci = self.cluster_actions.index(aa_cluster)
                dci = self.cluster_actions.index(da_cluster)
                U_counts[aci,dci] += 1
                U_cluster[aci,dci] = U_cluster[aci,dci] * (U_counts[aci,dci]-1)/U_counts[aci,dci] + U_full[i,j]/U_counts[aci,dci]
        c = np.zeros(num_cluster_acts+1)
        c[num_cluster_acts] = -1
        if self.player == 0:
            U_calc = -U_cluster.T #LP package requires the player to be the column player
        else:
            U_calc = -U_cluster
        A_ub = np.hstack((U_calc,np.ones((num_cluster_acts,1))))
        b_ub = np.zeros(num_cluster_acts)
        A_eq = np.ones((1,num_cluster_acts+1))
        A_eq[:,num_cluster_acts] = 0
        b_eq = 1
        bounds = [(0,None) for i in range(num_cluster_acts+1)]
        bounds[num_cluster_acts] = (None,None)
        options = {'maxiter': 1000,'tol': 1e-6,'presolve': True,'autoscale':True}
        res = linprog(c,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq,bounds=bounds,options=options)

        if not res["success"]:
            print("Failed to Optimize LP with exit status ",res["status"])
        value = res["fun"]
        policy = res["x"][0:num_cluster_acts]
        # print('policy: ', policy)
        # if self.player == 1:
        #     exit()
        return policy,U_cluster


    def get_large_policy(self,node_features=None,edge_index=None):
        import os
        num_player_actions = gymSpace2dim(self.action_space)
        num_p1_acts = gymSpace2dim(self.action_space)
        num_p2_acts = gymSpace2dim(self.action_space)
        expanded_features = node_features.repeat(num_p1_acts,1)
        U_player = np.zeros(len(self.all_actions))
        if edge_index is not None:
            edge_index = edge_index.unsqueeze(0)
            edge_index = edge_index.repeat(num_p1_acts,1,1)
        with tqdm(total=len(self.action_indices),desc=f'Player {self.player} Ego Policy Progress') as pbar:
            for a in range(len(self.all_actions)):
                if self.player == 0:
                    actions = torch.stack((a*torch.ones((len(self.all_actions),),device=self.device),self.action_indices)).T.to(torch.int)
                else:
                    actions = torch.stack((self.action_indices,a*torch.ones((len(self.all_actions),),device=self.device))).T.to(torch.int)
                out = torch.squeeze(self.Q(actions,expanded_features)).detach()*(-2*self.player +1)
                val = torch.mean(torch.mean(out,dim=-1),dim=-1).cpu().numpy()
                U_player[a] = val
                pbar.update(1)
        policy = F.gumbel_softmax(torch.tensor(U_player)).numpy()
        if self.model_path is not None:
            np.save(f'{self.model_path}/player{self.player}_policy.npy',policy)
            np.save(f'{self.model_path}/player{self.player}_util.npy',U_player)
        return policy, U_player
 
    def get_policy(self,node_features=None,edge_index=None):
        if self.cluster_map is not None:
            return self.get_toy_policy(node_features,edge_index)
        #print('t_obs in get_policy:', state)
        num_player_actions = gymSpace2dim(self.action_space)
        num_p1_acts = gymSpace2dim(self.action_space)
        num_p2_acts = gymSpace2dim(self.action_space)
        c = np.zeros(num_player_actions+1)
        c[num_player_actions] = -1
        # if num_player_actions > self.max_size_in_mem/4:
        #     U = np.zeros([num_p1_acts,num_p2_acts],dtype=np.half)
        # else:
        #     U = np.zeros([num_p1_acts,num_p2_acts])
        # U = (-2*self.player +1) * np.array([[0.1 for i in range(10)],[0.1 for i in range(10)],
        #     [0.2 for i in range(10)],[0.6 for i in range(10)],[0.1 for i in range(10)],
        #     [0.2 for i in range(10)],[0.2 for i in range(10)],[0.2 for i in range(10)],
        #     [0.1 for i in range(10)],[0.3 for i in range(10)]])
        #np.fill_diagonal(U,0)
        #U_noisy = U + np.random.randn(num_p1_acts,num_p2_acts)*1e-4
        if edge_index is not None:
            expanded_features = node_features.repeat(self.action_indices.shape[0]*self.action_indices.shape[0],1).unsqueeze(1)
            edge_index = edge_index.unsqueeze(0)
            edge_index = edge_index.repeat(self.action_indices.shape[0]*self.action_indices.shape[0],1,1)
            #expand actions to multi-hot for GNN
            multi_hot_actions = torch.zeros(self.action_indices.shape[0]*self.action_indices.shape[0],node_features.shape[-1],2)
            for i,aa in enumerate(self.all_actions):
                for j,dd in enumerate(self.all_actions):
                    multi_hot_actions[i*len(self.all_actions)+j,aa,0] = 1
                    multi_hot_actions[i*len(self.all_actions)+j,dd,1] = 1
            multi_hot_actions = multi_hot_actions.to(self.device)
            out = torch.squeeze(self.Q(multi_hot_actions,expanded_features,edge_index)).detach()*(-2*self.player +1)
            out = out.reshape(self.action_indices.shape[0],self.action_indices.shape[0],-1)
        else:
            if node_features is None:
                expanded_features = None
            else:
                expanded_features = node_features.unsqueeze(0)
                expanded_features = expanded_features.repeat(self.action_indices.shape[0],self.action_indices.shape[0],1)
                #expanded_features = expanded_features.unsqueeze(2)
            grid_p1,grid_p2 = torch.meshgrid(self.action_indices,self.action_indices)
            actions = torch.stack((grid_p1,grid_p2),dim=-1)
            out = torch.squeeze(self.Q(actions,expanded_features)).detach()*(-2*self.player +1)           
        U = torch.mean(out,dim=-1).cpu().numpy()
        if self.player == 0:
            U_calc = -U.T #LP package requires the player to be the column player
        else:
            U_calc = -U
        A_ub = np.hstack((U_calc,np.ones((num_player_actions,1))))
        b_ub = np.zeros(num_player_actions)
        A_eq = np.ones((1,num_player_actions+1))
        A_eq[:,num_player_actions] = 0
        b_eq = 1
        bounds = [(0,None) for i in range(num_player_actions+1)]
        bounds[num_player_actions] = (None,None)
        options = {'maxiter': 1000,'tol': 1e-6,'presolve': True,'autoscale':True}
        res = linprog(c,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq,bounds=bounds,options=options)

        if not res["success"]:
            print("Failed to Optimize LP with exit status ",res["status"])
        value = res["fun"]
        policy = res["x"][0:num_player_actions]
        # print('policy: ', policy)
        # if self.player == 1:
        #     exit()
        return policy,U

    def __call__(self,state=None):          
        with torch.no_grad():
            if self.policy is None or not self.eval_mode:
                if len(self.action_indices) > self.max_size_in_mem:
                    self.policy, self.value = self.get_large_policy(self.action_indices)
                else:
                    self.policy, self.value = self.get_policy(self.action_indices)
            try:
                pd = Categorical(torch.tensor(self.policy))
            except:
                print("Failed util: ",self.value)
                print("Failed with policy:", self.policy)
                pd = Categorical(1/len(policy)*torch.ones_like(torch.tensor(policy)))

            a = pd.sample().item()
            return self.all_actions[a]

    @property
    def Q(self):
        return self.model

class GNNMinimaxQCriticPolicy(ModelBasedPolicy):
    """
    The class of policies based on a Q critic-style function with two player zero sum reward
    
    :param model: (Model or torch.nn.Module) The q-value model 
    :param observation_space: (gym.Spaces) The observation space
    :param action_space: (gym.Spaces) The action space
    """
    def __init__(self, model, action_space = None,player=0,all_actions=[],act_degree=1):
        self.action_space = action_space
        self.model = model
        self.player = player
        self.policy = None
        self.degree = act_degree
        self.all_actions = all_actions

    def get_policy(self,state,featurized_actions):
        #print('t_obs in get_policy:', state)
        num_player_actions = gymSpace2dim(self.action_space)
        num_p1_acts = gymSpace2dim(self.action_space)
        num_p2_acts = gymSpace2dim(self.action_space)
        c = np.zeros(num_player_actions+1)
        c[num_player_actions] = -1
        U = np.zeros([num_p1_acts,num_p2_acts])
        # U = (-2*self.player +1) * np.array([[0.1 for i in range(10)],[0.1 for i in range(10)],
        #     [0.2 for i in range(10)],[0.6 for i in range(10)],[0.1 for i in range(10)],
        #     [0.2 for i in range(10)],[0.2 for i in range(10)],[0.2 for i in range(10)],
        #     [0.1 for i in range(10)],[0.3 for i in range(10)]])
        #np.fill_diagonal(U,0)
        #U_noisy = U + np.random.randn(num_p1_acts,num_p2_acts)*1e-4
        
        out = torch.squeeze(self.Q(z,node_features,edge_index)).detach()*(-2*self.player +1)
        U = torch.mean(out,dim=-1).cpu().numpy()

        if self.player == 0:
            U_calc = -U.T #LP package requires the player to be the column player
        else:
            U_calc = -U
        A_ub = np.hstack((U_calc,np.ones((num_player_actions,1))))
        b_ub = np.zeros(num_player_actions)
        A_eq = np.ones((1,num_player_actions+1))
        A_eq[:,num_player_actions] = 0
        b_eq = 1
        bounds = [(0,None) for i in range(num_player_actions+1)]
        bounds[num_player_actions] = (None,None)
        res = linprog(c,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq,bounds=bounds,options={'maxiter': 1000,'tol': 1e-6})

        if not res["success"]:
            print("Failed to Optimize LP with exit status ",res["status"])
        value = res["fun"]
        policy = res["x"][0:num_player_actions]
        # print('policy: ', policy)
        # if self.player == 1:
        #     exit()
        return policy,U

    def __call__(self, state,num_actions=1,policy=None):
        if self.degree > 1:
            featurized_actions = torch.stack([state[action].flatten() for action in self.all_actions])
        else:
            featurized_actions = torch.tensor(state)            
        with torch.no_grad():
            retries = 0
            retry_lim = 100
            if policy is None:
                policy, value = self.get_policy(featurized_actions)
            try:
                pd = Categorical(torch.tensor(policy))
            except:
                print("Failed util: ",value)
                print("Failed with policy:", policy)
                pd = Categorical(1/len(policy)*torch.ones_like(torch.tensor(policy)))
            actions = []
            num_samples = num_actions / self.degree
            for i in range(num_actions):
                retries=0
                a = pd.sample().item()
                while a in actions:
                    if retries < retry_lim:
                        a = pd.sample().item()
                        retries += 1
                    else:
                        a = self.action_space.sample()
                if num_samples == 1:
                    return self.all_actions[a]
                else:  
                    if type(self.all_actions[a]) == list:
                        for act in self.all_actions[a]: actions.append(act)          
                    else:
                        actions.append(self.all_actions[a])  
            actions.sort()
            return actions

    @property
    def Q(self):
        return self.model

class StochasticPolicy(ModelBasedPolicy):
    """
    The class of stochastic policies
    
    :param model: (Model or torch.nn.Module) The model of the policy 
    :param observation_space: (gym.Spaces) The observation space
    :param action_space: (gym.Spaces) The action space
    """
    
    def __init__(self, model, action_space=None,all_actions=[]):
        super(StochasticPolicy, self).__init__(model)
        self.observation_space = observation_space
        self.action_space = action_space
        self.all_actions = all_actions

        act_dim = gymSpace2dim(self.action_space)
        self.model = model #marl.model.make(model, obs_sp=obs_dim, act_sp=act_dim)
        
    def forward(self, x):
        x = self.model(x)
        pd = Categorical(x)
        return pd


    def get_policy(self,observation,action):
        pd = self.forward(observation)
        return pd.probs.detach().cpu().numpy(), None

    def __call__(self, observation,num_actions=1):
        observation = torch.tensor(observation).float()
        observation = torch.mean(observation,axis=0).float() #move this to a custom action function in the featurized agent
        with torch.no_grad():
            retries = 0
            retry_lim = 100
            pd = self.forward(observation)
            actions = []
            for i in range(num_actions):
                retries=0
                a = pd.sample().item()
                while a in actions:
                    if retries < retry_lim:
                        a = pd.sample().item()
                        retries += 1
                    else:
                        a = self.action_space.sample()
                actions.append(a)   
            if len(actions) == 1:
                return self.all_actions[actions[0]]    
            else:         
                return [self.all_actions[a] for a in actions]
        
class DeterministicPolicy(ModelBasedPolicy):
    """
    The class of deterministic policies
    
    :param model: (Model or torch.nn.Module) The model of the policy
    :param observation_space: (gym.Spaces) The observation space
    :param action_space: (gym.Spaces) The action space
    """
    
    def __init__(self, model, observation_space=None, action_space=None):
        super(DeterministicPolicy, self).__init__()
        
        self.observation_space = observation_space
        self.action_space = action_space
        
        self.low = self.action_space.low[0] if isinstance(self.action_space, gym.spaces.Box) else 0
        self.high = self.action_space.high[0] if isinstance(self.action_space, gym.spaces.Box) else 1
        
        obs_dim = gymSpace2dim(self.observation_space)
        act_dim = gymSpace2dim(self.action_space)
        self.model = marl.model.make(model, obs_sp=obs_dim, act_sp=act_dim)

    def __call__(self, observation,num_actions=1):
        observation = torch.tensor(observation).float()
        with torch.no_grad():
            action = np.array(self.model(observation))
            return np.clip(action, self.low, self.high)