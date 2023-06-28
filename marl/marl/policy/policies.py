import numpy as np
import time
import random
import multiprocessing as mp
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

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
    def __init__(self, action_space,num_actions=1,all_actions=[]):
        self.action_space = action_space
        self.num_actions=num_actions
        self.all_actions = all_actions
        
    def __call__(self, state,num_actions=1):
        """
        Return the highest degree nodes
        
        :param state: (Tensor) The current state
        """
        actions_list = []
        for j in range(len(state)):
            G = nx.from_numpy_matrix(state[j][:-1])
            node_degrees = [G.degree(n) for n in G.nodes]
            act_degrees = [node_degrees[act[0]] + node_degrees[act[1]] for act in self.all_actions]
            sorted_idx = np.flip(np.argsort(act_degrees))
            sorted_acts = [self.all_actions[idx] for idx in sorted_idx]
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
        self.num_actions=num_actions
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
    def __init__(self, model, action_space):
        self.action_space = action_space
        self.all_actions = get_combinatorial_actions(action_space,2)
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
    def __init__(self, model, action_space = None,player=0,all_actions=[],eval_mode=False,device='cpu'):
        self.action_space = action_space
        self.model = model
        self.player = player
        self.all_actions = all_actions
        self.action_indices = torch.tensor([i for i in range(len(self.all_actions))]).to(torch.long).to(device)
        self.eval_mode = eval_mode
        self.policy = None

    def get_policy(self,node_features=None,edge_index=None):
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
        expanded_features = None
        #expanded_features = node_features.unsqueeze(0)
        #expanded_features = expanded_features.repeat(self.action_indices.shape[0],self.action_indices.shape[0],1)
        grid_p1,grid_p2 = torch.meshgrid(self.action_indices,self.action_indices)
        actions = torch.stack((grid_p1,grid_p2),dim=-1)
        #p1_act_t = torch.unsqueeze(featurized_actions,dim=1)#torch.zeros(num_p1_acts,self.num_p2_acts,featurized_actions[0].numel())
        #p1_act_t = p1_act_t.expand(num_p1_acts,num_p2_acts,featurized_actions[0].numel())
        #p2_act_t = torch.unsqueeze(featurized_actions,dim=0)
        #p2_act_t = p2_act_t.expand(num_p1_acts,num_p2_acts,featurized_actions[0].numel())
        if edge_index is not None:
            out = torch.squeeze(self.Q(actions,expanded_features,edge_index)).detach()*(-2*self.player +1)
        else:
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
        res = linprog(c,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq,bounds=bounds,options={'maxiter': 1000,'tol': 1e-6})

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