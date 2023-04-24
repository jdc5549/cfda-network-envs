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


class RandomPolicy(Policy):
    """
    The class of random policies
    
    :param model: (Model or torch.nn.Module) The q-value model 
    :param action_space: (gym.Spaces) The action space
    """
    def __init__(self, action_space,num_actions=1,all_actions=[]):
        self.action_space = action_space
        self.num_actions=num_actions
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
            sorted_idx = np.flip(np.argsort(state[j][:,1]))
            sorted_acts = [self.all_actions[i] for i in sorted_idx]
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
    def __init__(self, action_space,pt,test_obs,num_actions=1,all_actions=[]):
        self.num_actions=num_actions
        self.all_actions = all_actions
        self.pt = pt
        self.test_obs = test_obs
        
    def __call__(self, state):
        """
        """
        rn = random.uniform(0,1)
        actions_list = []
        idx = self.test_obs.index(state)
        if rn <= self.pt[idx]:
            for j in range(len(state)):
                sorted_idx = np.flip(np.argsort(state[j][:,1]))
                sorted_acts = [self.all_actions[i] for i in sorted_idx]
                actions = sorted_acts[:self.num_actions]
                if len(actions) > 1:
                    actions_list.append(actions)
                else:
                    actions_list.append(actions[0])
        else:
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
        self.model = marl.model.make(model, obs_sp=gymSpace2dim(self.observation_space), p1_action_space=gymSpace2dim(self.action_space),p2_action_space=gymSpace2dim(self.action_space))
        self.player = player
        self.policy = None
        self.degree = act_degree
        self.all_actions = all_actions

    def _get_state_row(self,i):
        si = np.zeros(self.num_p2_acts,state.numel())
        p1i = np.zeros(self.num_p2_acts,featurized_actions[0].numel())
        p2i = np.zeros(self.num_p2_acts,featurized_actions[0].numel())
        for j in range(self.num_p2_acts):
            si[j] = 0
        return ui

    def get_policy(self,state,featurized_actions):
        #print('t_obs in get_policy:', state)
        num_player_actions = gymSpace2dim(self.action_space)
        num_p1_acts = gymSpace2dim(self.action_space)
        self.num_p2_acts = gymSpace2dim(self.action_space)
        c = np.zeros(num_player_actions+1)
        c[num_player_actions] = -1
        U = np.zeros([num_p1_acts,self.num_p2_acts])
        # U = (-2*self.player +1) * np.array([[0.1 for i in range(10)],[0.1 for i in range(10)],
        #     [0.2 for i in range(10)],[0.6 for i in range(10)],[0.1 for i in range(10)],
        #     [0.2 for i in range(10)],[0.2 for i in range(10)],[0.2 for i in range(10)],
        #     [0.1 for i in range(10)],[0.3 for i in range(10)]])
        #np.fill_diagonal(U,0)
        #U_noisy = U + np.random.randn(num_p1_acts,num_p2_acts)*1e-4
        #if num_p1_acts < mp.cpu_count()-2
        #state_t = state.repeat(num_p1_acts,self.num_p2_acts)
        # singleproc_tic = time.perf_counter()
        # #old
        # state_t = torch.zeros(num_p1_acts,self.num_p2_acts,state.numel())   
        # p1_act_t = torch.zeros(num_p1_acts,self.num_p2_acts,featurized_actions[0].numel())        
        # p2_act_t = torch.zeros(num_p1_acts,self.num_p2_acts,featurized_actions[0].numel())
        # for i in range(num_p1_acts):
        #     for j in range(self.num_p2_acts):
        #         state_t[i,j] = state
        #         p1_act_t[i,j] = featurized_actions[i]
        #         p2_act_t[i,j] = featurized_actions[j]
        # U = self.Q(state_t,p1_act_t,p2_act_t)
        # print(U)
        # singleproc_toc = time.perf_counter()
        # print(f'single proc time to get state mat: {singleproc_toc-singleproc_tic} seconds')
        #new 
        #singleproc_tic = time.perf_counter()
        state_t = state.repeat(num_p1_acts,self.num_p2_acts,1)
        p1_act_t = torch.unsqueeze(featurized_actions,dim=1)#torch.zeros(num_p1_acts,self.num_p2_acts,featurized_actions[0].numel())
        p1_act_t = p1_act_t.expand(num_p1_acts,self.num_p2_acts,featurized_actions[0].numel())
        p2_act_t = torch.unsqueeze(featurized_actions,dim=0)
        p2_act_t = p2_act_t.expand(num_p1_acts,self.num_p2_acts,featurized_actions[0].numel())
        U = torch.squeeze(self.Q(state_t,p1_act_t,p2_act_t)).cpu().detach().numpy()*(-2*self.player +1)
        #singleproc_toc = time.perf_counter()
        #print(f'single proc time to get state mat: {singleproc_toc-singleproc_tic} seconds')
        #exit()
        # multiproc_tic = time.perf_counter()
        # i_vals = [i for i in range(num_p1_acts)]
        # state_t = torch.repeat(state)
        # with mp.Pool(processes=min([num_p1_acts,mp.cpu_count()-2])) as pool:
        #     U = np.array(pool.map(self._get_state_row, i_vals))
        # pool.join()
        # mutliproc_toc = time.perf_counter()
        # print(f'multi proc get Umat: {mutliproc_toc-multiproc_tic} seconds')
        # exit()
        if self.player == 0:
            U_calc = -U.T #LP package requires the player to be the column player
        else:
            U_calc = -U
        #print('U_calc: ',U_calc)
        A_ub = np.hstack((U_calc,np.ones((num_player_actions,1))))
        b_ub = np.zeros(num_player_actions)
        A_eq = np.ones((1,num_player_actions+1))
        A_eq[:,num_player_actions] = 0
        b_eq = 1
        bounds = [(0,None) for i in range(num_player_actions+1)]
        bounds[num_player_actions] = (None,None)
        res = linprog(c,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq,bounds=bounds,options={'maxiter': 100,'tol': 1e-6})

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


class StochasticPolicy(ModelBasedPolicy):
    """
    The class of stochastic policies
    
    :param model: (Model or torch.nn.Module) The model of the policy 
    :param observation_space: (gym.Spaces) The observation space
    :param action_space: (gym.Spaces) The action space
    """
    
    def __init__(self, model, observation_space=None, action_space=None,all_actions=[]):
        super(StochasticPolicy, self).__init__(model)
        self.observation_space = observation_space
        self.action_space = action_space
        self.all_actions = all_actions

        obs_dim = gymSpace2dim(self.observation_space)
        act_dim = gymSpace2dim(self.action_space)
        self.model = marl.model.make(model, obs_sp=obs_dim, act_sp=act_dim)
        
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