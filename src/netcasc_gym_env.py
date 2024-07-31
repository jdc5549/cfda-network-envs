import time
import networkx as nx
import numpy as np
import torch
import math
import random
import gym
from gym import spaces
import sys
from scm import SCM
from graph_embedding import heuristic_feature_embedding
from utils import create_random_nets, ncr

class NetworkCascEnv(gym.Env):
    def __init__(self,p_atk,p_def,net_type,embed_size=1,degree=2,cascade_type='threshold',filename=None,discrete_obs=False,topo_eps=None,net_size=None):
        super(NetworkCascEnv,self).__init__()
        self.discrete_obs = discrete_obs
        self.p_atk = p_atk
        self.p_def = p_def
        self.network_type = net_type
        self.filename = filename
        self.cascade_type = cascade_type
        self.episode = 0
        self.topo_eps = topo_eps
        self.topo_count = 0
        immunization = False
        self.immunization = immunization

        if self.network_type == 'File' and filename is not None:
            if isinstance(self.filename,str):
                self.net = nx.read_gpickle(self.filename) 
                if self.cascade_type == 'coupled':
                    self.scm = SCM(self.net,cascade_type=self.cascade_type,comm_net=self.net.copy())
                else:
                    self.scm = SCM(self.net,cascade_type=self.cascade_type)
                self.obs = nx.to_numpy_array(self.net)
                self.fid = 0
            else:
                self.obs = []
                for fn in self.filename:
                    self.net = nx.read_gpickle(fn) 
                    if self.cascade_type == 'coupled':
                        self.scm = SCM(self.net,cascade_type=self.cascade_type,comm_net=self.net.copy())
                    else:
                        self.scm = SCM(self.net,cascade_type=self.cascade_type)
                    self.obs.append(nx.to_numpy_array(self.net))
                self.fid = len(self.filename)-1
        else:
            comm_net,self.net  = create_random_nets('', net_size,num2gen=1,show=False)
            if self.cascade_type == 'coupled':
                self.scm = SCM(self.net,cascade_type=self.cascade_type,comm_net=comm_net)
            else:
                self.scm = SCM(self.net,cascade_type=self.cascade_type)
        self.net_size = self.net.number_of_nodes()
        if self.p_atk < 1:
            self.num_nodes_attacked = int(math.floor(self.p_atk * self.net_size))
        else:
            self.num_nodes_attacked = int(self.p_atk)
        #print("Attacking {} of {} Nodes".format(self.num_nodes_attacked,self.net_b.number_of_nodes()))
        if self.p_def < 1:
            self.num_nodes_defended = int(math.floor(self.p_def * self.net_size))
        else:
            self.num_nodes_defended = int(self.p_def)        #print("Defending {} of {} Nodes".format(self.num_nodes_defended,self.net_b.number_of_nodes()))
        self.name = "NetworkCascEnv-v0"
        self.num_envs = 1
        self.degree = degree
        self.action_space = spaces.Discrete(ncr(self.net_size,degree))
        thresh_true = self.cascade_type == 'threshold' or self.cascade_type == 'shortPath'
        net_feature_size = embed_size+1
        self.observation_space = spaces.Box(low=-1,high=1,shape=(self.net_size,net_feature_size),dtype=np.float32)

    def step(self,node_lists):
        node_list_atk = node_lists[0]
        node_list_def = node_lists[1]
        if self.immunization:
            immunized_set = self.scm.check_cascading_failure(node_list_def)
            potential_fail_set = self.scm.check_cascading_failure(node_list_atk)
            fail_set = [n for n in potential_fail_set if n not in immunized_set]
        else:
            final_node_list = []
            if type(node_list_atk) == int:
                if node_list_atk != node_list_def:
                    final_node_list.append(node_list_atk)
            else:
                for node in node_list_atk:
                    if node not in node_list_def:
                        final_node_list.append(node)
            fail_set = self.scm.check_cascading_failure(final_node_list)
        self.scm.reset()
        r = len(fail_set)/self.net.number_of_nodes()
        reward = [r,-r]
        done = [True,True]
        env_id = self.fid if self.network_type == 'File' else self.topo_count
        if self.immunization:
            info = {'immunized_set': immunized_set, 'potential_fail_set': potential_fail_set, 'fail_set': fail_set,'edges': self.scm.G.edges(),'env_id': env_id}
        else:
            info = {'init_fail':final_node_list,'fail_set':fail_set,'edges': self.scm.G.edges(),'env_id': env_id}
        obs2 = self.observation_space.sample()
        return obs2,reward,done,info

    def reset(self,fid=None):
        thresh_true = self.cascade_type == 'threshold' or self.cascade_type == 'shortPath'
        if self.topo_eps is not None:
            if self.episode < self.topo_eps:
                self.episode += 1
                obs = nx.to_numpy_array(self.net)
                #if thresh_true: obs=np.append(obs,[self.scm.thresholds],axis=0)
                return obs
            else:
                #print('new topo in env')
                self.topo_count += 1
        self.episode = 1
        if self.network_type == 'SF':
            comm_net,self.net = create_random_nets('',self.net_size,num2gen=1,show=False)
            if self.cascade_type == 'coupled':
                self.scm = SCM(self.net,cascade_type=self.cascade_type,comm_net=comm_net)
            else:
                self.scm = SCM(self.net,cascade_type=self.cascade_type)
            obs = nx.to_numpy_array(self.net)
            #if thresh_true: obs=np.append(obs,[self.scm.thresholds],axis=0)
            return obs
        elif self.network_type == 'File':
            if not isinstance(self.filename,str):
                if fid is None:
                    self.fid = random.choice([i for i in range(len(self.filename))])
                else:
                    self.fid = fid
                fn = self.filename[self.fid]
                #self.net = nx.read_edgelist(fn,nodetype=int)
                self.net = nx.read_gpickle(fn)
                thresholds = np.load(fn[:-9] + '_thresh.npy')                
                if self.cascade_type == 'coupled':
                    self.scm = SCM(self.net,comm_net=self.net.copy(),cascade_type=self.cascade_type)
                else:
                    self.scm = SCM(self.net,thresholds=thresholds,cascade_type=self.cascade_type)
            obs = nx.to_numpy_array(self.net)
            if thresh_true: obs=np.append(obs,[self.scm.thresholds],axis=0)
            return obs