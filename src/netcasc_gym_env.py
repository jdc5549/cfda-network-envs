import time
import networkx as nx
import networkx.algorithms.centrality as central
import numpy as np
import math
import random
import gym
from gym import spaces
from src.scm import SCM
from src.utils import create_random_nets, ncr

class NetworkCascEnv(gym.Env):
    def __init__(self,net_size,p_atk,p_def,net_type,degree=1,cascade_type='threshold',filename=None,discrete_obs=False,topo_eps=None):
        super(NetworkCascEnv,self).__init__()
        self.net_size = net_size
        self.discrete_obs = discrete_obs
        self.p_atk = p_atk
        self.p_def = p_def
        self.network_type = net_type
        self.filename = filename
        self.cascade_type = cascade_type
        self.episode = 0
        self.topo_eps = topo_eps
        if self.network_type == 'File' and filename is not None:
            if isinstance(self.filename,str):
                self.net = nx.read_edgelist(self.filename,nodetype=int) 
                if self.cascade_type == 'coupled':
                    self.scm = SCM(self.net,comm_net=self.net,cascade_type=self.cascade_type)
                else:
                    thresholds = np.load(self.filename[:-9] + '_thresh.npy')
                    self.scm = SCM(self.net,thresholds=thresholds,cascade_type=self.cascade_type)
                self.obs = self.get_obs()
            else:
                self.obs = []
                for fn in self.filename:
                    self.net = nx.read_edgelist(fn,nodetype=int) 
                    if self.cascade_type == 'coupled':
                        self.scm = SCM(self.net,comm_net=self.net,cascade_type=self.cascade_type)
                    else:
                        thresholds = np.load(fn[:-9] + '_thresh.npy')
                        self.scm = SCM(self.net,thresholds=thresholds,cascade_type=self.cascade_type)
                    self.obs.append(self.get_obs())
                self.fid = len(self.filename)-1
        else:
            comm_net,self.net  = create_random_nets('', self.net_size,num2gen=1,show=False)
            if self.cascade_type == 'coupled':
                self.scm = SCM(self.net,cascade_type=self.cascade_type,comm_net=comm_net)
            else:
                self.scm = SCM(self.net,cascade_type=self.cascade_type)
        self.num_nodes_attacked = int(math.floor(p_atk * self.net.number_of_nodes()))
        #print("Attacking {} of {} Nodes".format(self.num_nodes_attacked,self.net_b.number_of_nodes()))
        self.num_nodes_defended = int(math.floor(p_def * self.net.number_of_nodes()))
        #print("Defending {} of {} Nodes".format(self.num_nodes_defended,self.net_b.number_of_nodes()))
        self.name = "NetworkCascEnv-v0"
        self.num_envs = 1
        self.degree = degree
        self.action_space = spaces.Discrete(ncr(self.net.number_of_nodes(),degree))
        if discrete_obs:
            self.observation_space = spaces.Discrete(len(self.filename))
            net_feature_size = 1
        else:
            net_feature_size = self.get_obs()[0].shape[-1]
            self.observation_space = spaces.Box(low=-1,high=1,shape=(self.net.number_of_nodes(),net_feature_size),dtype=np.float32)


    def step(self,node_lists):
        node_list_atk = node_lists[0]
        node_list_def = node_lists[1]
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
        info = {'init_fail':final_node_list,'fail_set':fail_set,'edges': self.scm.G.edges()}
        observation = [self.observation_space.sample(),self.observation_space.sample()]
        return observation,reward,done,info
    
    def get_obs(self):
        if self.discrete_obs:
            return [self.fid,self.fid]
        metrics = []

        #tic = time.perf_counter()
        n0 = sorted(self.net.nodes())[0] #recognize 0 vs 1 indexing of node names
        num_nodes = self.net.number_of_nodes()
        nodes = [i for i in range(num_nodes)]
        max_node = max(nodes)
        min_node = min(nodes)
        nodes = [2*(node-min_node)/(max_node-min_node)-1 if (max_node-min_node) != 0 else 0 for node in nodes]
        metrics.append(nodes)
        #toc = time.perf_counter()
        #print('Node Names: ', toc - tic)

        if self.cascade_type == 'threshold' or self.cascade_type == 'shortPath':
            #tic = time.perf_counter()
            max_t = max(self.scm.thresholds)
            min_t= min(self.scm.thresholds)
            norm_thresh = [2*(t-min_t)/(max_t-min_t)-1 if (max_t-min_t) != 0 else 0 for t in self.scm.thresholds]
            metrics.append(norm_thresh)
            #toc = time.perf_counter()
            #print('thresholds: ', toc - tic)

        # A = np.asarray(nx.adjacency_matrix(self.net).todense())
        # for row in A:
        #     metrics.append(row.tolist())
        #tic = time.perf_counter()
        degree_centralities = [central.degree_centrality(self.net)[i+n0] for i in range(num_nodes)]
        max_c = max(degree_centralities)
        min_c = min(degree_centralities)
        degree_centralities = [2*(c-min_c)/(max_c-min_c)-1 if (max_c-min_c) != 0 else 0 for c in degree_centralities]
        metrics.append(degree_centralities)
        #toc = time.perf_counter()
        #print('Degree: ', toc - tic)

        #tic = time.perf_counter()
        eigenvector_centralities = [central.eigenvector_centrality(self.net)[i+n0] for i in range(num_nodes)]
        max_c = max(eigenvector_centralities)
        min_c = min(eigenvector_centralities)
        eigenvector_centralities = [2*(c-min_c)/(max_c-min_c)-1 if (max_c-min_c) != 0 else 0 for c in eigenvector_centralities]
        metrics.append(eigenvector_centralities)
        #toc = time.perf_counter()
        #print('Eigen: ', toc - tic)

        #tic = time.perf_counter()
        closeness_centralities = [central.closeness_centrality(self.net)[i+n0] for i in range(num_nodes)]
        max_c = max(closeness_centralities)
        min_c = min(closeness_centralities)
        closeness_centralities = [2*(c-min_c)/(max_c-min_c)-1 if (max_c-min_c) != 0 else 0 for c in closeness_centralities]
        metrics.append(closeness_centralities)
        #toc = time.perf_counter()
        #print('Closeness: ', toc - tic)

        #tic = time.perf_counter()
        harmonic_centralities = [central.harmonic_centrality(self.net)[i+n0] for i in range(num_nodes)]
        max_c = max(harmonic_centralities)
        min_c = min(harmonic_centralities)
        harmonic_centralities = [2*(c-min_c)/(max_c-min_c)-1 if (max_c-min_c) != 0 else 0 for c in harmonic_centralities]
        metrics.append(harmonic_centralities)
        #toc = time.perf_counter()
        #print('Harmonic: ', toc - tic)

        #tic = time.perf_counter()
        betweenness_centralities = [central.betweenness_centrality(self.net)[i+n0] for i in range(num_nodes)]
        max_c = max(betweenness_centralities)
        min_c = min(betweenness_centralities)
        betweenness_centralities = [2*(c-min_c)/(max_c-min_c)-1 if (max_c-min_c) != 0 else 0 for c in betweenness_centralities]
        metrics.append(betweenness_centralities)
        #toc = time.perf_counter()
        #print('Betweeness: ', toc - tic)

        #tic = time.perf_counter()
        second_order_centralities = [central.second_order_centrality(self.net)[i+n0] for i in range(num_nodes)]
        max_c = max(second_order_centralities)
        min_c = min(second_order_centralities)
        second_order_centralities = [2*(c-min_c)/(max_c-min_c)-1 if (max_c-min_c) != 0 else 0 for c in second_order_centralities]
        metrics.append(second_order_centralities)
        #toc = time.perf_counter()
        #print('Second Order: ', toc - tic)

        # closeness_vitalities = [vital.closeness_vitality(self.net)[i+n0] for i in range(num_nodes)]
        # max_c = max(closeness_vitalities)
        # min_c = min(closeness_vitalities)
        # closeness_vitalities = [2*(c-min_c)/(max_c-min_c)-1 if (max_c-min_c) != 0 else 0 for c in closeness_vitalities]
        # metrics.append(closeness_vitalities)

        #k = min(self.net_b.number_of_nodes(),10)
        # tic = time.perf_counter()
        # flow_betweenness_centralities = [central.current_flow_betweenness_centrality(self.net)[i+n0] for i in range(num_nodes)]
        # max_c = max(flow_betweenness_centralities)
        # min_c = min(flow_betweenness_centralities)
        # flow_betweenness_centralities = [2*(c-min_c)/(max_c-min_c)-1 if (max_c-min_c) != 0 else 0 for c in flow_betweenness_centralities]
        # metrics.append(flow_betweenness_centralities)
        # toc = time.perf_counter()
        # print('Flow Betweeness: ', toc - tic)

        obs = np.stack(metrics).T
        return [obs,obs]

    def reset(self,fid=None):
        if self.topo_eps is not None:
            if self.episode < self.topo_eps:
                self.episode += 1
                return self.get_obs()
        self.episode = 1
        if self.network_type == 'SF':
            comm_net,self.net = create_random_nets('',self.net_size,num2gen=1,show=False)
            if self.cascade_type == 'coupled':
                self.scm = SCM(self.net,cascade_type=self.cascade_type,comm_net=comm_net)
            else:
                self.scm = SCM(self.net,cascade_type=self.cascade_type)
            return self.get_obs()
        elif self.network_type == 'File':
            if not isinstance(self.filename,str):
                if fid is None:
                    self.fid = random.choice([i for i in range(len(self.filename))])
                else:
                    self.fid = fid
                fn = self.filename[self.fid]
                self.net = nx.read_edgelist(fn,nodetype=int)
                thresholds = np.load(fn[:-9] + '_thresh.npy')
                self.scm = SCM(self.net,thresholds=thresholds,cascade_type=self.cascade_type)
            return self.get_obs()