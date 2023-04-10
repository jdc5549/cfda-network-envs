import time
import networkx as nx
import networkx.algorithms.centrality as central
import numpy as np
# import math
# import gym
from gym import spaces
from netcasc_gym_env import NetworkCascEnv
import sys
from typing import Callable
import random
import os
import copy

sys.path.append('./marl/')
import marl
from marl.agent import DQNAgent, DQNCriticAgent,MinimaxDQNCriticAgent,FeaturizedACAgent,MinimaxQAgent
from marl.model.qvalue import MultiQTable
from marl.exploration.eps_greedy import EpsGreedy
from marl.experience.replay_buffer import ReplayMemory
from marl.model.nn.mlpnet import CriticMlp,MultiCriticMlp,MlpNet,GumbelMlpNet
from marl.policy.policies import RandomPolicy,HeuristicPolicy,QCriticPolicy
from marl.tools import gymSpace2dim,ncr,get_combinatorial_actions
from marl.agent.agent import Agent
from marl.agent.cfa_agent import CFA_MinimaxDQNCriticAgent
from marl.marl import MARL_with_exploiters
sys.path.append('../stable-baselines3')
from stable_baselines3.common.vec_env import SubprocVecEnv,VecMonitor
from stable_baselines3.common.utils import set_random_seed
from scm import SCM

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, deque
from utils import create_random_nets

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Netcasc Gym Args')
    parser.add_argument("--train",default=False,type=bool,help='Trains a model when true, evaluates a model when false. Default True.')
    parser.add_argument("--training_epochs",default=10,type=int,help='How many epochs of training to do when updating the model')
    parser.add_argument("--cascade_type",default='threshold',type=str,help='Type of cascading model to use in cascading failure simulation.')
    parser.add_argument("--cfa",default=False,type=bool,help='Whether to use CFA for training.')
    parser.add_argument("--topo_eps",default=10,type=int,help='How many factual episodes to generate for each topology.')
    parser.add_argument("--training_steps",default=100e3,type=int,help='Number of steps to train the model if we are training.')
    parser.add_argument("--learning_rate",default=0.1,type=float,help='Reinforcement Learning rate.')
    parser.add_argument("--sched_step",default=50e3,type=int,help='How often to reduce the learning rate for training NN model')
    parser.add_argument("--sched_gamma",default=0.1,type=float,help='How much to reduce the learning rate after shed_step steps')
    parser.add_argument("--eps_deb",default=1.0,type=float,help='How much exploration to process when exploration begins')
    parser.add_argument("--eps_fin",default=0.01,type=float,help='How much exploration to process when exploration ends')
    parser.add_argument("--deb_expl",default=0,type=float,help='When (as fraction of total training steps) to begin exploration')
    parser.add_argument("--fin_expl",default=0.9,type=float,help='When (as fraction of total training steps) to end exploration')
    parser.add_argument("--batch_size",default=64,type=int,help='Batch size for NN training')
    parser.add_argument("--test_freq",default=10e3,type=int,help='Frequency at which to test the agent performance during training.')
    parser.add_argument("--save_freq",default=10e3,type=int,help='Frequency at which to save the agent model during training.')
    #parser.add_argument("--exploiters",default=False,type=bool,help='Whether to train exploiters. Default False.')
    #parser.add_argument("--exploited_type",default='NN',type=str,help='What type of agent to train exploiters against. Valid choices are: NN,Random, and Heuristic.')
    parser.add_argument("--testing_episodes",default=1000,type=int,help='Number of testing episodes for evaluation (when train is false).')
    parser.add_argument("--ego_model_dir",default=None,type=str,help='dir where nn model to load is for the ego agents')
    #parser.add_argument("--exploiter_model_dir",default=None,type=str,help='dir where nn model to load is for the exploiter agents')
    parser.add_argument("--p",default=0.1,type=float,help='Fraction of total nodes to be attacked/defended. Default 0.1')
    parser.add_argument("--degree",default=1,type=int,help='Number of nodes selected by the agent policy at a time. Default 1.')
    parser.add_argument("--embed_type",default='heuristic',help='Method for embedding a graph into a feature vector. Default "heuristic".')
    parser.add_argument("--embed_size",default=64,type=int,help='If using an autoencoder for embedding, determines the size of the embedded vector.')
    parser.add_argument("--net_type",default='SF',type=str,help='Strategy for network creation. Use "SF" for random net, or "File" to load the network in "net_file_train_dir".')
    parser.add_argument("--net_size",default=10,type=int,help='Number of nodes in the power network.')
    parser.add_argument("--net_file_train_dir",default=None,type=str,help='If "net_type" == "File", loads the network topologies from this file into the environment.')
    parser.add_argument("--mlp_hidden_size",default=64,type=int,help='Hidden layer size for MLP nets used for RL agent.')
    parser.add_argument("--discrete_obs",default=False,type=bool,help='When true, uses an integer index for the state. Only compatible with "File" net_type argument.')
    #parser.add_argument("--tabular_q",default=False,type=bool,help='Use tabular Q-learning instead of neural network Q learning')
    parser.add_argument("--nash_eqs_dir",default=None,type=str,help='Directory where Nash EQ benchmarks are stored')
    parser.add_argument("--test_nets_dir",default=None,type=str,help='Directory where the network topologies for testing are stored')
    parser.add_argument("--exp_name",default='my_exp',type=str,help='Name of experiment that will be associated with log and model files.')

    tic = time.perf_counter()
    args = parser.parse_args()
    num_acts = len(get_combinatorial_actions(args.net_size,args.degree))
    if num_acts > 225 and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.cuda.device(0)
        print('Using Device: cuda')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
        print('Using Device: cpu')
    args = parser.parse_args()
    if args.nash_eqs_dir is not None:
        fns = [f for f in os.listdir(args.nash_eqs_dir)]
        fns.sort()
        eqs = [np.load(os.path.join(args.nash_eqs_dir,f)) for f in fns if f'{args.cascade_type}Casc_eq_' in f]
        utils = [np.load(os.path.join(args.nash_eqs_dir,f)) for f in fns if f'{args.cascade_type}Casc_util_' in f]
        # for i,u in enumerate(utils):
        #     print(i)
        #     print(u)
        # exit()
    else:
        eqs = None
        utils = None
    # if args.discrete_obs:
    #     obs_sp = spaces.Discrete(len(self.filename))
    #     net_feature_size = 1
    # else:
    #     net_feature_size = self.get_obs()[0].shape[-1]
    #     self.observation_space = spaces.Box(low=-1,high=1,shape=(self.net.number_of_nodes(),net_feature_size),dtype=np.float32)
    # act_sp = env.action_space
    # num_samples = int(env.num_samples/args.degree)
    if args.train: 
        topo_eps = args.topo_eps
        hparams = {"training_steps": args.training_steps, "learning_rate": args.learning_rate, "sched_step": args.sched_step, "sched_gamma":args.sched_gamma,
            "training_epochs": args.training_epochs, "cascade_type": args.cascade_type, "cfa": args.cfa, "topo_eps": args.topo_eps, "batch_size": args.batch_size,
            "mlp_hidden_size": args.mlp_hidden_size, "embed_size": args.embed_size, "net_type": args.net_type, "net_size": args.net_size, "p": args.p, "degree": args.degree}
        fact_experience = ReplayMemory(capacity=int(args.training_steps))
        if args.cfa:
            cfact_experience = ReplayMemory(capacity=int(args.training_steps*10))
        else:
            cfact_experience = None
        if args.embed_type == 'GCN':
            from graph_embedding import GCN_Encoder
            embed_size = args.embed_size
            embed_model = GCN_Encoder(embed_size,args.mlp_hidden_size,args.net_size,True)
        elif args.embed_type == 'heuristic':
            embed_size = 6
            embed_model = None
        else:
            print(f'Node embedding type {args.embed_type} not recognized.')
            exit()
        #agent_train = not args.exploiters
        if args.ego_model_dir is not None:
            attacker_model_file = os.path.join(args.ego_model_dir + 'Attacker/',os.listdir(args.ego_model_dir + 'Attacker/')[-1])
            defender_model_file = os.path.join(args.ego_model_dir + 'Defender/',os.listdir(args.ego_model_dir + 'Defender/')[-1])
        else:
            attacker_model_file = None
            defender_model_file = None
        # if args.tabular_q:
        #     attacker_agent = MinimaxQAgent(obs_sp, act_sp, act_sp,act_degree=args.degree,index=0,experience=experience,exploration=exploration, batch_size=args.batch_size*num_samples,
        #         name = 'Attacker',lr=args.learning_rate)
        #     defender_agent = MinimaxQAgent(obs_sp, act_sp, act_sp,act_degree=args.degree,index=1,experience=experience,exploration=exploration, batch_size=args.batch_size*num_samples,
        #         name = 'Defender',lr=args.learning_rate)
        # else: 
        if args.net_type == 'File':
            print('Initializing Environments from File...')
            fns = [os.path.join(args.net_file_train_dir,f) for f in os.listdir(args.net_file_train_dir) if 'thresh' not in f]
            training_env = NetworkCascEnv(args.net_size,args.p,args.p,embed_size,'File',discrete_obs=args.discrete_obs,degree=args.degree,topo_eps=topo_eps,cascade_type=args.cascade_type,
            filename = fns)
            print('Done.')
        else:
            training_env = NetworkCascEnv(args.net_size,args.p,args.p,embed_size,args.net_type,degree=args.degree,discrete_obs=args.discrete_obs,topo_eps=topo_eps,cascade_type=args.cascade_type,filename=args.net_file_train_dir)
        obs_sp = training_env.observation_space
        act_sp = training_env.action_space
        num_samples = int(training_env.num_nodes_attacked/args.degree)
        exploration = EpsGreedy(eps_deb=args.eps_deb, eps_fin=args.eps_fin, deb_expl=0, fin_expl=args.fin_expl,num_actions=num_samples)

        qmodel = MultiCriticMlp(gymSpace2dim(obs_sp)[1],args.degree*gymSpace2dim(obs_sp)[1],args.degree*gymSpace2dim(obs_sp)[1],hidden_size=args.mlp_hidden_size) #action feature space same as obs space
        toc_q = time.perf_counter()
        #print('time to init qmodel: ',toc_q -tic)
        attacker_agent = CFA_MinimaxDQNCriticAgent(qmodel,embed_model,obs_sp,act_sp,act_sp,fact_experience,cfact_experience,training_env,args.training_epochs,args.topo_eps,act_degree=args.degree,index=0,
            exploration=exploration,batch_size=args.batch_size*num_samples,name='Attacker',lr=args.learning_rate,sched_step=args.sched_step,sched_gamma=args.sched_gamma,
            train=args.train,model=attacker_model_file)
        defender_agent = CFA_MinimaxDQNCriticAgent(qmodel,embed_model,obs_sp,act_sp,act_sp,fact_experience,cfact_experience,training_env,args.training_epochs,args.topo_eps,act_degree=args.degree,index=1,
            exploration=exploration,batch_size=args.batch_size*num_samples,name='Defender',lr=args.learning_rate,sched_step=args.sched_step,sched_gamma=args.sched_gamma,
            train=args.train,model=defender_model_file)

        #     env = SimpleCascadeEnv(args.net_size,args.p,args.p,'File',discrete_obs=args.discrete_obs,degree=args.degree,
        #     filename = os.path.join(args.net_file_train_dir,os.listdir(args.net_file_train_dir)[i]))
        #     for j in range(10):
        #         for k in range(10):
        #             _,rew,_,_ = env.step([j,k])
        #             diff = np.abs(rew[0] - utils[i][j][k])
        #             if diff > 0:
        #                 print(f'env: {i}')
        #                 print(f'actions: {j},{k}')
        #                 print(f'util: {utils[i][j][k]}, reward: {rew[0]}')
        # exit()
        # if args.exploiters:
        #     if args.exploited_type == 'NN':
        #         agent_list = [attacker_agent,defender_agent]
        #     elif args.exploited_type == 'Random':
        #         random_policy = RandomPolicy(act_sp,num_actions=num_samples,all_actions= get_combinatorial_actions(gymSpace2dim(obs_sp)[0],args.degree))
        #         agent_list = [copy.deepcopy(random_policy),copy.deepcopy(random_policy)]
        #     elif args.exploited_type == 'Heuristic':
        #         heuristic_policy = HeuristicPolicy(act_sp,num_actions=num_samples,all_actions= get_combinatorial_actions(gymSpace2dim(obs_sp)[0],args.degree))
        #         agent_list = [copy.deepcopy(heuristic_policy),copy.deepcopy(heuristic_policy)]

        #     qmodel_explt = CriticMlp(gymSpace2dim(obs_sp)[1],args.degree*gymSpace2dim(obs_sp)[1]) #action feature space same as obs space
        #     actor = GumbelMlpNet(gymSpace2dim(obs_sp)[1],gymSpace2dim(act_sp))#,last_activ=F.softmax)
        #     def_expltr = FeaturizedACAgent(copy.deepcopy(qmodel_explt),copy.deepcopy(actor),obs_sp,act_sp,experience=copy.deepcopy(experience),exploration=copy.deepcopy(exploration),batch_size=args.batch_size*num_samples,name='Defender Exploiter',
        #        lr_critic=1e-3,lr_actor=1e-4,act_degree=args.degree)#,target_update_freq=1000,name='Attacker')
        #     atk_expltr = FeaturizedACAgent(copy.deepcopy(qmodel_explt),copy.deepcopy(actor),obs_sp,act_sp,experience=copy.deepcopy(experience),exploration=copy.deepcopy(exploration),batch_size=args.batch_size*num_samples,name='Attacker Exploiter',
        #        lr_critic=1e-3,lr_actor=1e-4,act_degree=args.degree)
        #     #def_expltr = DQNCriticAgent(copy.deepcopy(qmodel_explt),obs_sp,act_sp,experience=copy.deepcopy(experience),lr=0.001,batch_size=args.batch_size,name="Defender Exploiter",
        #     #        train=True,model=None,act_degree=args.degree)
        #     #atk_expltr = DQNCriticAgent(copy.deepcopy(qmodel_explt),obs_sp,act_sp,experience=copy.deepcopy(experience),lr=0.001,batch_size=args.batch_size,name="Attacker Exploiter",
        #     #        train=True,model=None,act_degree=args.degree)
        #     mas = MARL_with_exploiters(agent_list,[def_expltr,atk_expltr],log_dir='marl_logs',name=args.exp_name,obs=[ob[0] for ob in env.obs],nash_policies=eqs,
        #             exploited=args.exploited_type,explt_opp_update_freq=args.test_freq)
        # else:
        mas = marl.MARL([attacker_agent,defender_agent],name=args.exp_name,log_dir='marl_logs',nash_policies=eqs,utils=utils,act_degree=args.degree,hparams=hparams)
        # if args.exploited_type == 'NN':
        #     attacker_agent.set_mas(mas)
        #     defender_agent.set_mas(mas)
        test_envs = [NetworkCascEnv(args.net_size,args.p,args.p,embed_size,'File',discrete_obs=args.discrete_obs,degree=args.degree,cascade_type=args.cascade_type,
            filename = os.path.join(args.test_nets_dir,f)) for f in os.listdir(args.test_nets_dir) if 'thresh' not in f]
        toc = time.perf_counter()
        print(f'overall time to start learn {toc - tic}')
        #tic = time.perf_counter()
        mas.learn(training_env,nb_timesteps=args.training_steps,test_freq=args.test_freq,save_freq=args.save_freq,multi_proc=False,verbose=2,test_envs=test_envs)#,exploiters=args.exploiters)
        toc = time.perf_counter()
        print(f"Completed in {toc - tic:0.4f} seconds")
    else:
        # if args.exploiters:
        #     if args.exploited_type == 'NN':
        #         attacker_model_file = os.path.join(args.ego_model_dir + 'Attacker/',os.listdir(args.ego_model_dir + 'Attacker/')[-1])
        #         defender_model_file = os.path.join(args.ego_model_dir + 'Defender/',os.listdir(args.ego_model_dir + 'Defender/')[-1])
        #         if args.tabular_q:
        #             attacker_agent = MinimaxQAgent(obs_sp, act_sp, act_sp,act_degree=args.degree,index=0,batch_size=args.batch_size*num_samples,
        #                 name = 'Attacker',train=False,model=attacker_model_file)
        #             defender_agent = MinimaxQAgent(obs_sp, act_sp, act_sp,act_degree=args.degree,index=1, batch_size=args.batch_size*num_samples,
        #                 name = 'Defender',train=False,model=defender_model_file)
        #         else:
        #             qmodel= MultiCriticMlp(gymSpace2dim(obs_sp)[1],args.degree*gymSpace2dim(obs_sp)[1],args.degree*gymSpace2dim(obs_sp)[1],hidden_size=args.mlp_hidden_size) #action feature space same as obs space
        #             attacker_agent = MinimaxDQNCriticAgent(copy.deepcopy(qmodel),obs_sp,act_sp,act_sp,act_degree=args.degree,index=0,name='Attacker',train=False,model=attacker_model_file)
        #             defender_agent = MinimaxDQNCriticAgent(copy.deepcopy(qmodel),obs_sp,act_sp,act_sp,act_degree=args.degree,index=1,name='Defender',train=False,model=defender_model_file)
        #         agent_list = [attacker_agent,defender_agent]
        #     elif args.exploited_type == 'Random':
        #         random_policy = RandomPolicy(act_sp,num_actions=num_samples,all_actions= get_combinatorial_actions(gymSpace2dim(obs_sp)[0],args.degree))
        #         agent_list = [copy.deepcopy(random_policy),copy.deepcopy(random_policy)]
        #     elif args.exploited_type == 'Heuristic':
        #         heuristic_policy = HeuristicPolicy(act_sp,num_actions=num_samples,all_actions= get_combinatorial_actions(gymSpace2dim(obs_sp)[0],args.degree))
        #         agent_list = [copy.deepcopy(heuristic_policy),copy.deepcopy(heuristic_policy)]
        #     if args.tabular_q:
        #         qmodel_explt = None
        #         actor = GumbelMlpNet(gymSpace2dim(obs_sp),gymSpace2dim(act_sp))
        #     else:
        #         qmodel_explt = CriticMlp(gymSpace2dim(obs_sp)[1],args.degree*gymSpace2dim(obs_sp)[1]) #action feature space same as obs space
        #         actor = GumbelMlpNet(gymSpace2dim(obs_sp)[1],gymSpace2dim(act_sp))#,last_activ=F.softmax)
        #     def_expltr_model_file = os.path.join(args.exploiter_model_dir + 'Defender Exploiter/',os.listdir(args.exploiter_model_dir + 'Defender Exploiter/')[-1])
        #     def_expltr = FeaturizedACAgent(copy.deepcopy(qmodel_explt),copy.deepcopy(actor),obs_sp,act_sp,name='Defender Exploiter',act_degree=args.degree,model=def_expltr_model_file,train=False)
        #     atk_expltr_model_file = os.path.join(args.exploiter_model_dir + 'Attacker Exploiter/',os.listdir(args.exploiter_model_dir + 'Attacker Exploiter/')[-1])
        #     atk_expltr = FeaturizedACAgent(copy.deepcopy(qmodel_explt),copy.deepcopy(actor),obs_sp,act_sp,name='Attacker Exploiter',act_degree=args.degree,model=atk_expltr_model_file,train=False)
        #     mas = MARL_with_exploiters(agent_list,[def_expltr,atk_expltr],name=args.exp_name,obs=[ob[0] for ob in env.obs],nash_policies=eqs,exploited=args.exploited_type)
        # else:
        attacker_model_file = os.path.join(args.ego_model_dir + 'Attacker/',os.listdir(args.ego_model_dir + 'Attacker/')[-1])
        defender_model_file = os.path.join(args.ego_model_dir + 'Defender/',os.listdir(args.ego_model_dir + 'Defender/')[-1])
        # if args.tabular_q:
        #     attacker_agent = MinimaxQAgent(obs_sp, act_sp, act_sp,act_degree=args.degree,index=0,batch_size=args.batch_size*num_samples,
        #         name = 'Attacker',train=False,model=attacker_model_file)
        #     defender_agent = MinimaxQAgent(obs_sp, act_sp, act_sp,act_degree=args.degree,index=1, batch_size=args.batch_size*num_samples,
        #         name = 'Defender',train=False,model=defender_model_file)
        # else:
        #     if args.discrete_obs:
        #         attacker_agent = MinimaxQAgent(obs_sp, act_sp, act_sp,act_degree=args.degree,index=0,batch_size=args.batch_size*num_samples,
        #             name = 'Attacker',train=False,model=attacker_model_file)
        #         defender_agent = MinimaxQAgent(obs_sp, act_sp, act_sp,act_degree=args.degree,index=1, batch_size=args.batch_size*num_samples,
        #             name = 'Defender',train=False,model=defender_model_file)
        #     else:
        qmodel=MultiCriticMlp(gymSpace2dim(obs_sp)[1],args.degree*gymSpace2dim(obs_sp)[1],args.degree*gymSpace2dim(obs_sp)[1],hidden_size=args.mlp_hidden_size) #action feature space same as obs space
        attacker_agent = CFA_MinimaxDQNCriticAgent(copy.deepcopy(qmodel),obs_sp,act_sp,act_sp,"ReplayMemory-1000","ReplayMemory-1000",env.scm,act_degree=args.degree,index=0,name='Attacker',train=False,model=attacker_model_file)
        defender_agent = CFA_MinimaxDQNCriticAgent(copy.deepcopy(qmodel),obs_sp,act_sp,act_sp,"ReplayMemory-1000","ReplayMemory-1000",env.scm,act_degree=args.degree,index=1,name='Defender',train=False,model=defender_model_file)
        agent_list = [attacker_agent,defender_agent]
        mas = marl.MARL(agent_list,name=args.exp_name,log_dir='marl_logs',nash_policies=eqs,utils=utils,act_degree=args.degree)
        test_envs = [NetworkCascEnv(args.net_size,args.p,args.p,'File',discrete_obs=args.discrete_obs,cascade_type=args.cascade_type,filename = os.path.join(args.test_nets_dir,f)) for f in os.listdir(args.test_nets_dir) if 'thresh' not in f]
        test_dict = mas.test(test_envs,nb_episodes=args.testing_episodes,nashEQ_policies=eqs,utils=utils,render=False)
        save_dict = {}
        save_dict['policies'] = test_dict['policies'].tolist()
        save_dict['ego_attacker_rew'] = np.reshape(test_dict['agent_rewards'][0],[len(test_envs),args.testing_episodes]).tolist()
        # if args.exploiters:
        #     save_dict['exploitability'] =  np.reshape(test_dict['exploitability'][0],[len(test_envs),args.testing_episodes]).tolist()
        #     save_dict['exploiter_rew'] = np.reshape(test_dict['exploiter_rewards'][0],[len(test_envs),args.testing_episodes,2]).tolist()
        if args.nash_eqs_dir is not None:
            save_dict['nash_eqs'] = np.asarray(eqs).tolist()
            save_dict['nash_kl_div'] = np.reshape(test_dict['nash_kl_div'][0],[len(test_envs),args.testing_episodes]).tolist()
            save_dict['critic_err'] = np.reshape(test_dict['util_mse'][0],[len(test_envs),args.testing_episodes]).tolist()

        test_results_dir = 'output/test_results/'
        import json
        save_fn = test_results_dir + args.exp_name + '.json'
        with open(save_fn,'w') as f:
            json.dump(save_dict,f)
        print('Finished Testing. Saved results to ',save_fn)
        print('Ego Attacker Reward: Mean {}, Dev. {}'.format(test_dict['agent_rewards'][1],test_dict['agent_rewards'][2]))
        if args.nash_eqs_dir is not None:
            print('Nash KL Divergence: Mean {}, Dev. {}'.format(test_dict['nash_kl_div'][1],test_dict['nash_kl_div'][2]))
            print('Ego Agents MSE: Mean {}, Dev. {}'.format(test_dict['util_mse'][1],test_dict['util_mse'][2]))

        # if args.exploiters:
        #     print('Exploitability: Mean {}, Dev. {}'.format(test_dict['exploitability'][1],test_dict['exploitability'][2]))
        #     print('Defender Exploiter Reward: Mean {}'.format(test_dict['exploiter_rewards'][1]))
        #     print('Attacker Exploiter Reward: Mean {}'.format(test_dict['exploiter_rewards'][2]))
