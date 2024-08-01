import os
import sys
import numpy as np
import torch
import torch.nn as nn
import itertools

from scipy.stats import entropy
from torch.utils.data import DataLoader

from utils import get_combinatorial_actions, ncr
from graph_embedding import get_featurized_obs
from NetCascDataset import NetCascDataset
sys.path.append('./marl/')
from marl.policy import MinimaxQCriticPolicy,SubactMinimaxQCriticPolicy
from models import MLP_Critic, GCN_Critic, GAT_Critic
from marl.tools import gymSpace2dim

class Validator():
	def __init__(self,envs,p=2,embedding=None,subact_sets=None,dataset=None,nash_eqs_dir=None,device='cpu',exploiter_model_dir=None,gnn=False):
		self.envs = envs
		self.gnn = gnn
		#self.embedding = embedding
		self.device = device
		self.obs_space = envs[0].observation_space
		self.act_space = envs[0].action_space
		num_nodes = gymSpace2dim(self.obs_space)[0]
		self.p = p
		self.all_actions = get_combinatorial_actions(num_nodes,p)
		#self.train_set = self.get_validation_set(subact_sets)
		# if os.path.isdir(nash_eqs_dir):
		#Get Nash EQs
		if nash_eqs_dir is not None and os.path.isdir(nash_eqs_dir):
			fns = [f for f in os.listdir(nash_eqs_dir)]
			fns.sort()
			self.nashEQ_policies = [np.load(os.path.join(nash_eqs_dir,f)) for f in fns if (f'eq' in f and 'cluster' not in f)]
			self.utils = [np.load(os.path.join(nash_eqs_dir,f)) for f in fns if (f'util' in f and 'cluster' not in f)]
		else:
			self.nashEQ_policies = []
			self.utils = []

		if dataset is not None:
			self.dataset = dataset
			if self.dataset.__len__() > 0:
				self.data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
			else:
				self.data_loader = None
		else:
			self.data_loader = None
	# def get_validation_set(self,subact_sets):
	# 	train_set = []
	# 	for subset in subact_sets:
	# 		all_act_indicies = []
	# 		for i in range(len(subset)):
	# 			for j in range(i+1,len(subset)):
	# 				train_act = (subset[i],subset[j])
	# 				all_act_idx = self.all_actions.index(train_act)
	# 				all_act_indicies.append(all_act_idx)
	# 		for a1 in all_act_indicies:
	# 			for a2 in all_act_indicies:
	# 				train_set.append((a1,a2))
	# 	train_set = list(set(train_set))
	# 	# validation_set = []
	# 	# for i in range(len(self.all_actions)):
	# 	# 	for j in range(len(self.all_actions)):
	# 	# 		if (i,j) not in train_set:
	# 	# 			validation_set.append((i,j))
	# 	return train_set

	def validate(self,q_model,device='cpu'):
		val_err = None
		util_errs = []
		nash_eq_divergences = []
		if q_model is not None:
			criterion = nn.SmoothL1Loss()
			with torch.no_grad():
				#Test on Validation Trial Data
				if self.data_loader is not None:
					val_err = 0
					for i, data in enumerate(self.data_loader):
						if self.gnn:
							(node_features,edge_index,actions),(reward,multi_hot_failures) = data
							edge_index = edge_index.to(device)
						else:
							(node_features,actions), (reward,_) = data
							edge_index=None

						node_features = node_features.to(device)
						actions = actions.to(device)
						B = reward.shape[0]
						reward = reward.to(device)

						if self.gnn:
							pred = q_model(actions,node_features,edge_index)
						else:
							pred = q_model(actions,node_features)
						#multi_hot_pred = torch.zeros_like(pred)
						#multi_hot_pred[pred > 0.5] = 1
						pred_reward = torch.mean(pred,dim=1)
						val_err += np.sum(np.abs((pred_reward-reward).detach().cpu().numpy()))/self.dataset.__len__()
				#Compare to ground truth utility and NashEQ if available
				if len(self.nashEQ_policies) > 0:
					G = self.envs[0].net
					num_nodes = G.number_of_nodes()
					node_features = torch.tensor([G.nodes[n]['threshold'] for n in range(num_nodes)])
					node_features = node_features.to(device)
					edge_index = from_networkx(G).edge_index
					for i,env in enumerate(self.envs):
						atk_policy = SubactMinimaxQCriticPolicy(q_model,action_space=self.act_space,player=0,all_actions=self.all_actions,device=device)
						def_policy = SubactMinimaxQCriticPolicy(q_model,action_space=self.act_space,player=1,all_actions=self.all_actions,device=device)
						if self.gnn:
							atk_pd,atk_Q_val = atk_policy.get_policy(node_features,edge_index)
							def_pd,def_Q_val = def_policy.get_policy(node_features,edge_index)
						else:
							if True:
								atk_pd,atk_Q_val = atk_policy.get_policy(node_features)
								def_pd,def_Q_val = def_policy.get_policy(node_features)
							else:
								atk_pd,atk_Q_val = atk_policy.get_large_policy(node_features)
								def_pd,def_Q_val = def_policy.get_large_policy(node_features)
						nashEQ_policy = self.nashEQ_policies[i]
						base_policy = np.array([atk_pd,def_pd])
						bp_size = len(base_policy[0].flatten())
						neq_size = len(nashEQ_policy[0].flatten())
						#RCR
						if len(nashEQ_policy[0].flatten()) == bp_size:
							impl_policy = base_policy
						else:
							RCR_sizes = [ncr(num_nodes,i) for i in range(1,5)]
							if neq_size in RCR_sizes:
								k = RCR_sizes.index(neq_size)+1
							else:
								print('Could not identify RCR m.')
								exit()
							km_ratio = int(k/args.p)
							impl_policy = []
							for bp in base_policy:
								for combo in itertools.combinations(base_policy[0], km_ratio):
									# Calculate the product of probabilities for the current combination
									joint_prob = np.prod(combo)
									# Append the joint probability to the list
									impl_policy.append(joint_prob)
							# Convert the list to a np.array
							impl_policy = np.array(impl_policy)
							impl_policy = np.reshape(impl_policy,(2,-1))

						best_kl = np.inf
						reg = np.ones_like(nashEQ_policy.flatten())*1e-6
						kl = entropy(impl_policy.flatten()+reg,nashEQ_policy.flatten()+reg)
						nash_eq_divergences.append(kl)
						if self.utils[i].shape == atk_Q_val.shape:
							err_mat = [atk_Q_val-self.utils[i],def_Q_val+self.utils[i]]
							err = []
							#val_err = []
							#debug_high_err = {}
							#for j, errj in enumerate(err_mat):
							for k, errk in enumerate(err_mat[1]):
								for l,errl in enumerate(errk):
									abs_errl = np.abs(errl)
									err.append(abs_errl)
										# if (k,l) not in self.train_set:
										# 	val_err.append(abs_errl)
										#if abs_errl > 1 and j == 0:
										#	debug_high_err[f'{[self.all_actions[k],self.all_actions[l]]}'] = [atk_Q_val[k,l],self.utils[i][k,l]]s
							util_errs.append(np.mean(err))
		util_err_ret = np.mean(util_errs) if len(util_errs) > 0 else None
		nash_div_ret = np.mean(nash_eq_divergences) if len(nash_eq_divergences) > 0 else None
		return val_err,util_err_ret,nash_div_ret


if __name__ == '__main__':	
	import argparse
	parser = argparse.ArgumentParser(description='Netcasc SL Training Args')
	parser.add_argument("--data_dir",default=None,type=str,help='Directory to load data from.')
	parser.add_argument("--p",default=2,type=int,help='Number of nodes chosen by attacker/defender.')
	parser.add_argument("--q_model_path",default=None,type=str,help='Path of q_model to evaluate.')
	parser.add_argument("--net_size",default=5,type=int,help='Number of nodes in network.')
	parser.add_argument("--cascade_type",default='threshold',type=str,help='Type of cascading dynamics.')
	#parser.add_argument("--mlp_hidden_size",default=64,type=int,help='Size of hidden layers in MLP')
	#parser.add_argument("--embed_size",default=16,type=int,help='Size of hidden layers in MLP')
	parser.add_argument("--ego_model_type",default='MLP',type=str,help='Type of NN model.')
	#parser.add_argument("--mlp_hidden_depth",default=2,type=int,help='Size of hidden layers in MLP')

	args = parser.parse_args()
	device = 'cpu'
	topology_fn = args.data_dir + 'net_0.gpickle'
	nash_eqs_dir = args.data_dir + 'thresholdcasc_NashEQs/'
	import networkx as nx
	G = nx.read_gpickle(topology_fn)

	val_data_path = args.data_dir + f'{args.cascade_type}casc_trialdata.npy'
	if val_data_path is not None and os.path.isfile(val_data_path):
		val_dataset = NetCascDataset(args.data_dir,args.cascade_type)
	else: 
		val_dataset = None
	from netcasc_gym_env import NetworkCascEnv
	test_env = NetworkCascEnv(args.p,args.p,'File',cascade_type=args.cascade_type,filename=topology_fn,degree=args.p)

	state_dict = torch.load(args.q_model_path)
	if args.ego_model_type == 'MLP':
		embed_size = state_dict['act_embedding.weight'].shape[1]
		hidden_size = state_dict['mlp_layers.0.weight'].shape[0]
		output_size = state_dict['output_layer.weight'].shape[0]
		depth = int(len([key for key in state_dict if 'mlp_layers' in key])/2)
		q_model = MLP_Critic(embed_size,hidden_size,output_size,num_mlp_layers=depth)
		edge_index = None
		gnn = False
	elif args.ego_model_type == 'GCN':
		embed_size = state_dict['convs.0.lin.weight'].shape[0]
		hidden_size = state_dict['mlp_layers.0.weight'].shape[0]
		output_size = state_dict['output_layer.weight'].shape[0]
		depth = int(len([key for key in state_dict if 'mlp_layers' in key])/2)
		q_model = GCN_Critic(embed_size,hidden_size,output_size,num_mlp_layers=depth)
		from torch_geometric.utils import from_networkx
		edge_index = from_networkx(G).edge_index
		gnn = True
	V = Validator([test_env],p=args.p,dataset=val_dataset,nash_eqs_dir=nash_eqs_dir,gnn=gnn)

	#q_model = MLP_Critic(embed_size,hidden_size,args.net_size,num_node_features=args.net_size,p=args.p,num_mlp_layers=depth)
	q_model.load_state_dict(state_dict)
	q_model.to(device)
	q_model.eval()

	test_err,util_err,nash_div = V.validate(q_model)
	print(util_err)
	print(nash_div)