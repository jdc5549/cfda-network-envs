import os
import sys
import numpy as np
import torch
import torch.nn as nn

from scipy.stats import entropy
from torch.utils.data import DataLoader

from utils import get_combinatorial_actions
from graph_embedding import get_featurized_obs
from NetCascDataset import NetCascDataset
sys.path.append('./marl/')
from marl.policy import MinimaxQCriticPolicy,SubactMinimaxQCriticPolicy
from marl.model.nn.mlpnet import MultiCriticMlp
from marl.tools import gymSpace2dim

class Validator():
	def __init__(self,envs,embedding=None,subact_sets=None,dataset=None,nash_eqs_dir=None,device='cpu',exploiter_model_dir=None,gnn=False):
		self.envs = envs
		self.gnn = gnn
		#self.embedding = embedding
		self.device = device
		self.obs_space = envs[0].observation_space
		self.act_space = envs[0].action_space
		num_nodes = gymSpace2dim(self.obs_space)[0]
		self.all_actions = get_combinatorial_actions(num_nodes,2)
		#self.train_set = self.get_validation_set(subact_sets)
		# if os.path.isdir(nash_eqs_dir):
		#Get Nash EQs

		if nash_eqs_dir is not None and os.path.isdir(nash_eqs_dir):
			fns = [f for f in os.listdir(nash_eqs_dir)]
			fns.sort()
			self.nashEQ_policies = [np.load(os.path.join(nash_eqs_dir,f)) for f in fns if f'eq' in f]
			self.utils = [np.load(os.path.join(nash_eqs_dir,f)) for f in fns if f'util' in f]
		else:
			self.nashEQ_policies = []
			self.utils = []

		if dataset is not None:
			self.dataset = dataset
			self.data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
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
							edge_index.to(device)
						else:
							(node_features,actions), (reward,multi_hot_failures) = data
							edge_index=None
						node_features.to(device)
						actions.to(device)
						B = reward.shape[0]
						reward = reward.to(device)
						multi_hot_failures = multi_hot_failures.to(device)

						if self.gnn:
							pred = q_model(actions,node_features,edge_index)
						else:
							pred = q_model(actions,node_features)
						multi_hot_pred = torch.zeros_like(pred)
						multi_hot_pred[pred > 0.5] = 1
						pred_reward = torch.mean(multi_hot_pred,dim=1)
						val_err += np.sum(np.abs((pred_reward-reward).detach().cpu().numpy()))/self.dataset.__len__()
				#Compare to ground truth utility and NashEQ if available
				if len(self.nashEQ_policies) > 0:
					for i,env in enumerate(self.envs):
						atk_policy = SubactMinimaxQCriticPolicy(q_model,action_space=self.act_space,player=0,all_actions=self.all_actions)
						def_policy = SubactMinimaxQCriticPolicy(q_model,action_space=self.act_space,player=1,all_actions=self.all_actions)
						atk_pd,atk_Q_val = atk_policy.get_policy(node_features[0],edge_index)
						def_pd,def_Q_val = def_policy.get_policy(node_features[0],edge_index)
						# else:
						# 	#observation = env.reset(fid=i) #torch.tensor(env.reset(fid=i)[0])
						# 	#feat_obs = get_featurized_obs([observation],embed_model=embed_model).detach().squeeze().to(device)
						# 	feat_actions = self.embedding.embed_action(torch.tensor([action for action in self.all_actions])).float().to(device)
						# 	#t_obs = torch.mean(feat_obs,axis=0)
						# 	t_obs = feat_topo.unsqueeze(0).unsqueeze(0).repeat(len(self.all_actions),len(self.all_actions),1)
						# 	atk_pd,atk_Q_val = atk_policy.get_policy(t_obs,feat_actions)
						# 	def_pd,def_Q_val = def_policy.get_policy(t_obs,feat_actions)
						impl_policy = np.array([atk_pd,def_pd])
						best_kl = np.inf
						#for j,pol in enumerate(self.nashEQ_policies[i]):
						nashEQ_policy = self.nashEQ_policies[i]
						reg = np.ones_like(nashEQ_policy.flatten())*1e-6
						#nash_err = np.abs(nashEQ_policy-impl_policy)
						kl = entropy(impl_policy.flatten()+reg,nashEQ_policy.flatten()+reg)
						#kl = np.linalg.norm(nashEQ_policy.flatten()-impl_policy.flatten())
						# if kl < best_kl: 
						# 	best_kl = kl
						nash_eq_divergences.append(kl)
						err_mat = [atk_Q_val-self.utils[i],def_Q_val+self.utils[i]]
						err = []
						#val_err = []
						#debug_high_err = {}
						for j, errj in enumerate(err_mat):
							for k, errk in enumerate(errj):
								for l,errl in enumerate(errk):
									abs_errl = np.abs(errl)
									err.append(abs_errl)
									# if (k,l) not in self.train_set:
									# 	val_err.append(abs_errl)
									#if abs_errl > 1 and j == 0:
									#	debug_high_err[f'{[self.all_actions[k],self.all_actions[l]]}'] = [atk_Q_val[k,l],self.utils[i][k,l]]s
						util_errs.append(np.mean(err))
						#val_errs.append(val_err)
		util_err_ret = np.mean(util_errs) if len(util_errs) > 0 else None
		nash_div_ret = np.mean(nash_eq_divergences) if len(nash_eq_divergences) > 0 else None
		return val_err,util_err_ret,nash_div_ret


if __name__ == '__main__':	
	import argparse
	parser = argparse.ArgumentParser(description='Netcasc SL Training Args')
	parser.add_argument("--data_dir",default=None,type=str,help='Directory to load data from.')
	parser.add_argument("--q_model_path",default=None,type=str,help='Path of q_model to evaluate.')
	parser.add_argument("--net_size",default=5,type=int,help='Number of nodes in network.')
	parser.add_argument("--cascade_type",default='threshold',type=str,help='Type of cascading dynamics.')
	parser.add_argument("--mlp_hidden_size",default=64,type=int,help='Size of hidden layers in MLP')
	args = parser.parse_args()

	p = 2/args.net_size
	topology_dir = args.data_dir + 'topologies/'
	nash_eqs_dir = args.data_dir + 'thresholdcasc_NashEQs/'

	val_data_path = args.data_dir + f'{args.cascade_type}casc_trialdata.npy'
	if val_data_path is not None and os.path.isfile(val_data_path):
		val_dataset = NetCascDataset(args.data_dir,args.cascade_type)
	else: 
		val_dataset = None
	from network_gym_env import NetworkCascEnv
	test_envs = [NetworkCascEnv(args.net_size,p,p,'File',cascade_type=args.cascade_type,
				filename = os.path.join(topology_dir,f)) for f in os.listdir(topology_dir) if 'thresh' not in f]
	V = Validator(test_envs,dataset=val_dataset,nash_eqs_dir=nash_eqs_dir)

	embed_size = 7
	q_model = MultiCriticMlp(embed_size,embed_size*2,embed_size*2,hidden_size=args.mlp_hidden_size)
	q_model.load_state_dict(torch.load(args.q_model_path))
	q_model.eval()

	test_err,util_err,nash_div = V.validate(q_model)
	print(util_err)
	print(nash_div)