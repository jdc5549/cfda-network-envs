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
from marl.policy import MinimaxQCriticPolicy
from marl.model.nn.mlpnet import MultiCriticMlp
from marl.tools import gymSpace2dim

class Validator():
	def __init__(self,envs,dataset=None,nash_eqs_dir=None,device='cpu',exploiter_model_dir=None):
		self.envs = envs
		self.device = device
		self.obs_space = envs[0].observation_space
		self.act_space = envs[0].action_space
		num_nodes = gymSpace2dim(self.obs_space)[0]
		self.all_actions = get_combinatorial_actions(num_nodes,2)

		# if os.path.isdir(nash_eqs_dir):
		#Get Nash EQs
		fns = [f for f in os.listdir(nash_eqs_dir)]
		fns.sort()
		if nash_eqs_dir is not None and os.path.isdir(nash_eqs_dir):
			self.nashEQ_policies = [np.load(os.path.join(nash_eqs_dir,f)) for f in fns if f'eq_' in f]
			self.utils = [np.load(os.path.join(nash_eqs_dir,f)) for f in fns if f'util_' in f]
		else:
			self.nashEQ_policies = []
			self.utils = []

		if dataset is not None:
			self.dataset = dataset
			self.data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
		else:
			self.data_loader = None

	def validate(self,q_model,embed_model=None,device='cpu'):
		pred_err = None
		util_errs = []
		nash_eq_divergences = []
		criterion = nn.SmoothL1Loss()
		with torch.no_grad():
			#Test on Validation Trial Data
			if self.data_loader is not None:
				pred_err = 0
				for i, data in enumerate(self.data_loader):
					(feat_topo,actions), reward = data
					B = feat_topo.shape[0]
					atk_acts = actions[:,:2]
					def_acts = actions[:,2:]
					feat_topo = feat_topo.to(device)
					atk_acts = atk_acts.to(device)
					def_acts = def_acts.to(device)
					reward = reward.to(device)

					#select rows from featurized topology corresponding to nodes attacked
					feat_atk = feat_topo[torch.arange(feat_topo.size(0))[:, None], atk_acts, :]
					#flatten into 1 dimension (not including batch dim)
					feat_atk = feat_atk.view(B,-1)

					feat_def = feat_topo[torch.arange(feat_topo.size(0))[:, None], def_acts, :]
					feat_def = feat_def.view(B,-1)

					feat_topo_mean = torch.mean(feat_topo,dim=1)
					pred_reward = q_model(feat_topo_mean,feat_atk,feat_def).squeeze()
					pred_err += np.sum(np.abs((pred_reward-reward).detach().cpu().numpy()))/self.dataset.__len__()

			#Compare to ground truth utility and NashEQ if available
			if len(self.nashEQ_policies) > 0:
				for i,env in enumerate(self.envs):
					atk_policy = MinimaxQCriticPolicy(q_model,action_space=self.act_space,observation_space=self.obs_space,player=0,all_actions=self.all_actions,act_degree=2)
					def_policy = MinimaxQCriticPolicy(q_model,action_space=self.act_space,observation_space=self.obs_space,player=1,all_actions=self.all_actions,act_degree=2)
					observation = env.reset(fid=i) #torch.tensor(env.reset(fid=i)[0])
					feat_obs = get_featurized_obs([observation],embed_model=embed_model).detach().squeeze().to(device)
					feat_actions = torch.stack([feat_obs[action].flatten() for action in self.all_actions]).float().to(device)
					t_obs = torch.mean(feat_obs,axis=0)
					atk_pd,atk_Q_val = atk_policy.get_policy(t_obs,feat_actions)
					def_pd,def_Q_val = def_policy.get_policy(t_obs,feat_actions)
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
					for j, errj in enumerate(err_mat):
						for k, errk in enumerate(errj):
							for l,errl in enumerate(errk):
								abs_errl = np.abs(errl)
								err.append(abs_errl)
					util_errs.append(np.mean(err))

		util_err_ret = np.mean(util_errs) if len(util_errs) > 0 else None
		nash_div_ret = np.mean(nash_eq_divergences) if len(nash_eq_divergences) > 0 else None
		return pred_err,util_err_ret,nash_div_ret


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
	#print(test_err)
	print(util_err)
	print(nash_div)