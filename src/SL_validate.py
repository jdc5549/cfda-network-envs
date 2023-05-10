import os
import sys
import numpy as np
import torch

from scipy.stats import entropy
from utils import get_combinatorial_actions
from graph_embedding import get_featurized_obs
sys.path.append('./marl/')
from marl.policy import MinimaxQCriticPolicy
from marl.model.nn.mlpnet import MultiCriticMlp
from marl.tools import gymSpace2dim

class Validator():
	def __init__(self,envs,nash_eqs_dir=None):
		self.envs = envs
		self.obs_space = envs[0].observation_space
		self.act_space = envs[0].action_space
		num_nodes = gymSpace2dim(self.obs_space)[0]
		self.all_actions = get_combinatorial_actions(num_nodes,2)

		#Get Nash EQs
		if os.path.isdir(nash_eqs_dir):
			fns = [f for f in os.listdir(nash_eqs_dir)]
			fns.sort()
			self.nashEQ_policies = [np.load(os.path.join(nash_eqs_dir,f)) for f in fns if f'{envs[0].cascade_type}Casc_eq_' in f]
			self.utils = [np.load(os.path.join(nash_eqs_dir,f)) for f in fns if f'{envs[0].cascade_type}Casc_util_' in f]

	def validate(self,q_model,embed_model=None):
		nash_eq_divergences = []
		util_errs = []
		with torch.no_grad():
			for i,env in enumerate(self.envs):
				#policy_i = []
				atk_policy = MinimaxQCriticPolicy(q_model,action_space=self.act_space,observation_space=self.obs_space,player=0,all_actions=self.all_actions,act_degree=2)
				def_policy = MinimaxQCriticPolicy(q_model,action_space=self.act_space,observation_space=self.obs_space,player=1,all_actions=self.all_actions,act_degree=2)
				observation = env.reset(fid=i) #torch.tensor(env.reset(fid=i)[0])
				feat_obs = get_featurized_obs([observation],embed_model=embed_model).detach().squeeze()
				feat_actions = torch.stack([feat_obs[action].flatten() for action in self.all_actions]).float()
				t_obs = torch.mean(feat_obs,axis=0)

				atk_pd,atk_Q_val = atk_policy.get_policy(t_obs,feat_actions)
				def_pd,def_Q_val = def_policy.get_policy(t_obs,feat_actions)

				impl_policy = np.array([atk_pd,def_pd])
				nashEQ_policy = self.nashEQ_policies[i]
				kl = entropy(nashEQ_policy.flatten(),impl_policy.flatten())
				nash_eq_divergences.append(kl)
				err_mat = [atk_Q_val-self.utils[i],def_Q_val+self.utils[i]]
				err = []
				for j, errj in enumerate(err_mat):
					for k, errk in enumerate(errj):
						for l,errl in enumerate(errk):
							abs_errl = np.abs(errl)
							err.append(abs_errl)
				util_errs.append(np.mean(err))
			return np.mean(nash_eq_divergences),np.mean(util_errs)


if __name__ == '__main__':	
	import argparse
	parser = argparse.ArgumentParser(description='Netcasc SL Training Args')
	parser.add_argument("--data_dir",default=None,type=str,help='Directory to load data from.')
	parser.add_argument("--q_model_path",default=None,type=str,help='Path of q_model to evaluate.')
	parser.add_argument("--net_size",default=5,type=int,help='Number of nodes in network.')
	parser.add_argument("--cascade_type",default='threshold',type=str,help='Type of cascading dynamics.')
	args = parser.parse_args()

	p = 2/args.net_size
	topology_dir = args.data_dir + 'topologies/'
	nash_eqs_dir = args.data_dir + 'nash_eqs/'

	from network_gym_env import NetworkCascEnv
	test_envs = [NetworkCascEnv(args.net_size,p,p,'File',cascade_type=args.cascade_type,
				filename = os.path.join(topology_dir,f)) for f in os.listdir(topology_dir) if 'thresh' not in f]
	V = Validator(test_envs,nash_eqs_dir=nash_eqs_dir)

	embed_size = 7
	q_model = MultiCriticMlp(embed_size,embed_size*2,embed_size*2,hidden_size=64)
	q_model.load_state_dict(torch.load(args.q_model_path))
	q_model.eval()

	V.validate(q_model)