import networkx as nx
from networkx.algorithms.centrality import degree_centrality
from scm import SCM
import numpy as np
import random
import matplotlib.pyplot as plt

class Counterfactual_Cascade_Fns():
	def __init__(self,env):
		self.env = env
		self.casc_type = self.env.scm.cascade_type
		if self.env.scm.cascade_type == 'shortPath':
			self.deltaL = np.zeros([self.env.scm.G.number_of_nodes(),self.env.scm.G.number_of_nodes()])
			self.L = np.zeros(self.deltaL.shape[0])
		elif self.env.scm.cascade_type == 'coupled':
			self.deltaDp = np.zeros([self.env.scm.G.number_of_nodes(),self.env.scm.G.number_of_nodes()])
			self.deltaDc = np.zeros([self.env.scm.comm_net.number_of_nodes(),self.env.scm.comm_net.number_of_nodes()]) 

	def check_failure_independence(self,f1,f2):
		f1_inits = list(f1.keys())
		f2_inits = list(f2.keys())
		indep_sets = []
		if self.casc_type == 'threshold':
			for i,k1 in enumerate(f1_inits):
				if k1 in f2_inits: continue
				for j,k2 in enumerate(f2_inits):
					if k2 in f1_inits: continue
					f1_init = f1_inits[i]
					f2_init = f2_inits[j]
					f1_fail_set = f1[f1_init]
					f2_fail_set = f2[f2_init]
					#if one cascade is subset of the other no info can be gained from combining them
					if set(f1_fail_set).issubset(set(f2_fail_set)) or set(f2_fail_set).issubset(set(f1_fail_set)):
						return []
					#check independence through neighbors
					thresh_copy = self.env.scm.thresholds.copy()
					all_fail_nodes = list(set(f1_fail_set) | set(f2_fail_set)) #merge lists without repeating elements 
					casc_thresh = False
					for node in all_fail_nodes:
						for n in self.env.scm.G[node]: 
							if n not in all_fail_nodes: 
								thresh_copy[n] -= 1/len(self.env.scm.G[n])    #decrement thresh of neighbor nodes
								if thresh_copy[n] <= 0:
									casc_thresh = True
					if not casc_thresh:
						indep_sets.append([k1,k2])
		elif self.casc_type == 'shortPath':
			f1_inits.remove(-1)
			f2_inits.remove(-1)
			#print(f1_inits)
			#print(f2_inits)
			for i,k1 in enumerate(f1_inits):
				if k1 in f2_inits: continue
				for j,k2 in enumerate(f2_inits):
					if k2 in f1_inits: continue
					f1_init = f1_inits[i]
					f2_init = f2_inits[j]
					f1_fail_set = f1[f1_init]
					f2_fail_set = f2[f2_init]
					unsure_fail_set = f1[-1] + f2[-1]
					l0 = self.env.scm.loads
					for n in [f1_init,f2_init]:
						solo_fail_set = self.env.scm.check_cascading_failure([n])
						if np.array_equal(self.deltaL[n],np.zeros_like(self.deltaL[n])):
							self.L[n] = np.sum([l0[f] for f in solo_fail_set])-len(solo_fail_set)*(len(l0)-1)
							lf = self.env.scm.loads
							self.deltaL[n] = lf - l0
						self.env.scm.reset()
						found_fails = [v for v in solo_fail_set if v in unsure_fail_set]
						for f in found_fails:
							if f in f1[-1] and n == f1_init: 
								f1[f1_init].append(f)
								f1[-1].remove(f)
							if f in f2[-1] and n == f2_init: 
								f2[f2_init].append(f)
								f2[-1].remove(f)
					if set(f2_fail_set).issubset(set(f1_fail_set)) or set(f1_fail_set).issubset(set(f2_fail_set)):
						#print(f'One of these should be a subset of the other: {f1_fail_set},{f2_fail_set}')
						continue
					if any([f not in f1[f1_init]+f2[f2_init] for f in unsure_fail_set]):
						#print(f'element in {unsure_fail_set} not found in {f1[f1_init]+f2[f2_init]}')
						#print(f'f1: {f1}, f2: {f2}')
						continue

					capacity = self.env.scm.capacity
					indep = True
					for n in self.env.scm.G.nodes():
						if n not in f1_fail_set or f2_fail_set: 
							if capacity[n] < l0[n] + self.deltaL[f1_init,n] + self.L[f2_init] or capacity[n] < l0[n] + self.deltaL[f2_init,n] + self.L[f1_init]:
								indep = False
					if indep:
						indep_sets.append([k1,k2])
		elif self.casc_type == 'coupled':
			f1_inits.remove(-1)
			f2_inits.remove(-1)
			for i,k1 in enumerate(f1_inits):
				if k1 in f2_inits: continue
				for j,k2 in enumerate(f2_inits):
					if k2 in f1_inits: continue
					f1_init = f1_inits[i]
					f2_init = f2_inits[j]
					f1_fail_set = f1[f1_init]
					f2_fail_set = f2[f2_init]
					unsure_fail_set = f1[-1] + f2[-1]
				dp0 = self.env.scm.dp
				dc0 = self.env.scm.dc
				for n in [f1_init,f2_init]:
					solo_fail_set = self.env.scm.check_cascading_failure([n])	
					dpf = self.env.scm.dp
					dcf = self.env.scm.dc
					if np.array_equal(self.deltaDp[n],np.zeros_like(self.deltaDp[n])):
						self.deltaDp[n] = dpf - dp0
						self.deltaDc[n] = dcf - dc0
					self.env.scm.reset()
					found_fails = [v for v in solo_fail_set if v in unsure_fail_set]
					for f in found_fails:
						if f in f1[-1] and n == f1_init: 
							f1[f1_init].append(f)
							f1[-1].remove(f)
						if f in f2[-1] and n == f2_init: 
							f2[f2_init].append(f)
							f2[-1].remove(f)
				if set(f2_fail_set).issubset(set(f1_fail_set)) or set(f1_fail_set).issubset(set(f2_fail_set)):
					#print(f'One of these should be a subset of the other: {f1_fail_set},{f2_fail_set}')
					continue
				if any([f not in f1[f1_init]+f2[f2_init] for f in unsure_fail_set]):
					#print(f'element in {unsure_fail_set} not found in {f1[f1_init]+f2[f2_init]}')
					#print(f'f1: {f1}, f2: {f2}')
					continue
				indep = True
				for n in self.env.scm.G.nodes():
					if n not in f1_fail_set or f2_fail_set: 
						if  dp0 + self.deltaDp[f1_init,n] + self.deltaDp[f2_init,n] < 0 or dc0 + self.detlaDc[f1_init,n] + self.deltaDc[f2_init,n] < 0:
							indep = False						
				if indep:
					indep_sets.append([k1,k2])				
		else: 
			print(f'Error: Cascade Type {self.casc_type} is not recognized')
			exit()
		for idp in indep_sets:
			if len(idp) > 2:
				print(idp)
				exit()
			if len(set(idp)) > len(idp):
				print(idp)
				exit()
		return indep_sets


	def gen_cfacs(self,trial_data,trial_info):
		fail_components = []
		cfac_trial_data = []
		cfac_trial_info = []
		for j,info in enumerate(trial_info):
			fail_set = info['fail_set']
			init_fail = info['init_fail']
			if self.casc_type == 'threshold':
				sub = self.env.scm.G.subgraph(fail_set)
				new_failure_component = {}
				for c in nx.connected_components(sub):
					c_init = [n for n in c if n in init_fail]
					if len(c_init) > 1:
						continue
					new_failure_component[c_init[0]] = list(c)
					# for i in c_init:
					# 	fail_components[i] = list(c)
			else:
				new_failure_component = {-1: []}
				for f in fail_set:
					if f in init_fail: 
						new_failure_component[f] = [f]
					else:
						new_failure_component[-1].append(f)
			fail_components.append(new_failure_component)
		for j,fcj in enumerate(fail_components[:-1]):
			act_j = trial_data[j][:-1]
			for k,fck in enumerate(fail_components[j+1:]):
				act_k = trial_data[k][:-1]
				num_inits = len(fcj.keys()) + len(fck.keys()) - 2   #sum([1 for n in cc1 if n in init_fail_new or n in self.init_fails[i]])
				if num_inits > len(act_k):
					continue
				indep_sets = self.check_failure_independence(fcj,fck)
				for idp in indep_sets:
					cfac_atk_action = idp.copy()
					all_def_actions = list(act_j[2:]) + list(act_k[2:])
					random.shuffle(all_def_actions)
					cfac_def_action = []
					for d in all_def_actions:
						if d not in cfac_def_action and d not in cfac_atk_action:
							cfac_def_action.append(d)
						if len(cfac_def_action) >= len(list(act_j[2:])):
							break
					if len(cfac_def_action) < len(list(act_j[2:])):
						d = np.random.choice([a for a in range(self.env.scm.G.number_of_nodes()) if a not in cfac_atk_action])
						cfac_def_action.append(d)
					cfac_action = [cfac_atk_action,cfac_def_action]

					#skip any cfac actions that already exist in the factual data
					for trial in trial_data:
						if trial[:4] == np.array(cfac_action): continue

					cfac_init_fail = cfac_atk_action #[n for n in cfac_atk_action if n not in cfac_def_action]
					for f in idp:
						if f in fcj.keys():
							def_1 = f
						if f in fck.keys():
							def_2 = f
					#TODO: Account for case where init fail from one component is a cascaded fail in the other
					# def_1 = cc1 if any(n in cc1 for n in cfac_init_fail) else []
					# def_2 = cc2 if any(n in cc2 for n in cfac_init_fail) else []
					counterfac_casc_fail = fcj[def_1] + fck[def_2]
					#counterfac_casc_fail = new_failure_component[cfac_init_fail] + 
					fac_cfac_casc_fail = self.env.scm.check_cascading_failure(cfac_init_fail)
					self.env.scm.reset()
					#self.env.scm.reset()
					#Check that cfac is valid
					if set(fac_cfac_casc_fail) != set(counterfac_casc_fail):
						print('action1: ', act_j)
						print('action2: ', act_k)
						print('init1: ',trial_info[j]['init_fail'])
						print('init2: ', trial_info[k]['init_fail'])
						print(f'orig_failset1: ', trial_info[j]['fail_set'])
						print('orig_failset2: ', trial_info[k]['fail_set'])
						print('cfac_init: ', cfac_init_fail)
						print('comp1: ', fcj)
						print('comp2: ', fck)
						if self.cfa_cascade_fns.casc_type == 'shortPath':
							print('Factual Fail Set 1: ', self.env.scm.check_cascading_failure(trial_info[j]['init_fail']))
							self.env.scm.reset()
							print('Factual Fail Set 2: ', self.env.scm.check_cascading_failure(trial_info[k]['init_fail']))
							self.env.scm.reset()
						print('cfac_actions: ',cfac_action)
						print('cfac_init_fail: ', cfac_init_fail)
						print('def_1: ', def_1)
						print('def_2: ', def_2)
						print('counterfac_casc_fail: ', counterfac_casc_fail)
						print('fac_cfac_casc_fail: ',fac_cfac_casc_fail)
						exit()
					if cfac_init_fail == trial_info[j]['init_fail'] or cfac_init_fail == trial_info[k]['init_fail']:
							continue
					r = len(counterfac_casc_fail)/self.env.scm.G.number_of_nodes()
					if r == 0 and len(counterfac_casc_fail) > 0:
						print('Reward: ',r)
						print('Num node: ',self.env.scm.G.number_of_nodes())
						print('Old init: ', init_fails[i])
						print('New init: ',init_fail_new)
						print('CC1: ', cc1)
						print('CC2: ',cc2)
						print("Cfac Casc: ", counterfac_casc_fail)
						print('Cfac Init:',fcac_init_fail)
						exit()		
					trial_result = cfac_action + [r]
					cfac_trial_data.append(trial_result)
					info = {'init_fail':cfac_init_fail,'fail_set':counterfac_casc_fail}
					cfac_trial_info.append(info)

		return cfac_trial_data, cfac_trial_info

if __name__ == '__main__':
	import argparse
	from create_SL_data_set import create_dataset
	net_size = 25
	exploration_types = ['CDME','Random','RandomCycle']
	trial_counts = [10,100,1000]
	load_dir = 'data/25C2/validation_data/10topo_0trials_RandomExpl/'

	cfac_data = {}
	for expl in exploration_types:
		cfac_data[expl] = []
		print(f'Exploration type {expl}')
		for count in trial_counts:
			print(f'{count} Trials')
			args = argparse.Namespace(
				train=True,
				cfda=True,
				calc_nash=False,
				net_size=net_size,
				num_nodes_chosen=2,
				num_topologies=10,
				num_trials=count,
				cascade_type='threshold',
				load_dir=load_dir,
				exploration_type=expl,
				epsilon=0.99,
				overwrite=True
			)
			cfac_counts = create_dataset(args)
			cfac_data[expl].append(cfac_counts)

	import pickle
	with open(load_dir + f'Cfac_test_data.pkl','wb') as file:
		pickle.dump(cfac_data,file)  

	import matplotlib.pyplot as plt

	means_Random = [np.mean(data) for data in cfac_data['Random']]
	stds_Random = [np.std(data) for data in cfac_data['Random']]
	confidence_intervals_Random = [(mean - 1.96 * (std / np.sqrt(len(data))),
	                         mean + 1.96 * (std / np.sqrt(len(data))))
	                        for mean, std, data in zip(means_Random, stds_Random, cfac_data['Random'])]

	# Plot the mean with shaded confidence interval
	plt.errorbar(trial_counts, means_Random, yerr=np.transpose(confidence_intervals_Random), fmt='o', capsize=5)

	means_RandomCycle = [np.mean(data) for data in cfac_data['RandomCycle']]
	stds_RandomCycle = [np.std(data) for data in cfac_data['RandomCycle']]
	confidence_intervals_RandomCycle = [(mean - 1.96 * (std / np.sqrt(len(data))),
	                         mean + 1.96 * (std / np.sqrt(len(data))))
	                        for mean, std, data in zip(means_RandomCycle, stds_RandomCycle, cfac_data['RandomCycle'])]

	# Plot the mean with shaded confidence interval
	plt.errorbar(trial_counts, means_Random, yerr=np.transpose(confidence_intervals_Random), fmt='o', capsize=5)

	means_CDME = [np.mean(data) for data in cfac_data['CDME']]
	stds_CDME = [np.std(data) for data in cfac_data['CDME']]
	confidence_intervals_CDME = [(mean - 1.96 * (std / np.sqrt(len(data))),
	                         mean + 1.96 * (std / np.sqrt(len(data))))
	                        for mean, std, data in zip(means_CDME, stds_CDME, cfac_data['CDME'])]

	# Plot the mean with shaded confidence interval
	plt.errorbar(trial_counts, means_CDME, yerr=np.transpose(confidence_intervals_CDME), fmt='o', capsize=5)

	plt.legend(['Random','RandomCycle','CDME'])

	# Set labels and title
	plt.xlabel('Trial Counts')
	plt.ylabel('Mean')
	plt.title('Mean with 95% Confidence Interval')

	# Show the plot
	plt.show()





