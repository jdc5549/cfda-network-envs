import networkx as nx
from networkx.algorithms.centrality import degree_centrality
from scm import SCM
import numpy as np
import random
import matplotlib.pyplot as plt
from utils import ncr
import itertools
import time

from tqdm import tqdm


class Counterfactual_Cascade_Fns():
	def __init__(self,env):
		self.env = env
		self.casc_type = self.env.scm.cascade_type
		if self.env.scm.cascade_type == 'shortPath':
			num_nodes = self.env.scm.G.number_of_nodes()
			self.deltaL = np.zeros([num_nodes,num_nodes])
			self.L = np.zeros(num_nodes)
			self.solo_fail_sets = [[] for i in range(num_nodes)]
			l0 = self.env.scm.loads
			ignore_val = 0.1
			for n in range(num_nodes):
				solo_fail_set = self.env.scm.check_cascading_failure([n])
				self.solo_fail_sets[n] = solo_fail_set
				self.L[n] = np.sum([l0[f] for f in solo_fail_set])-len(solo_fail_set)*(len(l0)-1)
				lf = self.env.scm.loads
				self.deltaL[n] = lf - l0
				self.deltaL[n,solo_fail_set] = ignore_val
				self.env.scm.reset()
			# print(l0)
			# print(self.env.scm.capacity)
			# with np.printoptions(threshold=np.inf):
			# 	print(self.deltaL)
			# self.env.scm.show_graph()
			# exit()
		elif self.env.scm.cascade_type == 'coupled':
			self.deltaDp = np.zeros([self.env.scm.G.number_of_nodes(),self.env.scm.G.number_of_nodes()])
			self.deltaDc = np.zeros([self.env.scm.comm_net.number_of_nodes(),self.env.scm.comm_net.number_of_nodes()]) 

	def check_failure_independence(self,f1,f1_inits,f2,f2_inits):
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
						continue
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
						solo_fail_set = self.solo_fail_sets[n]#elf.env.scm.check_cascading_failure([n])
						# if np.array_equal(self.deltaL[n],np.zeros_like(self.deltaL[n])):
						# 	self.L[n] = np.sum([l0[f] for f in solo_fail_set])-len(solo_fail_set)*(len(l0)-1)
						# 	lf = self.env.scm.loads
						# 	self.deltaL[n] = lf - l0
						self.env.scm.reset()
						found_fails = [v for v in solo_fail_set if v in unsure_fail_set] #extrapolation from individual fail sets?
						for f in found_fails:
							if f in f1[-1] and n == f1_init: 
								f1[f1_init].append(f)
								f1[-1].remove(f)
							if f in f2[-1] and n == f2_init: 
								f2[f2_init].append(f)
								f2[-1].remove(f)
					if any([f not in f1[f1_init]+f2[f2_init] for f in unsure_fail_set]):
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
		indep_check_count = 0
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
	
	def _process_cfac(self,args,combs):
		inits_tried = set()
		cfac_keys = {}
		trial_data,trial_info,fac_keys,all_comb_acts,max_comb_data = args
		num_nodes = max(max(all_comb_acts))+1
		if combs[0][0][0] == 0 and combs[0][1][0] == 1:
			first_chunk = True
			pbar = tqdm(total=len(combs),desc='First chunk progress')
		else:
			first_chunk = False
		# shared_cfac_trial_data,shared_cfac_trial_info,
		cfac_trial_data = []
		cfac_trial_info = []
		total_data_count = 0
		comb_data_count = 0
		for i,comb in enumerate(combs):
			total_data_count += comb_data_count
			comb_data_count = 0
			if first_chunk:
				data_limit = max_comb_data*(i+1)
				pbar.set_postfix({'data_count': total_data_count,'data_limit': data_limit})
				pbar.update()
			max_data_reached = False
			i,fcs_subset_i = comb[0]
			p,fcs_subset_p = comb[1]
			for j,fcj in enumerate(fcs_subset_i):
				if max_data_reached:
					break
				act_j = trial_data[i][j][:-1]
				for k,fck in enumerate(fcs_subset_p):
					if max_data_reached:
						break
					act_k = trial_data[p][k][:-1]
					num_inits = len(fcj.keys()) + len(fck.keys()) - 2   #sum([1 for n in cc1 if n in init_fail_new or n in self.init_fails[i]])
					if num_inits > len(act_k):
						continue
					init1 = [key for key in fcj.keys() if key != -1]
					init1_keep = [False for i1 in init1]
					init2 = [key for key in fck.keys() if key != -1]
					init2_keep = [False for i2 in init2]
					init_keys = []
					for idx1,i1 in enumerate(init1):
						for idx2,i2 in enumerate(init2):
							cfac_init_fail = [i1,i2]
							cfac_init_fail.sort()
							if cfac_init_fail == trial_info[i][j]['init_fail'] or cfac_init_fail == trial_info[p][k]['init_fail']:
								continue
							init_key = num_nodes*i1 + i2
							if init_key not in inits_tried:
								init1_keep[idx1] = True
								init2_keep[idx2] = True
								init_keys.append(init_key)
					filtered_init_j = [i1 for i1, keep in zip(init1, init1_keep) if keep]
					filtered_init_k = [i2 for i2, keep in zip(init2, init2_keep) if keep]
					#print(f'Time to filtered inits: {toc-tic}')
					if len(filtered_init_j) < 1 or len(filtered_init_k) < 1:
						continue
					else:
						indep_sets = self.check_failure_independence(fcj,filtered_init_j,fck,filtered_init_k)
						#if len(indep_sets) == 0:
						for init_key in init_keys:
							inits_tried.add(init_key)
					for idp in indep_sets:
						if max_data_reached:
							break
						cfac_atk_action = idp.copy()
						cfac_atk_action.sort()
						# cfac_init_fail = cfac_atk_action #[n for n in cfac_atk_action if n not in cfac_def_action]
						# if cfac_init_fail == trial_info[i][j]['init_fail'] or cfac_init_fail == trial_info[p][k]['init_fail']:
						# 	continue
						#all_def_actions = list(act_j[2:]) + list(act_k[2:])
						#all_def_actions = list(map(int,all_def_actions))
						#random.shuffle(all_def_actions)
						# second_gen = False
						# for d in all_def_actions:
						# 	if d not in cfac_def_action and d not in cfac_atk_action:
						# 		cfac_def_action.append(d)
						# 	if len(cfac_def_action) >= len(list(act_j[2:])):
						# 		break
						# if len(cfac_def_action) < len(list(act_j[2:])):
						# 	second_gen = True
						# 	d = np.random.choice([a for a in range(self.env.scm.G.number_of_nodes()) if a not in cfac_atk_action and a not in cfac_def_action])
						# 	cfac_def_action.append(d)
						# cfac_def_action.sort()
						#cfac_action = [tuple(cfac_atk_action),tuple(cfac_def_action)]
						#cfac_def_actions = [d for d in all_comb_acts if not any([a in d for a in cfac_atk_action])]
						cfac_def_action = random.sample(all_comb_acts,1)[0]
						while any([a in cfac_def_action for a in cfac_atk_action]):
							cfac_def_action = random.sample(all_comb_acts,1)[0]
						#cfac_actions = [[tuple(cfac_atk_action),tuple(d)] for d in cfac_def_actions]
						cfac_actions = [[tuple(cfac_atk_action),tuple(cfac_def_action)]]
						#print(f'Time to get cfac action from indep set: {toc-tic}')
						#skip any cfac actions that already exist in the factual data
						for cfac_action in cfac_actions:
							key = len(all_comb_acts)*all_comb_acts.index(cfac_action[0]) + all_comb_acts.index(cfac_action[1])
							if key in fac_keys or key in cfac_keys: 
								continue
							else:
								cfac_keys[key] = len(cfac_trial_data) #corresponding index in cfac_trial_data after it is added

							for f in idp:
								if f in fcj.keys():
									def_1 = f
								if f in fck.keys():
									def_2 = f
							counterfac_casc_fail = list(set(fcj[def_1] + fck[def_2]))

							r = len(counterfac_casc_fail)/self.env.scm.G.number_of_nodes()
							# if r == 0 and len(counterfac_casc_fail) > 0:
							trial_list = [a for act in cfac_action for a in act]
							trial_list.append(r)
							trial_result = np.asarray(trial_list)
							cfac_trial_data.append(trial_result)
							comb_data_count += 1
							info = {'init_fail':cfac_init_fail,'fail_set':counterfac_casc_fail}
							cfac_trial_info.append(info)
							if comb_data_count >= max_comb_data:
								max_data_reached = True
								break

		# shared_cfac_trial_data += local_cfac_trial_data
		# shared_cfac_trial_info += local_cfac_trial_info
		return cfac_trial_data,cfac_trial_info, cfac_keys,len(combs)
	def gen_cross_subset_cfacs(self,trial_data,trial_info,fac_keys,all_comb_acts,max_ratio=1000):
		num_subsets = trial_data.shape[0]
		num_trials = trial_data.shape[1]
		fail_components = [[] for j in range(num_subsets)]
		from multiprocessing import Pool,Lock, Manager,cpu_count
		for i,sub_info in enumerate(trial_info):
			for j,info in sub_info.items():
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
				fail_components[i].append(new_failure_component)

		from functools import partial
		manager = Manager()
		cfac_trial_data = manager.list()
		cfac_trial_info = manager.list()
		cfac_keys = manager.dict()
		# for i,fcs_subset_i in enumerate(fail_components):
		# 	for p,fcs_subset_p in enumerate(fail_components[i+1:]):
		num_combs = ncr(len(fail_components),2)
		subset_combinations = list(itertools.combinations(enumerate(fail_components), 2))
		cfac_data_cap = max((max_ratio*num_subsets*num_trials,len(subset_combinations)))
		print(f'Counterfactual data cap set to: {cfac_data_cap}')
		data_per_combo = cfac_data_cap//len(subset_combinations)
		#print(f'Cap for a single combination: {data_per_combo}')
		#num_chunks = max(cpu_count()-2, len(subset_combinations)//500)
		num_chunks = cpu_count()-2
		max_chunk_size = 1e6
		chunk_size = min([len(subset_combinations) // num_chunks,max_chunk_size])
		chunks = [subset_combinations[i:i+chunk_size] for i in range(0, len(subset_combinations), chunk_size)]
		# print(num_chunks)
		# print(f'{len(chunks)} chunks of size {chunk_size} each.')
		# exit()
		lock = Lock()
		tic = time.perf_counter()
		with Pool(processes=num_chunks) as pool, tqdm(total=num_combs, desc='CFac Data Progress') as cfac_pbar:
			args = (trial_data,trial_info,fac_keys,all_comb_acts,data_per_combo)
			for (chunk_trial_data, chunk_trial_info,chunk_cfac_keys,chunk_len) in pool.imap_unordered(partial(self._process_cfac, args), chunks):
				kept_idxs = []
				for key in chunk_cfac_keys:
					with lock:
						if key not in cfac_keys:
							cfac_keys[key] = chunk_cfac_keys[key]
							kept_idxs.append(chunk_cfac_keys[key])
				cfac_trial_data.extend([chunk_trial_data[i] for i in kept_idxs])
				cfac_trial_info.extend([chunk_trial_info[i] for i in kept_idxs])
				cfac_pbar.update(chunk_len)
		toc = time.perf_counter()
		cfac_time = toc-tic
		print('copying to regular data structures')
		cfac_trial_data = np.array(cfac_trial_data)
		cfac_trial_info = list(cfac_trial_info)
		cfac_keys = dict(cfac_keys)
		print('finished')

		# cfac_trial_data = []
		# cfac_trial_info = []
		# failed_inits = set()
		# cfac_keys = set()
		# cfac_pbar = tqdm(total=num_combs,desc='Cfac Gen Progress')
		# for comb in subset_combinations:
		# 	i,fcs_subset_i = comb[0]
		# 	p,fcs_subset_p = comb[1]
		# 	cfac_pbar.update()
		# 	for j,fcj in enumerate(fcs_subset_i):
		# 		act_j = trial_data[i][j][:-1]
		# 		for k,fck in enumerate(fcs_subset_p):
		# 			act_k = trial_data[p][k][:-1]
		# 			num_inits = len(fcj.keys()) + len(fck.keys()) - 2   #sum([1 for n in cc1 if n in init_fail_new or n in self.init_fails[i]])
		# 			if num_inits > len(act_k):
		# 				continue
		# 			init1 = list(fcj.keys())
		# 			init2 = list(fck.keys())
		# 			key1 = len(all_comb_acts)*init1[0] if len(init1) > 0 else 0
		# 			key2 = init2[0] if len(init2) > 0 else 0
		# 			init_key = key1 + key2
		# 			if init_key in failed_inits:
		# 				continue
		# 			else:
		# 				indep_sets = self.check_failure_independence(fcj,fck)
		# 				if len(indep_sets) == 0:
		# 					failed_inits.add(init_key)
		# 			for idp in indep_sets:
		# 				cfac_atk_action = idp.copy()
		# 				all_def_actions = list(act_j[2:]) + list(act_k[2:])
		# 				all_def_actions = list(map(int,all_def_actions))
		# 				random.shuffle(all_def_actions)
		# 				cfac_def_action = []
		# 				second_gen = False
		# 				for d in all_def_actions:
		# 					if d not in cfac_def_action and d not in cfac_atk_action:
		# 						cfac_def_action.append(d)
		# 					if len(cfac_def_action) >= len(list(act_j[2:])):
		# 						break
		# 				if len(cfac_def_action) < len(list(act_j[2:])):
		# 					second_gen = True
		# 					d = np.random.choice([a for a in range(self.env.scm.G.number_of_nodes()) if a not in cfac_atk_action and a not in cfac_def_action])
		# 					cfac_def_action.append(d)
		# 				cfac_atk_action.sort()
		# 				cfac_def_action.sort()
		# 				cfac_action = [tuple(cfac_atk_action),tuple(cfac_def_action)]

		# 				#skip any cfac actions that already exist in the factual data
		# 				key = len(all_comb_acts)*all_comb_acts.index(cfac_action[0]) + all_comb_acts.index(cfac_action[1])
		# 				if key in fac_keys or key in cfac_keys: 
		# 					continue
		# 				else:
		# 					cfac_keys.add(key)

		# 				cfac_init_fail = cfac_atk_action #[n for n in cfac_atk_action if n not in cfac_def_action]
		# 				for f in idp:
		# 					if f in fcj.keys():
		# 						def_1 = f
		# 					if f in fck.keys():
		# 						def_2 = f
		# 				#TODO: Account for case where init fail from one component is a cascaded fail in the other
		# 				# def_1 = cc1 if any(n in cc1 for n in cfac_init_fail) else []
		# 				# def_2 = cc2 if any(n in cc2 for n in cfac_init_fail) else []
		# 				counterfac_casc_fail = list(set(fcj[def_1] + fck[def_2]))
		# 				#fac_cfac_casc_fail = self.env.scm.check_cascading_failure(cfac_init_fail)
		# 				#self.env.scm.reset()
		# 				#Check that cfac is valid
		# 				# if set(fac_cfac_casc_fail) != set(counterfac_casc_fail):
		# 				# 	print('action1: ', act_j)
		# 				# 	print('action2: ', act_k)
		# 				# 	print('init1: ',trial_info[j]['init_fail'])
		# 				# 	print('init2: ', trial_info[k]['init_fail'])
		# 				# 	print(f'orig_failset1: ', trial_info[j]['fail_set'])
		# 				# 	print('orig_failset2: ', trial_info[k]['fail_set'])
		# 				# 	print('cfac_init: ', cfac_init_fail)
		# 				# 	print('comp1: ', fcj)
		# 				# 	print('comp2: ', fck)
		# 				# 	if self.cfa_cascade_fns.casc_type == 'shortPath':
		# 				# 		print('Factual Fail Set 1: ', self.env.scm.check_cascading_failure(trial_info[j]['init_fail']))
		# 				# 		self.env.scm.reset()
		# 				# 		print('Factual Fail Set 2: ', self.env.scm.check_cascading_failure(trial_info[k]['init_fail']))
		# 				# 		self.env.scm.reset()
		# 				# 	print('cfac_actions: ',cfac_action)
		# 				# 	print('cfac_init_fail: ', cfac_init_fail)
		# 				# 	print('def_1: ', def_1)
		# 				# 	print('def_2: ', def_2)
		# 				# 	print('counterfac_casc_fail: ', counterfac_casc_fail)
		# 				# 	print('fac_cfac_casc_fail: ',fac_cfac_casc_fail)
		# 				# 	exit()
		# 				if cfac_init_fail == trial_info[i][j]['init_fail'] or cfac_init_fail == trial_info[p][k]['init_fail']:
		# 						continue
		# 				r = len(counterfac_casc_fail)/self.env.scm.G.number_of_nodes()
		# 				# if r == 0 and len(counterfac_casc_fail) > 0:
		# 				trial_list = [a for act in cfac_action for a in act]
		# 				trial_list.append(r)
		# 				trial_result = np.asarray(trial_list)
		# 				cfac_trial_data.append(trial_result)
		# 				info = {'init_fail':cfac_init_fail,'fail_set':counterfac_casc_fail}
		# 				cfac_trial_info.append(info)
		# cfac_pbar.close()
		cfac_trial_data = np.stack(cfac_trial_data) if len(cfac_trial_data) > 0 else []
		return cfac_trial_data, cfac_trial_info,cfac_time

if __name__ == '__main__':
	import argparse
	from create_subact_dataset import create_dataset
	savedata_fn = './data/Cfac_tests/shortPath'
	num_subtargets = 5
	train_trials = 100
	#cascade_type = 'threshold'
	cascade_types = ['shortPath']#,'coupled']

	plot_params = [10,100,1000] #net_size
	subact_sets = [16*p//5 for p in plot_params]#[p*16//5 for p in plot_params]
	overwrite = [True,True]
	eval_method = ['NashEQ','valtrials_RandomCycleExpl','valtrials_RandomCycleExpl']

	cfac_data = {'threshold':np.zeros((len(plot_params),4)),'shortPath':np.zeros((len(plot_params),4))}#,'coupled':np.zeros((len(plot_params),4))}
	#cfac_data = {'shortPath':np.zeros((len(plot_params),4))}
	top_dir = './data/Cfac_tests'

	for i,param in enumerate(plot_params):
		for casc_type in cascade_types:
			print(f'Net size {param}: {casc_type} cascading.')
			args = argparse.Namespace(
				ego_graph_size=param,
				num_nodes_chosen=2,
				num_subact_targets=num_subtargets,
				num_subact_sets=subact_sets[i],
				cfda=True,
				calc_nash_ego=False,
				max_valset_trials=0,
				num_trials_sub=train_trials,
				cascade_type=casc_type,
				exploration_type='RandomCycle',
				load_graph_dir=f'{top_dir}/{param}C2/ego_/',
				top_dir=top_dir,
				overwrite=True
			)
			data = create_dataset(args)
			cfac_data[casc_type][i,:2] = data[:2]
			cfac_data[casc_type][i,2] = data[2]/data[0] if data[2] is not None and data[0] > 0 else 0
			cfac_data[casc_type][i,3] = data[3]/data[1] if data[1] > 0 else 0
			#print(cfac_data[casc_type][i])
			print(f'num cfac data: {cfac_data[casc_type][i,1]}')

	import pickle
	with open(f'{savedata_fn}.pkl','wb') as file:
		pickle.dump(cfac_data,file)

	# np.save(savedata_fn,cfac_data)
	# for data in cfac_data:
	# 	print(data)

	import matplotlib.pyplot as plt
	plt.loglog(plot_params,cfac_data['threshold'][:,1])
	plt.loglog(plot_params,cfac_data['shortPath'][:,1])
	#plt.plot(plot_params,casc_data['coupled'][1,:])

	# Set labels and title
	plt.xlabel('Number of Nodes in Network')
	plt.ylabel('Number of Counterfactual Data')
	plt.title('Counterfactual Data and Network Size')
	plt.legend(['Threshold','Shortest Path'])
	# Show the plot
	plt.show()


	# means_Random = [np.mean(data) for data in cfac_data['Random']]
	# stds_Random = [np.std(data) for data in cfac_data['Random']]
	# confidence_intervals_Random = [(mean - 1.96 * (std / np.sqrt(len(data))),
	#                          mean + 1.96 * (std / np.sqrt(len(data))))
	#                         for mean, std, data in zip(means_Random, stds_Random, cfac_data['Random'])]

	# # Plot the mean with shaded confidence interval
	# plt.errorbar(trial_counts, means_Random, yerr=np.transpose(confidence_intervals_Random), fmt='o', capsize=5)

	# means_RandomCycle = [np.mean(data) for data in cfac_data['RandomCycle']]
	# stds_RandomCycle = [np.std(data) for data in cfac_data['RandomCycle']]
	# confidence_intervals_RandomCycle = [(mean - 1.96 * (std / np.sqrt(len(data))),
	#                          mean + 1.96 * (std / np.sqrt(len(data))))
	#                         for mean, std, data in zip(means_RandomCycle, stds_RandomCycle, cfac_data['RandomCycle'])]

	# # Plot the mean with shaded confidence interval
	# plt.errorbar(trial_counts, means_Random, yerr=np.transpose(confidence_intervals_Random), fmt='o', capsize=5)

	# means_CDME = [np.mean(data) for data in cfac_data['CDME']]
	# stds_CDME = [np.std(data) for data in cfac_data['CDME']]
	# confidence_intervals_CDME = [(mean - 1.96 * (std / np.sqrt(len(data))),
	#                          mean + 1.96 * (std / np.sqrt(len(data))))
	#                         for mean, std, data in zip(means_CDME, stds_CDME, cfac_data['CDME'])]

	# # Plot the mean with shaded confidence interval
	# plt.errorbar(trial_counts, means_CDME, yerr=np.transpose(confidence_intervals_CDME), fmt='o', capsize=5)

	# plt.legend(['Random','RandomCycle','CDME'])

	# # Set labels and title
	# plt.xlabel('Trial Counts')
	# plt.ylabel('Mean')
	# plt.title('Mean with 95% Confidence Interval')

	# # Show the plot
	# plt.show()





