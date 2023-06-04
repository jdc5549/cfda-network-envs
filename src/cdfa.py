import random

import NetCascDataset
import create_SL_data_set

def gen_cfacs(env,trial_data,trial_info,cascade_type='threshold'):
	fail_component_data = []
	for j,data in enumerate(trial_data):
		fail_set = trial_info[j]['fail_set']
		init_fail_new = trial_info[j]['init_fail']
		if cascade_type == 'threshold':
			sub = env.scm.G.subgraph(fail_set)
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

		for k,fcd in enumerate(fail_component_data):
			num_inits = len(new_failure_component.keys()) + len(fcd.keys()) - 2   #sum([1 for n in cc1 if n in init_fail_new or n in self.init_fails[i]])
			if num_inits > len(experience[1][0])+len(experience[1][1]):
				continue
			indep_sets = self.cfa_cascade_fns.check_failure_independence(new_failure_component,fcd)
			for idp in indep_sets:
				cfac_atk_action = idp.copy()
				# fac_atk_actions = [n for n in experience[1][0] if (n in idp and n not in experience[1][1])] + [n for n in list(a[0]) if (n in cc2 and n not in list(a[1]))]
				# #if len(fac_atk_actions) < len(experience[1][0]):
				# #	continue
				# random.shuffle(fac_atk_actions)
				# cfac_atk_action = []
				# for atk in fac_atk_actions:
				# 	if atk not in cfac_atk_action:
				# 		cfac_atk_action.append(atk)
				# 	if len(cfac_atk_action) >= len(list(a[0])):
				# 		break
				all_def_actions = experience[1][1] + list(a[1])
				random.shuffle(all_def_actions)
				cfac_def_action = []
				for d in all_def_actions:
					if d not in cfac_def_action and d not in cfac_atk_action:
						cfac_def_action.append(d)
					if len(cfac_def_action) >= len(list(a[1])):
						break
				if len(cfac_def_action) < len(list(a[1])):
					d = np.random.choice([a for a in range(self.env.scm. G.number_of_nodes()) if a not in cfac_atk_action])
					cfac_def_action.append(d)
				cfac_action = [cfac_atk_action,cfac_def_action]
				# if cfac_action in fac_actions:
				# 	continue
				cfac_init_fail = cfac_atk_action #[n for n in cfac_atk_action if n not in cfac_def_action]
				for f in idp:
					if f in new_failure_component.keys():
						def_1 = f
					if f in fcd.keys():
						def_2 = f
				#TODO: Account for case where init fail from one component is a cascaded fail in the other
				# def_1 = cc1 if any(n in cc1 for n in cfac_init_fail) else []
				# def_2 = cc2 if any(n in cc2 for n in cfac_init_fail) else []
				counterfac_casc_fail = new_failure_component[def_1] + fcd[def_2]
				#counterfac_casc_fail = new_failure_component[cfac_init_fail] + 
				fac_cfac_casc_fail = env.scm.check_cascading_failure(cfac_init_fail)
				env.scm.reset()
				#self.env.scm.reset()
				#Check that cfac is valid
				if set(fac_cfac_casc_fail) != set(counterfac_casc_fail):
					print('action1: ', experience[1])
					print('action2: ', [list(a[0]),list(a[1])])
					print('init1: ',init_fail_new)
					print('init2: ', trial_info[k]['init_fail'])
					print(f'orig_failset1: ', experience[-1]['fail_set'])
					print('orig_failset2: ', fac_info[i]['fail_set'])
					print('cfac_init: ', cfac_init_fail)
					print('comp1: ', new_failure_component)
					print('comp2: ', fcd)
					if self.cfa_cascade_fns.casc_type == 'shortPath':
						print('Factual Fail Set 1: ', env.scm.check_cascading_failure(init_fail_new))
						env.scm.reset()
						print('Factual Fail Set 2: ', env.scm.check_cascading_failure(trial_info[k]['init_fail']))
						env.scm.reset()
					print('cfac_actions: ',cfac_action)
					print('cfac_init_fail: ', cfac_init_fail)
					print('def_1: ', def_1)
					print('def_2: ', def_2)
					print('counterfac_casc_fail: ', counterfac_casc_fail)
					print('fac_cfac_casc_fail: ',fac_cfac_casc_fail)
					exit()
				if cfac_init_fail == init_fail_new or cfac_init_fail == trial_info[k]['init_fail']:
						continue
				r = len(counterfac_casc_fail)/env.scm.G.number_of_nodes()
				reward = [r,-r]
				if r == 0 and len(counterfac_casc_fail) > 0:
					print('Reward: ',r)
					print('Num node: ',env.scm.G.number_of_nodes())
					print('Old init: ', init_fails[i])
					print('New init: ',init_fail_new)
					print('CC1: ', cc1)
					print('CC2: ',cc2)
					print("Cfac Casc: ", counterfac_casc_fail)
					print('Cfac Init:',fcac_init_fail)
					exit()		
				done = [True,True]
				info = {'init_fail':cfac_init_fail,'fail_set':counterfac_casc_fail}
				#print('cfac def action: ', cfac_def_action)
				# if len(cfac_init_fail) > 2:
				# 	#print('num_inits:', num_inits)
				# 	print('cc1:',cc1)
				# 	print('cc2:',cc2)
				# 	print('init fail new: ', init_fail_new)
				# 	print('init fail old: ', self.init_fails[i])
				# 	print('cfac action:',cfac_action)
				# # 	exit()
				cfac_count += 1

if __name__ == '__main__':
	dataset_sizes = [10,100,1000,10000,10000]
