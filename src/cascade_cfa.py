import networkx as nx
from networkx.algorithms.centrality import degree_centrality
from scm import SCM
import numpy as np
import matplotlib.pyplot as plt

class Counterfactual_Cascade_Fns():
	def __init__(self,env):
		self.env = env
		if self.env.scm.cascade_type == 'shortPath':
			self.deltaL = np.zeros([self.env.scm.G.number_of_nodes(),self.env.scm.G.number_of_nodes()])
			self.L = np.zeros(self.deltaL.shape[0])
		elif self.env.scm.cascade_type == 'coupled':
			self.deltaDp = np.zeros([self.env.scm.G.number_of_nodes(),self.env.scm.G.number_of_nodes()])
			self.deltaDc = np.zeros([self.env.scm.comm_net.number_of_nodes(),self.env.scm.comm_net.number_of_nodes()]) 
		self.casc_type = self.env.scm.cascade_type

	def generate_data(self,num2gen,fail_size):
		initial_failures = []
		failure_sets = []
		for i in range(num2gen):
			initial_failure = sorted(list(np.random.choice([j for j in range(0,self.env.scm.G.number_of_nodes())],size=fail_size)))
			while len(initial_failure) > len(set(initial_failure)):
				initial_failure = sorted(list(np.random.choice([j for j in range(0,self.env.scm.G.number_of_nodes())],size=fail_size)))
			initial_failures.append(initial_failure)
			failure_set = self.env.scm.check_cascading_failure(initial_failures=initial_failure)
			failure_sets.append(failure_set)
			self.env.scm.reset()
		return initial_failures, failure_sets

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
						return False
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
						if  dp0 + self.detlaDp[f1_init,n] + self.deltaDp[f2_init,n] < 0 or dc0 + self.detlaDc[f1_init,n] + self.deltaDc[f2_init,n] < 0:
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


if __name__ == '__main__':
	from src.netcasc_gym_env import NetworkCascEnv
	from src.utils import create_random_nets
	num_nodes = 10
	env = NetworkCascEnv(num_nodes,0.1,0.1,'SF',degree=1,cascade_type='coupled')
	ccf = Counterfactual_Cascade_Fns(env)

	casc2gen = 1
	initial_failures, failure_sets = ccf.generate_data(casc2gen,2)
	counterfac_init_fails = []
	counterfac_casc_fails = []
	fac_cfac_casc_fails = []
