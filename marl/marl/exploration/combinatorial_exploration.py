import random
from marl.exploration.expl_process import ExplorationProcess

class CombinatorialExplProcess(ExplorationProcess):
    def __init__(self,player, n_items, all_combinations,lam=0.99):
        self.lam = lam
        self.all_actions = [[c1,c2] for c1 in all_combinations for c2 in all_combinations]
        #self.comb_act_mapping = {i: [] for i in range(n_items)} #lookup table for which combinatorial actions contain which of the n items
        # for i,act in enumerate(all_comb_actions):
        #     for a in act:
        #         self.comb_act_mapping[a].append(i)
        self.player = player
        self.n_items = n_items
        self.count_atk = [0 for _ in range(self.n_items)]
        self.count_def = [0 for _ in range(self.n_items)]
        self.action_count = [0 for _ in range(len(self.all_actions))]
        self.action_score = [0 for _ in range(len(self.all_actions))]


    def reset(self, t=None):
        """ Reinitialize the state of the process """
        self.count_atk = [0 for _ in range(self.n_items)]
        self.count_def = [0 for _ in range(self.n_items)]
        self.action_count = [0 for _ in range(len(self.all_actions))]
        self.action_score = [0 for _ in range(len(self.all_actions))]

    def topo_reset(self, t=None):
        self.reset(t=t)
        
    def update(self, t):
        for i,act in enumerate(self.all_actions):
            self.action_score[i] = self.lam*self.action_count[i] + (1-self.lam)*(sum([self.count_atk[a] for a in act[0]])+sum([self.count_def[a] for a in act[1]]))
        return self.lam

    def __call__(self, policy, observation):
        # sort_index = np.argsort(self.comb_count)
        # action = self.all_comb_actions[sort_index[0]]
        # self.comb_count[sort_index[0]] += 1
        min_value = min(self.action_score)
        min_indices = [i for i,x in enumerate(self.action_score) if x == min_value]
        min_index = min_indices[0]
        action = self.all_actions[min_index]
        self.action_count[min_index] += 1
        for a in action[0]:
            self.count_atk[a] += 1
        for a in action[1]:
            self.count_def[a] += 1
        return action[self.player]

if __name__ == '__main__':
    import sys
    import numpy as np
    sys.path.append('/home/james/Documents/cfda-network-envs/marl/marl')
    from tools import get_combinatorial_actions
    n_items = 5
    all_comb_actions = get_combinatorial_actions(n_items,2)
    cb_expl = CombinatorialExplProcess(0,n_items,all_comb_actions)
    n_iter_test = 50
    for t in range(n_iter_test):
        action = cb_expl(0,0)
        cb_expl.update(t)
        print('--------------------')
        print('Iteration: ',t)
        print('Attacker Count: ', cb_expl.count_atk)
        print('Defender Count: ', cb_expl.count_def)
        print('Action Count: ',cb_expl.action_count)
        print('Action Score: ', np.round(cb_expl.action_score,2))