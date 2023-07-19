import math
import torch
import random
from . import ExplorationProcess

class UCB1(ExplorationProcess):
    def __init__(self, n_actions,c=1.7,device='cpu'):
        self.n_actions = n_actions
        self.count = torch.tensor([1 for _ in range(n_actions)]).to(device)
        self.t = sum(self.count)
        self.c = c

    def reset(self, t=None):
        """ Reinitialize the state of the process """
        self.count = torch.tensor([1 for _ in range(n_actions)]).to(device)
        self.t = sum(self.count)
        
    def update(self, t):
        self.t = sum(self.count)
            
    def __call__(self, policy):
        q_value = policy.model()
        expl_term = self.c * torch.sqrt(torch.log(self.t)/self.count)
        UCB_value = q_value + expl_term
        action = UCB_value.argmax().cpu().item()
        self.count[action] += 1
        return action

class EpsGreedy(ExplorationProcess):
    def __init__(self, eps_deb=1.0, eps_fin=0.1, deb_expl=0, fin_expl=0.9,device='cpu'):
        self.eps_deb = eps_deb
        self.eps_fin = eps_fin
        self.eps = self.eps_deb
        if fin_expl < deb_expl:
            raise ValueError("'deb_expl' must be lower than 'fin_expl'")
        self.deb_expl = deb_expl
        self.fin_expl = fin_expl

    def reset(self, t=None):
        """ Reinitialize the state of the process """
        """ Reinitialize some parameters """
        self.eps = self.eps_deb
        self.init_expl_step = int(self.deb_expl * t)
        self.final_expl_step = int(self.fin_expl * t)
        
    def update(self, t):
        if t >= self.init_expl_step:
            if self.eps_deb >= self.eps_fin:
                self.eps = max(self.eps_fin, self.eps_deb - (t-self.init_expl_step)*(self.eps_deb-self.eps_fin)/(self.final_expl_step-self.init_expl_step))
            else:
                self.eps = min(self.eps_fin, self.eps_deb - (t-self.init_expl_step)*(self.eps_deb-self.eps_fin)/(self.final_expl_step-self.init_expl_step))
        return self.eps

    def greedy_action(self, policy):
        q_value = policy.model() 
        action = q_value.argmax().cpu().item()       
        return action

    def expl_action(self,policy):
        q_value = policy.model()
        action = torch.randint(0,len(q_value),(1,)).cpu().item()
        return action

    def __call__(self, policy):
        if random.random() < self.eps:
            random_action = self.expl_action(policy)
            return random_action
        else:
            greedy_action = self.greedy_action(policy)
            return greedy_action