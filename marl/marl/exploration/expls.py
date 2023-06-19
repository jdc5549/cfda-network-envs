import math
import torch
from . import ExplorationProcess

class UCB1(ExplorationProcess):
    def __init__(self, n_actions,c=1.7):
        self.n_actions = n_actions
        self.count = torch.tensor([1 for _ in range(n_actions)])
        self.t = torch.tensor(sum(self.count))
        self.c = c

    def reset(self, t=None):
        """ Reinitialize the state of the process """
        self.count = [1 for _ in range(self.n_actions)]         
        self.t = sum(self.count)
        
    def update(self, t):
        self.t = torch.tensor(sum(self.count))
            
    def __call__(self, policy):
        q_value = policy.model()
        expl_term = self.c * torch.sqrt(torch.log(self.t)/self.count)
        UCB_value = q_value + expl_term
        action = UCB_value.argmax().cpu().item()
        self.count[action] += 1
        return action