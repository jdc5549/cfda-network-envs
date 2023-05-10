import random

class SLExploration(object): #general SL exploration class
    def reset(self):
        pass

    def update(self):
        pass

    def __call__(self):
        raise NotImplementedError

class RandomExploration(SLExploration): #general SL exploration class
    def __init__(self,all_combinations):
        self.all_actions = [[c1,c2] for c1 in all_combinations for c2 in all_combinations]

    def __call__(self):
        action = random.choice(self.all_actions)
        return action

class CDMExploration(SLExploration): #general SL exploration class
    def __init__(self,all_combinations,eps=0.99):
        max_value = max(max(sublist) for sublist in all_combinations)
        self.n_items = max_value + 1
        self.all_actions = [[c1,c2] for c1 in all_combinations for c2 in all_combinations]
        self.eps = eps

    def reset():
        self.count_atk = [0 for _ in range(self.n_items)]
        self.count_def = [0 for _ in range(self.n_items)]
        self.action_count = [0 for _ in range(len(self.all_actions))]
        self.action_score = [0 for _ in range(len(self.all_actions))]
    
    def update(self):
        for i,act in enumerate(self.all_actions):
            self.action_score[i] = self.eps*self.action_count[i] + (1-self.eps)*(sum([self.count_atk[a] for a in act[0]])+sum([self.count_def[a] for a in act[1]]))
        return self.eps

    def __call__(self):
        min_value = min(self.action_score)
        min_indices = [i for i,x in enumerate(self.action_score) if x == min_value]
        min_index = random.choice(min_indices)
        action = self.all_actions[min_index]
        self.action_count[min_index] += 1
        for a in action[0]:
            self.count_atk[a] += 1
        for a in action[1]:
            self.count_def[a] += 1
        return action