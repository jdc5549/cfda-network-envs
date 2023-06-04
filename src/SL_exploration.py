import random

class SLExploration(object): #general SL exploration class
    def __init__(self,target_set):
        all_combinations = []
        for i in range(len(target_set)):
            for j in range(i + 1, len(target_set)):
                all_combinations.append((min(target_set[i], target_set[j]), max(target_set[i], target_set[j])))
        self.all_actions = [[c1,c2] for c1 in all_combinations for c2 in all_combinations]

    def reset(self):
        pass

    def update(self):
        pass

    def __call__(self):
        raise NotImplementedError

class RandomExploration(SLExploration):
    def __call__(self):
        action = random.choice(self.all_actions)
        return action

class RandomCycleExploration(SLExploration): 
    def __init__(self,target_set):
        super().__init__(target_set)
        self.order = self.all_actions[:]
        random.shuffle(self.order)
        self.index = 0

    def reset(self):
        random.shuffle(self.order)
        self.index = 0

    def update(self):
        self.index += 1

    def __call__(self):
        i = self.index % len(self.all_actions)
        return self.order[i] 

class CDMExploration(SLExploration): 
    def __init__(self,target_set,eps=0.99):
        super().__init__(target_set)
        self.n_items = len(target_set)
        self.count_atk = [0 for _ in range(self.n_items)]
        self.count_def = [0 for _ in range(self.n_items)]
        self.eps = eps

    def reset(self):
        self.count_atk = [0 for _ in range(self.n_items)]
        self.count_def = [0 for _ in range(self.n_items)]
        self.action_count = [0 for _ in range(len(self.all_actions))]
        self.action_score = [0 for _ in range(len(self.all_actions))]

    
    def update(self):
        for i,act in enumerate(self.all_actions):
            self.action_score[i] = self.eps*self.action_count[i] + (1-self.eps)*(sum([self.count_atk[a] for a in act[0]])+sum([self.count_def[a] for a in act[1]]))

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

if __name__ == '__main__':
    from utils import get_combinatorial_actions,ncr
    import numpy as np
    all_actions = get_combinatorial_actions(5,2)
    exploration = CDMExploration(all_actions)
    exploration.reset()
    actions = []
    for i in range(100):
        action = exploration()
        actions.append(np.array(action))
        exploration.update()

    match_count = 0
    for i,a1 in enumerate(actions):
        for j,a2 in enumerate(actions):
            if i != j and np.array_equal(a1,a2):
                match_count += 1
    print(match_count)





