import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import time
import copy

#Config Globals
K = 2.422
RANDOM_REWIRE_PROB = 0.0

def create_random_nets(save_dir,num_nodes,num2gen=10,gen_threshes=False,show=False):  
    #import random
    #random.seed(np.random.randint(10000))
    for i in range(num2gen):
        [network_a, network_b] = create_networks('SF',num_nodes=num_nodes)
        if save_dir != '':
            f = save_dir + 'net_{}.edgelist'.format(i)
            nx.write_edgelist(network_b,f)
            if gen_threshes:
                thresholds = []
                for node in network_b.nodes():
                    thresh = 1/len(network_b[node])*np.random.choice([i for i in range(1,len(network_b[node])+1)])
                    thresholds.append(thresh)
                ft = save_dir + 'net_{}_thresh.npy'.format(i)
                np.save(ft,np.asarray(thresholds))
                #print(f'Saved to {ft}')
    if show:
        print('Showing one of the generated networks')
        import matplotlib.pyplot as plt
        nx.draw(network_b,with_labels=True)
        plt.draw()
        plt.show()
    return [network_a,network_b]
    
def make_comms(network_b, copy,num_nodes=None):
    if num_nodes is None:
        print("Num nodes not specified")
        exit()
    if RANDOM_REWIRE_PROB == -1:
        degree_sequence = sorted(nx.degree(network_b).values(), reverse=True)
        network_a = nx.gnerators.configuration_model(degree_sequence)
    else:
        if copy is True:
            network_a = network_b.copy()
        else:
            network_a = network_b
        nodes = []
        targets = []
        for i, j in network_a.edges():
            nodes.append(i)
            targets.append(j)
        # rewire edges from each node, adapted from NetworkX W/S graph generator
        # http://networkx.github.io/documentation/latest/_modules/networkx/generators/random_graphs.html#watts_strogatz_graph
        # no self loops or multiple edges allowed
        for u, v in network_a.edges():
            if random.random() < RANDOM_REWIRE_PROB:
                w = random.choice(nodes)
                # Enforce no self-loops or multiple edges
                while w == u or network_a.has_edge(u, w):
                    w = random.choice(nodes)
                # print "node: " + str(u) + ", target: " + str(v)
                network_a.remove_edge(u, v)
                network_a.add_edge(u, w)
    if copy is True:
        mapping = dict(zip(network_a.nodes(), range(1, num_nodes + 1)))  # 2384 for Polish, 4942 for western. relabel the nodes to start at 1 like network_a
        network_a = nx.relabel_nodes(network_a, mapping)
    return network_a


def create_networks(network_type,num_nodes,verbose=False):
    #config vars
    if num_nodes is None:
        print("Num nodes not specified")
        exit()
    if network_type == 'ER':
        # Erdos-Renyi random graphs
        network_a = nx.generators.random_graphs.gnp_random_graph(num_nodes, ep)
        network_b = nx.generators.random_graphs.gnp_random_graph(num_nodes, ep)
    elif network_type == 'RR':
        # random regular graphs
        network_a = nx.random_regular_graph(int(K), num_nodes)
        network_b = nx.random_regular_graph(int(K), num_nodes)
    elif network_type == 'SF':
        # Scale free networks
        # m==2 gives <k>==4, for this lambda/gamma is always 3
        network_a = nx.barabasi_albert_graph(num_nodes, 2)
        network_b = nx.barabasi_albert_graph(num_nodes, 2)
        if RANDOM_REWIRE_PROB != -1:
            network_a = make_comms(network_a, False,num_nodes=num_nodes)
            network_b = make_comms(network_b, False,num_nodes=num_nodes)
    elif network_type == 'Lattice':
        l = math.sqrt(num_nodes)
        if(n % l != 0):
            print("Number of nodes, " + str(num_nodes) + ", not square (i.e. sqrt(n) has a remainder) for lattice. Adjust n and retry.")
            raise
            sys.exit(-1)
        l = int(l)
        network_a = nx.grid_2d_graph(l, l, periodic=True)
        network_b = nx.grid_2d_graph(l, l, periodic=True)
    elif network_type == 'CFS-SW':
        '''network_b is a topological representation of the power network. network_a
        is a generated  communication network created through either a configuration
        model of the power grid or by randomly rewiring the power grid topology.
        '''
        path = relpath + "data/power-grid/network_Polish_2383nodes_2.422deg_0.0rw_12345seed.edgelist"
        network_b = nx.read_edgelist(path, nodetype=int)
        mapping = dict(zip(network_b.nodes(), range(1, num_nodes + 1)))  # renumber the nodes to start at 1 for MATLAB
        network_b = nx.relabel_nodes(network_b, mapping)
        '''make the comms network'''
        if verbose is True:
            print("^^^^^ Making comm network ^^^^^^")
        network_a = make_comms(network_b, True)
    else:
        print('Invalid network type: ' + network_type)
        return []
    return [network_a, network_b]

def ncr(n, r):
    import operator as op
    from functools import reduce
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2

def get_combinatorial_actions(total_nodes,num_nodes_chosen):
    if num_nodes_chosen == 1:
        return [i for i in range(total_nodes)]
    #Note: currently only works for num_nodes_chosen = 2
    num_actions = ncr(total_nodes,num_nodes_chosen)
    curr_action = [i for i in range(num_nodes_chosen)]
    last_action = [i for i in range(total_nodes-1,total_nodes-num_nodes_chosen-1,-1)]
    last_action.reverse()
    all_actions = [tuple(curr_action.copy())]
    while curr_action != last_action:
        for i in range(num_nodes_chosen,0,-1):
            if curr_action[i-1] < total_nodes-(num_nodes_chosen-i+1):
                curr_action[i-1] += 1
                break
            else:
                curr_action[i-1] = curr_action[i-2]+2
                for j in range(i,num_nodes_chosen):
                    curr_action[j] = curr_action[j-1]+1
        all_actions.append(tuple(curr_action.copy()))
    return all_actions

def get_rtmixed_nash(envs,targeted_policy,random_policy):
    print('Getting NashEQ for RTMixed Benchmark Strategy')
    tic = time.perf_counter()
    num_data = 1000
    p_atk = np.zeros(len(envs))
    p_def = np.zeros(len(envs))
    for i,env in enumerate(envs):
        U = np.zeros((2,2)) #[[atdt,atdr],[ardt,ardr]]
        atk_policy = copy.deepcopy(targeted_policy)
        def_policy = copy.deepcopy(random_policy)
        rewards = []
        for j in range(num_data):
            obs = env.reset()
            action = [atk_policy([obs])[0],def_policy([obs])[0]]
            _,reward,_,_ = env.step(action)
            rewards.append(reward[0])
        U[0,1] = np.mean(rewards)

        atk_policy = copy.deepcopy(random_policy)
        def_policy = copy.deepcopy(targeted_policy)
        rewards = []
        for j in range(num_data):
            obs = env.reset()
            action = [atk_policy([obs])[0],def_policy([obs])[0]]
            _,reward,_,_ = env.step(action)
            rewards.append(reward[0])
        U[1,0] = np.mean(rewards)

        atk_policy = copy.deepcopy(random_policy)
        def_policy = copy.deepcopy(random_policy)
        rewards = []
        for j in range(num_data):
            obs = env.reset()
            action = [atk_policy([obs])[0],def_policy([obs])[0]]
            _,reward,_,_ = env.step(action)
            rewards.append(reward[0])
        U[1,1] = np.mean(rewards)
        if U[1,0] <= U[1,1]:
            p_atk[i] = 0
            p_def[i] = 1
        else:
            if U[0,1] <= U[1,1]:
                p_atk[i] = 0
                p_def[i] = 0
            else:
                p_atk[i] = (U[1,0] - U[1,1])/(U[0,1]+U[1,0]-U[1,1])
                p_def[i] = (U[0,1] - U[1,1])/(U[0,1]+U[1,0]-U[1,1])
        print(U)
    toc = time.perf_counter()
    print(f'Finished in {toc-tic} seconds')
    return p_atk,p_def