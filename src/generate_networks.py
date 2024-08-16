import pygambit as gambit
import numpy as np
import networkx as nx
import time
import os
from utils import create_networks, get_combinatorial_actions, ncr

def get_nash_eqs(env,env_id,cascade_type):
    progress_step = 0.10
    num_nodes_attacked = env.num_nodes_attacked
    net_size = env.net_size
    num_actions = ncr(net_size,num_nodes_attacked)
    U = np.zeros([num_actions,num_actions],dtype=gambit.Rational)
    all_actions = get_combinatorial_actions(net_size,num_nodes_attacked)
    # curr_action = [i for i in range(num_nodes_attacked)]
    # last_action = [i for i in range(net_size-1,net_size-num_nodes_attacked-1,-1)]
    # last_action.reverse()
    # all_actions = [curr_action.copy()]
    # while curr_action != last_action:
    #     for i in range(num_nodes_attacked,0,-1):
    #         if curr_action[i-1] < net_size-(num_nodes_attacked-i+1):
    #             curr_action[i-1] += 1
    #             break
    #         else:
    #             curr_action[i-1] = curr_action[i-2]+2
    #             for j in range(i,num_nodes_attacked):
    #                 curr_action[j] = curr_action[j-1]+1
    #     all_actions.append(curr_action.copy())
    print(f'Env {env_id} {cascade_type} Casc: Caculating Ultility Matrix of size {num_actions*num_actions}')
    count = 0
    last_print = 0
    for i in range(num_actions):
        for j in range(num_actions):
            node_lists = [all_actions[i],all_actions[j]]
            last_rew = 0
            _,reward,_,_ = env.step(node_lists)
            U[i,j] = gambit.Rational(reward[0])
            count += 1
            if (count - last_print)/(num_actions*num_actions) >= 0.1:
                print(f'Env {env_id} {cascade_type} Casc: {round(count/(num_actions*num_actions)*100,2)}% of Utility Matrix Calculated')
                last_print = count
    #U = U.astype(gambit.Rational)
    print(f'Env {env_id} {cascade_type} Casc: Caculating Nash Eqs')
    g = gambit.Game.from_arrays(U,-U)
    g.players[0].label = 'Attacker'
    g.players[1].label = 'Defender'
    eqs = gambit.nash.lp_solve(g)

    eqs = np.array(eqs,dtype=float)
    eqs = np.reshape(eqs,(2,num_actions))
    print(f'Env {i} {cascade_type} Casc: Done.')
    U = np.array(U,dtype=float)
    return eqs, U

def _gen_utils_eqs(fnci):
    fn = fnci[0]
    c = fnci[1]
    i = fnci[2]

    filename = os.path.basename(fn)
    filename, _ = os.path.splitext(filename)
    env = NetworkCascEnv(args.p,args.p,'File',6,filename=fn,cascade_type=c,degree=args.p)
    eqs,U = get_nash_eqs(env,i,c)
    print(f'U: {U}')
    f_util = args.nash_eqs_dir + f'{c}Casc_{filename}_d{env.degree}_util.npy'
    np.save(f_util,U)
    f_eq = args.nash_eqs_dir + f'{c}Casc_{filename}_d{env.degree}_eq.npy'
    print(f'EQ: {eqs}')
    np.save(f_eq,eqs)

def get_toy_eq(nash_eqs_dir,node_map,degree=2):
    num_clusters = int(max(node_map)+1)
    num_nodes = len(node_map)
    num_nodes_attacked = degree
    good_cluster_combs = get_combinatorial_actions(num_clusters,degree)
    pairs_zero = [(-1,n) for n in range(num_clusters)]
    same_pairs = [(n,n) for n in range(num_clusters)]
    wrong_cluster_combs = pairs_zero + same_pairs
    all_cluster_actions = good_cluster_combs + wrong_cluster_combs

    num_cluster_actions = len(all_cluster_actions)

    reward_map = np.zeros(num_clusters)
    for cluster in node_map:
        cluster = int(cluster)
        if cluster >= 0:
            reward_map[cluster] += 1/num_nodes

    U_cluster = np.zeros([num_cluster_actions,num_cluster_actions],dtype=gambit.Rational)
    
    print(f'Caculating Ultility Matrix of size {num_cluster_actions*num_cluster_actions}')
    count = 0
    for i in range(num_cluster_actions):
        for j in range(num_cluster_actions):
            atk_list = all_cluster_actions[i]
            def_list = all_cluster_actions[j]
            casc_list = list(set([cluster for cluster in atk_list if cluster not in def_list]))
            reward = 0
            for cluster in casc_list:
                if cluster != -1:
                    reward += reward_map[cluster]
                else:
                    reward += 1/num_nodes
            U_cluster[i,j] = gambit.Rational(reward)
    print(f'Caculating Nash Eqs')
    g = gambit.Game.from_arrays(U_cluster,-U_cluster)
    g.players[0].label = 'Attacker'
    g.players[1].label = 'Defender'
    eqs_cluster = gambit.nash.lp_solve(g)
    eqs_cluster = np.array(eqs_cluster,dtype=float)
    eqs_cluster = np.reshape(eqs_cluster,(2,num_cluster_actions))


    print(f'Done.')
    print(f'EQ (cluster): {eqs_cluster}')
    f_eq = nash_eqs_dir + f'/net_0_eq_cluster.npy'
    np.save(f_eq,eqs_cluster)
    print(f'U (cluster): {np.array(U_cluster,dtype=float)}')
    f_util = nash_eqs_dir + f'/net_0_util_cluster.npy'
    U_cluster = np.array(U_cluster,dtype=float)
    np.save(f_util,U_cluster)

    #Convert back to nodes
    all_actions = get_combinatorial_actions(num_nodes,degree)
    num_actions = len(all_actions)
    if num_actions > 225000:
        print('Utility Matrix too large for node representaion')
    else:
        cluster_reps = {}
        for node, cluster in enumerate(node_map):
            if cluster not in cluster_reps:
                cluster_reps[cluster] = [node]
            else:
                cluster_reps[cluster].append(node)

        U = np.zeros([num_actions,num_actions],dtype=gambit.Rational)
        for i in range(num_actions):
            for j in range(num_actions):
                action_i = all_actions[i]
                action_j = all_actions[j]
                cluster_action_i = tuple(sorted([node_map[node] for node in action_i]))
                cluster_action_j = tuple(sorted([node_map[node] for node in action_j]))
                if cluster_action_i in all_cluster_actions and cluster_action_j in all_cluster_actions:
                    idx_i = all_cluster_actions.index(cluster_action_i)
                    idx_j = all_cluster_actions.index(cluster_action_j)
                    U[i, j] = U_cluster[idx_i, idx_j]
                else:
                    U[i, j] = gambit.Rational(0)
        U = np.array(U,dtype=float)

        eqs = np.zeros((2,num_actions))
        for i,eq_c in enumerate(eqs_cluster):
            for j,p_c in enumerate(eq_c):
                #action = tuple(sorted([cluster_reps[c] for c in all_cluster_actions[j]]))
                action = []
                for c in all_cluster_actions[j]:
                    if cluster_reps[c][0] not in action:
                        action.append(cluster_reps[c][0])
                    else:
                        action.append(cluster_reps[c][1])
                action = tuple(action)
                idx = all_actions.index(action)
                eqs[i,idx] = p_c

        print(f'EQ: {eqs}')
        f_eq = nash_eqs_dir + f'/net_0_eq.npy'
        np.save(f_eq,eqs)
        print(f'U: {U}')
        f_util = nash_eqs_dir + f'/net_0_util.npy'
        np.save(f_util,U)
        return eqs, U

if __name__ == '__main__':
    import argparse
    from netcasc_gym_env import NetworkCascEnv
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Network Generation Args')
    parser.add_argument("--graph_file",default='data/',type=str,help='Directory where the network topologies will be saved.')
    parser.add_argument("--num_nodes",default=10,type=int,help='Number of nodes if generating.')
    parser.add_argument("--cascade_type", default='threshold', help='What type of cascading to use to get utility.')
    parser.add_argument("--p",type=int,default=1)
    args = parser.parse_args()

    if os.path.isdir(args.graph_file):
        graph_path = args.graph_file + f'{args.num_nodes}/net_0.gpickle'
        [network_a, G] = create_networks('SF',num_nodes=args.num_nodes)
        thresh = np.random.uniform(0, 1, size=len(G.nodes()))
        for node, threshold in zip(G.nodes(), thresh):
            nx.set_node_attributes(G, {node: {'threshold': threshold}})
            nx.write_gpickle(G,graph_path)
        dir_name = os.path.dirname(graph_path)
    elif os.path.isfile(args.graph_file):
        G = nx.read_gpickle(args.graph_file)
        dir_name = os.path.dirname(args.graph_file)
        graph_path = args.graph_file

    exp_name = os.path.basename(dir_name)
    nash_eqs_dir = f'{dir_name}/{args.cascade_type}casc_NashEQs/'
    args.nash_eqs_dir = nash_eqs_dir
    args.num_nodes = G.number_of_nodes()
    if not os.path.isdir(nash_eqs_dir):
        os.mkdir(nash_eqs_dir)
    if 'toy' in dir_name:
        from analyze_results import get_toy_clusters
        node_map = get_toy_clusters(G)
        eqs,U = get_toy_eq(nash_eqs_dir,node_map,args.p)
    else:
        tic = time.perf_counter()
        fnci = (graph_path,args.cascade_type,0)
        _gen_utils_eqs(fnci)
        toc = time.perf_counter()
        print(f"Completed in {toc - tic:0.4f} seconds")


    # parser = argparse.ArgumentParser(description='Network Generation Args')
    # parser.add_argument("--num_nodes",type=int,default=100)
    # parser.add_argument("--num2gen",type=int,default=10)
    # parser.add_argument("--net_save_dir",default='data/networks/generated/',type=str,help='Directory where the network topologies will be saved.')
    # parser.add_argument("--nash_eqs_dir",default=None,type=str,help='Directory where Nash EQ benchmarks will be written. If None (default), then does not calculate Nash EQs.')
    # parser.add_argument("--p",default=0.1,type=float,help='If calculating Nash EQs, the percent of nodes to be attacked/defended.')
    # parser.add_argument("--env_type", default='NetworkCascEnv',help='What type of gym environment should be used to generate the NashEQ utility')
    # parser.add_argument("--cascade_type", default='all', help='What type of cascading to use to get utility.')
    # args = parser.parse_args()
    # Cascade_Types = ['threshold','shortPath','coupled']

    # if args.net_save_dir[-1] != '/':
    #     args.net_save_dir += '/'
    # #full_dir = args.net_save_dir + f'SF_{args.num_nodes}n_2.422deg_{args.env_type}_p{args.p}_{args.cascade_type}Casc_{args.num2gen}nets/'
    # if not os.path.isdir(args.net_save_dir):
    #     os.mkdir(args.net_save_dir)
    # if args.env_type == 'NetworkCascEnv':
    #     gen_threshes = True
    # else:
    #     gen_threshes = False
    # create_random_nets(args.net_save_dir,args.num_nodes,gen_threshes=gen_threshes,num2gen=args.num2gen)
    # if args.nash_eqs_dir is not None:
    #     if args.nash_eqs_dir[-1] != '/':
    #         args.nash_eqs_dir += '/'
    #     #args.nash_eqs_dir = args.nash_eqs_dir + f'SF_{args.num_nodes}n_2.422deg_{args.env_type}_p{args.p}_{args.cascade_type}Casc_{args.num2gen}nets/'
    #     if not os.path.isdir(args.nash_eqs_dir):
    #         os.mkdir(args.nash_eqs_dir)
    #     tic = time.perf_counter()
    #     files = [f for f in os.listdir(args.net_save_dir) if 'thresh' not in f]
    #     if args.cascade_type == 'all': casc = Cascade_Types 
    #     else: casc = [args.cascade_type]
    #     if casc not in Cascade_Types:
    #         print(f'Unrecognized Cascade Type in {casc}. Recognized Cascade Types are {Cascade_Types}.')
    #     f_args = []
    #     for c in casc:
    #         for i,f in enumerate(files):
    #             f_args.append((os.path.join(args.net_save_dir,f),c,i))
    #     with mp.Pool(processes=min([len(f_args),mp.cpu_count()-2])) as pool:
    #         pool.map(_gen_utils_eqs, f_args)
    #     pool.close()

    #     toc = time.perf_counter()
    #     print(f"Completed in {toc - tic:0.4f} seconds")
