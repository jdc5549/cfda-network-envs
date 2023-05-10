import pygambit as gambit
import numpy as np
import networkx as nx
import time
import os
from utils import create_random_nets, ncr

def get_nash_eqs(env,env_id,cascade_type):
    progress_step = 0.10
    num_nodes_attacked = env.num_nodes_attacked
    net_size = env.net_size
    num_actions = ncr(net_size,num_nodes_attacked)
    U = np.zeros([num_actions,num_actions],dtype=gambit.Rational)
    curr_action = [i for i in range(num_nodes_attacked)]
    last_action = [i for i in range(net_size-1,net_size-num_nodes_attacked-1,-1)]
    last_action.reverse()
    all_actions = [curr_action.copy()]
    while curr_action != last_action:
        for i in range(num_nodes_attacked,0,-1):
            if curr_action[i-1] < net_size-(num_nodes_attacked-i+1):
                curr_action[i-1] += 1
                break
            else:
                curr_action[i-1] = curr_action[i-2]+2
                for j in range(i,num_nodes_attacked):
                    curr_action[j] = curr_action[j-1]+1
        all_actions.append(curr_action.copy())
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
    U = np.array(U,dtype=float)
    print(f'Env {i} {c} Casc: Done.')
    return eqs, U

def _gen_utils_eqs(fnci):
    fn = fnci[0]
    c = fnci[1]
    i = fnci[2]
    if args.env_type == 'NetworkCascEnv':
        env = NetworkCascEnv(args.num_nodes,args.p,args.p,6,'File',filename=fn,cascade_type=c)
    else:
        print(f'Environment type {args.env_type} is not supported')
    eqs,U = get_nash_eqs(env,i,c)
    f_eq = args.nash_eqs_dir + f'{c}Casc_eq_{i}.npy'
    np.save(f_eq,eqs)
    f_util = args.nash_eqs_dir + f'{c}Casc_util_{i}.npy'
    np.save(f_util,U)

if __name__ == '__main__':
    import argparse
    from netcasc_gym_env import NetworkCascEnv
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Network Generation Args')
    parser.add_argument("--num_nodes",type=int,default=100)
    parser.add_argument("--num2gen",type=int,default=10)
    parser.add_argument("--net_save_dir",default='data/networks/generated/',type=str,help='Directory where the network topologies will be saved.')
    parser.add_argument("--nash_eqs_dir",default=None,type=str,help='Directory where Nash EQ benchmarks will be written. If None (default), then does not calculate Nash EQs.')
    parser.add_argument("--p",default=0.1,type=float,help='If calculating Nash EQs, the percent of nodes to be attacked/defended.')
    parser.add_argument("--env_type", default='NetworkCascEnv',help='What type of gym environment should be used to generate the NashEQ utility')
    parser.add_argument("--cascade_type", default='all', help='What type of cascading to use to get utility.')
    args = parser.parse_args()
    Cascade_Types = ['threshold','shortPath','coupled']

    if args.net_save_dir[-1] != '/':
        args.net_save_dir += '/'
    #full_dir = args.net_save_dir + f'SF_{args.num_nodes}n_2.422deg_{args.env_type}_p{args.p}_{args.cascade_type}Casc_{args.num2gen}nets/'
    if not os.path.isdir(args.net_save_dir):
        os.mkdir(args.net_save_dir)
    if args.env_type == 'NetworkCascEnv':
        gen_threshes = True
    else:
        gen_threshes = False
    create_random_nets(args.net_save_dir,args.num_nodes,gen_threshes=gen_threshes,num2gen=args.num2gen)
    if args.nash_eqs_dir is not None:
        if args.nash_eqs_dir[-1] != '/':
            args.nash_eqs_dir += '/'
        #args.nash_eqs_dir = args.nash_eqs_dir + f'SF_{args.num_nodes}n_2.422deg_{args.env_type}_p{args.p}_{args.cascade_type}Casc_{args.num2gen}nets/'
        if not os.path.isdir(args.nash_eqs_dir):
            os.mkdir(args.nash_eqs_dir)
        tic = time.perf_counter()
        files = [f for f in os.listdir(args.net_save_dir) if 'thresh' not in f]
        if args.cascade_type == 'all': casc = Cascade_Types 
        else: casc = [args.cascade_type]
        if casc not in Cascade_Types:
            print(f'Unrecognized Cascade Type in {casc}. Recognized Cascade Types are {Cascade_Types}.')
        f_args = []
        for c in casc:
            for i,f in enumerate(files):
                f_args.append((os.path.join(args.net_save_dir,f),c,i))
        with mp.Pool(processes=min([len(f_args),mp.cpu_count()-2])) as pool:
            pool.map(_gen_utils_eqs, f_args)
        pool.close()

        toc = time.perf_counter()
        print(f"Completed in {toc - tic:0.4f} seconds")
