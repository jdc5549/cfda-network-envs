import os
import numpy as np
import pygambit as gambit

from netcasc_gym_env import NetworkCascEnv
from utils import create_random_nets, ncr, get_combinatorial_actions

def get_nash_eqs(env,env_id):
    progress_step = 0.20
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
    print(f'Env {env_id}: Caculating Ultility Matrix of size {num_actions*num_actions}')
    count = 0
    last_print = 0
    for i in range(num_actions):
        for j in range(num_actions):
            node_lists = [all_actions[i],all_actions[j]]
            last_rew = 0
            _,reward,_,_ = env.step(node_lists)
            U[i,j] = gambit.Rational(reward[0])
            count += 1
            if (count - last_print)/(num_actions*num_actions) >= progress_step:
                print(f'Env {env_id}: {round(count/(num_actions*num_actions)*100,2)}% of Utility Matrix Calculated')
                last_print = count
    #U = U.astype(gambit.Rational)
    print(f'Env {env_id}: Caculating Nash Eqs')
    g = gambit.Game.from_arrays(U,-U)
    g.players[0].label = 'Attacker'
    g.players[1].label = 'Defender'
    eqs = gambit.nash.lp_solve(g)
    eqs = np.array(eqs,dtype=float)
    eqs = np.reshape(eqs,(2,num_actions))
    U = np.array(U,dtype=float)
    print(f'Env {i}: Done.')
    return eqs, U

def _gen_utils_eqs(fn_c_i_dir_nn):
    fn = fn_c_i_dir_nn[0]
    c = fn_c_i_dir_nn[1]
    i = fn_c_i_dir_nn[2]
    save_dir = fn_c_i_dir_nn[3]
    num_nodes = fn_c_i_dir_nn[4]
    p = 2/num_nodes
    env = NetworkCascEnv(num_nodes,p,p,6,'File',filename=fn,cascade_type=c)
    eqs,U = get_nash_eqs(env,i)
    f_eq = save_dir + f'eq_{i}.npy'
    np.save(f_eq,eqs)
    f_util = save_dir + f'util_{i}.npy'
    np.save(f_util,U)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='SL Dataset Creation Args')
    parser.add_argument("--train",default=False,type=bool,help='Whether to create a training or validation set')
    parser.add_argument("--cfda",default=False,type=bool,help='Whether to use CFA for exploration.')
    parser.add_argument("--calc_nash",default=False,type=bool,help='Whether to calculate the NashEQ instead of perform trials in a validation set.')
    parser.add_argument("--net_size",default=10,type=int,help='Number of nodes in the networks to generate.')
    parser.add_argument("--num_nodes_chosen",default=2,type=int,help='Number of nodes attacker and defender choose to attack/defend')
    parser.add_argument("--num_topologies",default=100,type=int,help='Number of Topologies to generate')
    parser.add_argument("--num_trials",default=10,type=int,help='Number of attack/defense trials per episode')
    parser.add_argument("--cascade_type",default='threshold',type=str,help='Type of cascading model to use in cascading failure simulation.')
    parser.add_argument("--load_dir",default=None,type=str,help='Specifies dir to load topologies from instead of generating new ones.')
    parser.add_argument("--exploration_type",default='Random',type=str,help="Which exploration strategy to use to gen trials")
    parser.add_argument("--epsilon",default=0.99,type=int,help="Epsilon paramter for CDMExploration.")
    parser.add_argument("--cfa",default=False,type=bool,help='Whether to use CFA for training.')
    args = parser.parse_args()


    if args.cascade_type not in ['threshold','shortPath','coupled']:
        print(f'Unrecognized Cascade Type in {casc}. Recognized Cascade Types are {Cascade_Types}. Exiting.')
        exit()
    elif args.cascade_type == 'threshold' or args.cascade_type == 'shortestPath':
        gen_threshes = True
    else:
        gen_threshes = False

    if args.load_dir is not None:
        topology_dir = load_dir + 'topologies/'
        data_dir = load_dir
    else:
        sub_dir = 'training_data' if args.train else 'validation_data'
        eval_method =  f'{args.num_trials}trials' if (args.train or not args.calc_nash) else 'NashEQ'
        save_dir = f'./data/{args.net_size}C2/{sub_dir}/{args.num_topologies}topo_{eval_method}_{args.exploration_type}Expl'
        save_dir += '_CfDA/' if args.cfda else '/'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
            os.makedirs(save_dir + 'topologies/')
        else:
            print(f'A dataset with the specified parameters already exists in the directory {save_dir}. If you would like to generate a new one, please remove or rename this directory.')
        topology_dir = save_dir + 'topologies/'
        create_random_nets(topology_dir,args.net_size,gen_threshes=gen_threshes,num2gen=args.num_topologies)
        data_dir = save_dir

    if args.train or not args.calc_nash:
        from SL_exploration import SLExploration, RandomExploration, CDMExploration
        comb_acts = get_combinatorial_actions(args.net_size,args.num_nodes_chosen)
        if args.exploration_type == 'Random':
            exploration = RandomExploration(comb_acts)
        elif args.exploration_type == 'CDME':
            exploration = CDMExploration(comb_acts)
        else:
            print(f'Exploration type {args.exploration_type} not recognized. Exiting...')
            exit()
        trial_data = np.zeros([args.num_topologies,args.num_trials,5]) #last dim represents 2 attack nodes, 2 defense nodes, and attacker reward (in that order)
        p = 2/args.net_size
        for i in range(args.num_topologies):
            exploration.reset()
            env = NetworkCascEnv(args.net_size,p,p,6,'File',filename=topology_dir + f'net_{i}.edgelist',cascade_type=args.cascade_type)
            for j in range(args.num_trials):
                action = exploration()
                exploration.update()
                _, reward, _, _ = env.step(action)
                trial_data[i,j,:] = np.concatenate((action[0], action[1], [reward[0]]))
        np.save(data_dir + f'{args.cascade_type}casc_trialdata.npy',trial_data)s
    else:
        import multiprocessing as mp
        nash_eq_dir = data_dir + f'{args.cascade_type}casc_NashEQs/'
        os.makedirs(nash_eq_dir)
        #tic = time.perf_counter()
        files = [f for f in os.listdir(topology_dir) if 'thresh' not in f]
        f_args = []
        for i,f in enumerate(files):
            f_args.append((os.path.join(topology_dir,f),args.cascade_type,i,nash_eq_dir,args.net_size))
        with mp.Pool(processes=min([len(f_args),mp.cpu_count()-2])) as pool:
            pool.map(_gen_utils_eqs, f_args)
        pool.close()

        # toc = time.perf_counter()
        # print(f"Completed in {toc - tic:0.4f} seconds")







