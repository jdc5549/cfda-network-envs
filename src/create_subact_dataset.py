import os
import numpy as np
import pygambit as gambit
import nashpy as nash
import pickle
import random
import time

from netcasc_gym_env import NetworkCascEnv
from cascade_cfda import Counterfactual_Cascade_Fns
from utils import create_random_nets, ncr, get_combinatorial_actions
from scipy.optimize import linprog
from scipy.stats import entropy

def get_nash_eqs(env):
    progress_step = 0.2
    num_nodes_attacked = env.num_nodes_attacked
    net_size = env.net_size
    all_actions = get_combinatorial_actions(net_size,num_nodes_attacked)
    num_actions = len(all_actions)
    U = np.zeros([num_actions,num_actions],dtype=gambit.Rational)
    print(f'Caculating Ultility Matrix of size {num_actions*num_actions}')
    count = 0
    last_print = 0
    for i in range(num_actions):
        for j in range(num_actions):
            node_lists = [all_actions[i],all_actions[j]]
            last_rew = 0
            _,reward,_,_ = env.step(node_lists)
            U[i,j] = reward[0]#gambit.Rational(reward[0])
            count += 1
            if (count - last_print)/(num_actions*num_actions) >= progress_step:
                print(f'{round(count/(num_actions*num_actions)*100,2)}% of Utility Matrix Calculated')
                last_print = count
    #U = U.astype(gambit.Rational)
    print(f'Caculating Nash Eqs')

    # security_game = nash.Game(U)
    # equilibria = security_game.support_enumeration()
    # for eq in equilibria:
    #     print(eq)
    # exit()
    c = np.zeros(num_actions+1)
    c[num_actions] = -1

    U_calc = -U.T
    A_ub = np.hstack((U_calc,np.ones((num_actions,1))))
    b_ub = np.zeros(num_actions)
    A_eq = np.ones((1,num_actions+1))
    A_eq[:,num_actions] = 0
    b_eq = 1
    bounds = [(0,None) for i in range(num_actions+1)]
    bounds[num_actions] = (None,None)
    res = linprog(c,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq,bounds=bounds,options={'maxiter': 1000,'tol': 1e-6})
    atk_strat = res["x"][0:num_actions]

    U_calc = U
    A_ub = np.hstack((U_calc,np.ones((num_actions,1))))
    b_ub = np.zeros(num_actions)
    A_eq = np.ones((1,num_actions+1))
    A_eq[:,num_actions] = 0
    b_eq = 1
    bounds = [(0,None) for i in range(num_actions+1)]
    bounds[num_actions] = (None,None)
    res = linprog(c,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq,bounds=bounds,options={'maxiter': 1000,'tol': 1e-6})
    def_strat = res["x"][0:num_actions]
    eqs = np.array([atk_strat,def_strat])
    #reg_eqs = lp_eqs + np.ones_like(lp_eqs)*1e-16
    #print(f'Env {env_id} Linprog EQ: {lp_eqs}')

    # g = gambit.Game.from_arrays(U,-U)
    # g.players[0].label = 'Attacker'
    # g.players[1].label = 'Defender'
    # eqs = gambit.nash.enummixed_solve(g)
    # eqs = np.array(eqs,dtype=float)
    # L = eqs.shape[0]
    # g_eqs = np.reshape(eqs,(L,2,num_actions))
    # # print(f'Env {env_id} Gambit EQ: {g_eqs}')
    U = np.array(U,dtype=float)
    # best_div = np.inf
    # for j in range(L):
    #     div = entropy(g_eqs[j].flatten(),lp_eqs.flatten())
    #     if div < best_div: 
    #         best_div = div 
    #         print(f'Env {env_id} new bes div')
    #         print(g_eqs[j].flatten())
    #         print(lp_eqs.flatten())
    # print(best_div)
    # print(f'Divergence between EQs: {entropy(g_eqs.flatten(),lp_eqs.flatten())}')
    # print(f'Env {env_id}: Done.')
    return eqs, U

def _gen_utils_eqs(fn_c_dir_nn):
    fn = fn_c_dir_nn[0]
    c = fn_c_dir_nn[1]
    save_dir = fn_c_dir_nn[2]
    num_nodes = fn_c_dir_nn[3]
    p = 2/num_nodes
    env = NetworkCascEnv(num_nodes,p,p,6,'File',filename=fn,cascade_type=c)
    eqs,U = get_nash_eqs(env)
    f_eq = save_dir + f'eq.npy'
    np.save(f_eq,eqs)
    f_util = save_dir + f'util.npy'
    np.save(f_util,U)

def perform_training_trials(args,topo_fn,target_set):
    from SL_exploration import SLExploration, RandomExploration, CDMExploration,RandomCycleExploration
    if args.exploration_type == 'Random':
        exploration = RandomExploration(target_set)
    elif args.exploration_type == 'RandomCycle':
        exploration = RandomCycleExploration(target_set)
    elif args.exploration_type == 'CDME':
        exploration = CDMExploration(target_set,eps=args.epsilon)
    else:
        print(f'Exploration type {args.exploration_type} not recognized. Exiting...')
        exit()
    net_size = args.ego_graph_size
    num_trials = args.num_trials_sub
    p = args.num_nodes_chosen/net_size
    trial_data = np.zeros((num_trials,2*args.num_nodes_chosen+1)) #last dim represents n attack nodes, n defense nodes, and attacker reward (in that order)
    trial_info = {}
    exploration.reset()
    env = NetworkCascEnv(net_size,p,p,6,'File',filename=topo_fn,cascade_type=args.cascade_type)
    for j in range(num_trials):
        action = exploration()
        exploration.update()
        _, reward, _, info = env.step(action)
        trial_data[j,:] = np.concatenate((action[0], action[1], [reward[0]])) 
        trial_info[j] = info

    # np.save(data_dir + f'{args.cascade_type}casc_trialdata.npy',trial_data)
    # with open(data_dir + f'{args.cascade_type}casc_trialinfo.pkl','wb') as file:
    #     pickle.dump(trial_info,file)
    return trial_data,trial_info

def perform_val_trials(args,topo_fn,train_actions):
    net_size = args.ego_graph_size
    p = args.num_nodes_chosen/net_size
    env = NetworkCascEnv(net_size,p,p,6,'File',filename=topo_fn,cascade_type=args.cascade_type)
    all_actions = get_combinatorial_actions(net_size,2)
    val_actions = []
    break_flag = False
    for i,a1 in enumerate(all_actions):
        for j,a2 in enumerate(all_actions):
            casc = a1 + a2
            if not np.any(np.all(train_actions == casc,axis=1)):
                val_actions.append((a1,a2))
            if len(val_actions) >= args.max_valset_trials:
                break_flag = True
                break
        if break_flag:
            break
    val_trial_data = np.zeros((len(val_actions),5))
    for j,action in enumerate(val_actions):
        _, reward, _, info = env.step(action)
        val_trial_data[j,:] = np.concatenate((action[0], action[1], [reward[0]])) 
    return val_trial_data

def subset_selection(method):
    if method == 'Random':
        pass

def create_dataset(args):
    if args.cascade_type not in ['threshold','shortPath','coupled']:
        print(f'Unrecognized Cascade Type in {casc}. Recognized Cascade Types are {Cascade_Types}. Exiting.')
        exit()
    elif args.cascade_type == 'threshold' or args.cascade_type == 'shortPath':
        gen_threshes = True
    else:
        gen_threshes = False

    #generate data for the ego graph

    if args.load_dir is not None:
        data_dir = args.load_dir
    else:
        eval_method = ''
        if args.calc_nash_ego:
            eval_method += 'NashEQ'
        elif args.max_valset_trials > 0:
            if args.calc_nash_ego: eval_method += '_'
            eval_method +=  f'valtrials_{args.exploration_type}Expl'
        save_dir = f'{args.top_dir}/{args.ego_graph_size}C2/ego_{eval_method}/'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        elif args.overwrite:
            import shutil
            shutil.rmtree(save_dir)
            os.makedirs(save_dir)
        else:
            print(f'A dataset with the specified parameters already exists in the directory {save_dir}. If you would like to generate a new one, rename this directory or use the argument --overwrite True.')
            exit()
        data_dir = save_dir
        create_random_nets(data_dir,args.ego_graph_size,gen_threshes=gen_threshes,num2gen=1)
    topo_fn = data_dir + 'net_0.edgelist'

    if args.calc_nash_ego:
        nash_eq_dir = data_dir + f'{args.cascade_type}casc_NashEQs/'
        os.makedirs(nash_eq_dir)
        #tic = time.perf_counter()
        fn = 'net_0.edgelist'
        f_args = (os.path.join(data_dir,fn),args.cascade_type,nash_eq_dir,args.ego_graph_size)
        _gen_utils_eqs(f_args)

    #generate action subsets #TODO:make a method that selects nodes for the subset propto degree
    nodes = [i for i in range(args.ego_graph_size)]
    subact_sets = []
    for i in range(args.num_subact_sets):
        subset = np.sort(random.sample(nodes,args.num_subact_targets))
        while any(np.array_equal(subset,arr) for arr in subact_sets):
            subset = np.sort(random.sample(nodes,args.num_subact_targets))
        subact_sets.append(subset)

    eval_method = f'{args.num_trials_sub}trials_{args.exploration_type}Expl'
    save_dir = data_dir + f'{args.num_subact_sets}sets_{args.num_subact_targets}targets_{eval_method}'
    save_dir += '_CfDA/' if args.cfda else '/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)   
    np.save(save_dir + 'subact_sets.npy',np.stack(subact_sets))

    fac_start = time.perf_counter()
    allsets_trialdata = np.zeros((args.num_subact_sets,args.num_trials_sub,2*args.num_nodes_chosen+1))
    allsets_trialinfo = []
    for i,sa in enumerate(subact_sets):  
        trial_data,trial_info = perform_training_trials(args,topo_fn,sa)
        allsets_trialdata[i] = trial_data
        allsets_trialinfo.append(trial_info)



    casc_keys = set()
    from utils import get_combinatorial_actions
    casc_data = allsets_trialdata.reshape((-1,allsets_trialdata.shape[-1]))
    all_actions = get_combinatorial_actions(args.ego_graph_size,2)
    for i,c in enumerate(casc_data):
        c_tup = tuple(c.astype(int))
        key = len(all_actions)*all_actions.index(c_tup[:2]) + all_actions.index(c_tup[2:4])
        if key not in casc_keys:
            casc_keys.add(key)
        else:
            casc_data = np.concatenate((casc_data[:i],casc_data[i+1:]))
    fac_end = time.perf_counter()
    fac_data_time = fac_end-fac_start
    cfac_start = time.perf_counter()
    cfac_trials = None
    cfac_data_time = None
    if args.cfda:
        #generate counterfactuals from this data
        p = args.num_nodes_chosen/args.ego_graph_size
        env = NetworkCascEnv(args.ego_graph_size,p,p,6,'File',filename=topo_fn,cascade_type=args.cascade_type)
        cfac_fns = Counterfactual_Cascade_Fns(env)
        cfac_trials, cfac_info = cfac_fns.gen_cross_subset_cfacs(allsets_trialdata,allsets_trialinfo,casc_keys,all_actions)   
        cfac_end = time.perf_counter()
        cfac_data_time = cfac_end-cfac_start

        np.save(save_dir + f'subact_{args.cascade_type}casc_CFACtrialdata.npy',cfac_trials)
        with open(save_dir + f'subact_{args.cascade_type}casc_CFACtrialinfo.pkl','wb') as file:
            pickle.dump(cfac_info,file)

    np.save(save_dir + f'subact_{args.cascade_type}casc_trialdata.npy',casc_data)
    with open(save_dir + f'subact_{args.cascade_type}casc_trialinfo.pkl','wb') as file:
        pickle.dump(allsets_trialinfo,file)

    train_data = casc_data #casc_data.reshape((-1,2*args.num_nodes_chosen+1))[:]
    if args.max_valset_trials > 0:
        if args.cfda:
            train_data = np.concatenate((train_data,cfac_trials))
        val_data = perform_val_trials(args,topo_fn,train_data[:,:-1])
        np.save(save_dir + f'subact_{args.cascade_type}casc_valdata.npy',val_data)

    num_cfac_data = len(cfac_trials) if cfac_trials is not None else None
    return len(casc_keys),num_cfac_data,fac_data_time,cfac_data_time

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='SL Dataset Creation Args')
    parser.add_argument("--ego_graph_size",default=10,type=int,help='Number of nodes in the top level graph.')
    parser.add_argument("--num_nodes_chosen",default=2,type=int,help='Number of nodes attacker and defender choose to attack/defend')
    parser.add_argument("--num_subact_targets",default=5,type=int,help='Number of nodes in the subaction spaces.')
    parser.add_argument("--num_subact_sets",default=10,type=int,help='How many subaction spaces to do trials for.')
    parser.add_argument("--cfda",default=False,type=bool,help='Whether to use CfDA for exploration.')
    parser.add_argument("--calc_nash_ego",default=False,type=bool,help='Whether to calculate the NashEQ for the ego graph.')
    parser.add_argument("--max_valset_trials",default=1000,type=int,help='Number of attack/defense trials per episode')
    parser.add_argument("--num_trials_sub",default=100,type=int,help='Number of attack/defense trials per episode')
    parser.add_argument("--cascade_type",default='threshold',type=str,help='Type of cascading model to use in cascading failure simulation.')
    parser.add_argument("--exploration_type",default='RandomCycle',type=str,help="Which exploration strategy to use to gen trials")
    parser.add_argument("--epsilon",default=0.99,type=int,help="Epsilon paramter for CDMExploration.")
    parser.add_argument("--load_dir",default=None,type=str,help='Specifies dir to load ego topology from instead of generating a new one.')
    parser.add_argument("--top_dir",default='./data/Ego/',type=str,help='Top level directory to store created datsets in')
    parser.add_argument("--overwrite",default=False,type=bool,help='Will not overwrite directory of same name unless this flag is True')
    args = parser.parse_args()

    num_data_total, num_cfac_data,fac_data_time,cfac_data_time = create_dataset(args)
    print(num_data_total)
    print(num_cfac_data)
    print(fac_data_time)
    print(cfac_data_time)







