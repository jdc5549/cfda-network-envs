import os
import numpy as np
import pygambit as gambit
import pickle
import random
import time
import itertools
import networkx as nx

from netcasc_gym_env import NetworkCascEnv
from cascade_cfda import Counterfactual_Cascade_Fns
from utils import create_random_nets, ncr, get_combinatorial_actions
from scipy.optimize import linprog
from scipy.stats import entropy
from tqdm import tqdm

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
    env = NetworkCascEnv(p,p,'File',6,filename=fn,cascade_type=c)
    eqs,U = get_nash_eqs(env)
    f_eq = save_dir + f'eq.npy'
    np.save(f_eq,eqs)
    f_util = save_dir + f'util.npy'
    np.save(f_util,U)

def perform_training_trials(args,topo_fn,past_keys,target_set):
    from SL_exploration import SLExploration, RandomExploration, CDMExploration,RandomCycleExploration
    if args.exploration_type == 'Random':
        exploration = RandomExploration(target_set,p=args.p)
    elif args.exploration_type == 'RandomCycle':
        exploration = RandomCycleExploration(target_set,p=args.p)
    elif args.exploration_type == 'CDME':
        exploration = CDMExploration(target_set,p=args.p,eps=args.epsilon)
    else:
        print(f'Exploration type {args.exploration_type} not recognized. Exiting...')
        exit()
    G = nx.read_gpickle(topo_fn)
    net_size = G.number_of_nodes()
    #p = 2/net_size
    num_trials = args.num_trials_sub
    trial_data = np.zeros((num_trials,2*args.p+1)) #last dim represents n attack nodes, n defense nodes, and attacker reward (in that order)
    trial_info = {}
    exploration.reset()
    env = NetworkCascEnv(args.p,args.p,'File',6,filename=topo_fn,cascade_type=args.cascade_type,degree=args.p)
    env.scm.past_results = past_keys
    for j in range(num_trials):
        action = exploration()
        exploration.update()
        _, reward, _, info = env.step(action)
        trial_data[j,:] = np.concatenate((action[0], action[1], [reward[0]])) 
        trial_info[j] = {key: value for key,value in info.items() if key != 'edges'}

    # np.save(data_dir + f'{args.cascade_type}casc_trialdata.npy',trial_data)
    # with open(data_dir + f'{args.cascade_type}casc_trialinfo.pkl','wb') as file:
    #     pickle.dump(trial_info,file)
    return trial_data,trial_info,env.scm.past_results

def perform_val_trials(args,topo_fn,train_actions,cycle=True):
    G = nx.read_gpickle(topo_fn)
    net_size = G.number_of_nodes()
    p = 2/net_size
    env = NetworkCascEnv(args.p,args.p,'File',6,filename=topo_fn,cascade_type=args.cascade_type,degree=args.p)
    all_actions = get_combinatorial_actions(net_size,args.p)
    val_actions = []
    if cycle:
        action_combinations = [(a1,a2) for a1 in all_actions for a2 in all_actions]
        shuffled_combinations = action_combinations.copy()
        random.shuffle(shuffled_combinations)
        break_flag = False
    with tqdm(total=args.max_valset_trials, desc='Validation Action Selection', unit='iteration') as pbar:
        if cycle:
            for a1,a2 in shuffled_combinations:
                casc = a1 + a2
                if not np.any(np.all(train_actions == casc, axis=1)):
                    val_actions.append((a1, a2))
                    pbar.update(1)  # Update the progress bar with each iteration
                if len(val_actions) >= args.max_valset_trials:
                    break
        else:
            max_samples = 10*args.max_valset_trials
            count = 0
            while len(val_actions) < args.max_valset_trials:
                a1 = random.sample(all_actions,k=1)[0]
                a2 = random.sample(all_actions,k=1)[0]
                casc = a1 + a2
                if not np.any(np.all(train_actions == casc, axis=1)) and (a1,a2) not in val_actions:
                    val_actions.append((a1, a2))
                    pbar.update(1)  # Update the progress bar with each iteration
                count += 1
                if count > max_samples:
                    break


    # for i,a1 in enumerate(all_actions):
    #     for j,a2 in enumerate(all_actions):
    #         casc = a1 + a2
    #         if not np.any(np.all(train_actions == casc,axis=1)):
    #             val_actions.append((a1,a2))
    #         if len(val_actions) >= args.max_valset_trials:
    #             break_flag = True
    #             break
    #     if break_flag:
    #         break
    val_trial_data = np.zeros((len(val_actions),2*args.p+1))
    with tqdm(total=len(val_actions), desc='Validation Trials', unit='iteration') as pbar:
        for j,action in enumerate(val_actions):
            _, reward, _, info = env.step(action)
            val_trial_data[j,:] = np.concatenate((action[0], action[1], [reward[0]])) 
            pbar.update(1)
    return val_trial_data

def subset_selection(method):
    if method == 'Random':
        pass

# def process_subact_set(sa,arg1):
#     topo_fn = arg1
#     trial_data, trial_info = perform_training_trials(args, topo_fn, sa)
#     return trial_data, trial_info

# def process_subact_set(args, topo_fn, past_keys, sa):
#     trial_data, trial_info = perform_training_trials(args, topo_fn, past_keys, sa)
#     return trial_data, trial_info

def update_progress(pbar):
    pbar.update()

def create_dataset(args):
    if args.cascade_type not in ['threshold','shortPath','coupled']:
        print(f'Unrecognized Cascade Type in {casc}. Recognized Cascade Types are {Cascade_Types}. Exiting.')
        exit()
    elif args.cascade_type == 'threshold' or args.cascade_type == 'shortPath':
        gen_threshes = True
    else:
        gen_threshes = False

    #generate data for the ego graph
    if args.load_graph_dir is not None:
        data_dir = args.load_graph_dir
        fn = os.path.join(data_dir,'net_0.gpickle')
        G = nx.read_gpickle(fn)
        num_nodes = G.number_of_nodes()
    else:
        num_nodes = args.ego_graph_size
        eval_method = ''
        if args.calc_nash_ego:
            eval_method += 'NashEQ'
        elif args.max_valset_trials > 0:
            if args.calc_nash_ego: eval_method += '_'
            eval_method +=  f'valtrials_{args.exploration_type}Expl'
        save_dir = f'{args.top_dir}/{num_nodes}/C{args.p}/'
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
        create_random_nets(data_dir,num_nodes,gen_threshes=gen_threshes,num2gen=1)
    topo_fn = data_dir + 'net_0.gpickle'

    if args.calc_nash_ego:
        nash_eq_dir = data_dir + f'{args.cascade_type}casc_NashEQs/'
        os.makedirs(nash_eq_dir)
        #tic = time.perf_counter()
        fn = 'net_0.gpickle'
        f_args = (os.path.join(data_dir,fn),args.cascade_type,nash_eq_dir,num_nodes)
        _gen_utils_eqs(f_args)

    #generate action subsets
    eval_method = f'{args.num_trials_sub}trials_{args.exploration_type}Expl'
    save_dir = data_dir + f'{args.num_subact_sets}sets_{args.num_subact_targets}targets_{eval_method}'
    save_dir += '_CfDA/' if args.cfda else '/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if not args.overwrite and os.path.exists(save_dir + 'subact_sets.npy'):
        subact_sets = np.load(save_dir + 'subact_sets.npy')
    else:
        nodes = [i for i in range(num_nodes)]
        subact_sets = []
        for i in tqdm(range(args.num_subact_sets),desc='Subset Generation'):
            subset = np.sort(random.sample(nodes,args.num_subact_targets))
            while any(np.array_equal(subset,arr) for arr in subact_sets):
                subset = np.sort(random.sample(nodes,args.num_subact_targets))
            subact_sets.append(subset)
        np.save(save_dir + 'subact_sets.npy',np.stack(subact_sets))

    fac_start = time.perf_counter()

    # allsets_trialdata = np.zeros((args.num_subact_sets,args.num_trials_sub,5))
    # allsets_trialinfo = [[] for i in range(args.num_subact_sets)]
    # casc_info = []
    # for i,sa in enumerate(subact_sets):  
    #     trial_data,trial_info = perform_training_trials(args,topo_fn,sa)
    #     allsets_trialdata[i] = trial_data
    #     allsets_trialinfo.append(trial_info)
    #     for key,info in trial_info.items():
    #         casc_info.append(info)
    #     fac_data_pbar.update()
    # fac_data_pbar.close()
    from utils import get_combinatorial_actions
    all_actions = get_combinatorial_actions(num_nodes,args.p)
    subact_save_name = save_dir + f'subact_{args.cascade_type}casc'
    if not args.overwrite and os.path.exists(f'{subact_save_name}_splitbyset_trialdata.npy') and os.path.exists(f'{subact_save_name}_splitbyset_trialinfo.pkl'):
        print(f'Loading Factual Data from: {subact_save_name}_trialdata.npy')
        allsets_trialdata = np.load(f'{subact_save_name}_splitbyset_trialdata.npy')
        # with open(f'{subact_save_name}_splitbyset_trialinfo.pkl','rb') as file:
        #     allsets_trialinfo = pickle.load(file)
        casc_data = np.load(f'{subact_save_name}_trialdata.npy')
        # with open(f'{subact_save_name}_trialinfo.pkl','rb') as file:
        #     casc_info = pickle.load(file)
        with open(f'{subact_save_name}_keys.pkl','rb') as file:
            casc_keys = pickle.load(file)
        fac_data_time = None
    else:
        from multiprocessing import Pool,Manager,cpu_count
        manager = Manager()
        shared_allsets_trialdata = manager.list(np.zeros((args.num_subact_sets, args.num_trials_sub, 5)))
        shared_allsets_trialinfo = manager.list([[] for _ in range(args.num_subact_sets)])
        all_pastkeys = manager.dict()
        shared_casc_info = manager.list()
        # num_chunks = cpu_count()-2
        # chunk_size = len(subset_combinations) // num_chunks
        # chunks = [subset_combinations[i:i+chunk_size] for i in range(0, len(subset_combinations), chunk_size)]
        from functools import partial
        with Pool(processes=4) as pool, tqdm(total=len(subact_sets), desc='Fac Data Progress') as pbar:
            for i, (trial_data, trial_info,trial_keys) in enumerate(pool.imap_unordered(partial(perform_training_trials, args, topo_fn,dict(all_pastkeys)), subact_sets)):
                shared_allsets_trialdata[i] = trial_data
                shared_allsets_trialinfo[i] = trial_info
                for key, info in trial_info.items():
                    try:
                        shared_casc_info.append(info)
                    except:
                        print(info)
                for key,value in trial_keys.items():
                    if key not in all_pastkeys:
                        all_pastkeys[key] = value
                pbar.set_postfix({'num_pastkeys': len(all_pastkeys.keys())})
                update_progress(pbar)

        allsets_trialdata = np.array(shared_allsets_trialdata)
        allsets_trialinfo = list(shared_allsets_trialinfo)
        np.save(save_dir + f'subact_{args.cascade_type}casc_splitbyset_trialdata.npy',allsets_trialdata)
        with open(save_dir + f'subact_{args.cascade_type}casc_splitbyset_trialinfo.pkl','wb') as file:
            pickle.dump(allsets_trialinfo,file)

        casc_info = list(shared_casc_info)  
        casc_keys = set()
        casc_data = allsets_trialdata.reshape((-1,allsets_trialdata.shape[-1]))
        duplicate_bool = []

        duplicate_bool = []  # Initialize the list outside the loop
        with tqdm(total=len(casc_data), desc='Filtering duplicates in fac data', unit='iteration') as pbar:
            for i,c in enumerate(casc_data):
                c_tup = tuple(c.astype(int))
                key = len(all_actions) * all_actions.index(c_tup[:args.p]) + all_actions.index(c_tup[args.p:-1])
                if key not in casc_keys:
                    casc_keys.add(key)
                    duplicate_bool.append(False)
                else:
                    duplicate_bool.append(True)
                pbar.update(1)  # Update the progress bar with each iteration
        # for i,c in enumerate(casc_data):
        #     c_tup = tuple(c.astype(int))
        #     key = len(all_actions)*all_actions.index((min(c_tup[:2]),max(c_tup[:2]))) + all_actions.index((min(c_tup[2:4]),max(c_tup[2:4])))
        #     if key not in casc_keys:
        #         casc_keys.add(key)
        #         duplicate_bool.append(False)
        #     else:
        #         duplicate_bool.append(True)

        casc_data = casc_data[~np.array(duplicate_bool)]
        casc_info = [casc_info[i] for i,dup in enumerate(duplicate_bool) if not dup]
        #     else:
        #         casc_data = np.concatenate((casc_data[:i],casc_data[i+1:]))
        #         casc_info = casc_info[:i] + casc_info[i+1:]
        fac_end = time.perf_counter()
        fac_data_time = fac_end-fac_start

        np.save(save_dir + f'subact_{args.cascade_type}casc_trialdata.npy',casc_data)
        with open(save_dir + f'subact_{args.cascade_type}casc_trialinfo.pkl','wb') as file:
            pickle.dump(casc_info,file)
        with open(save_dir + f'subact_{args.cascade_type}casc_keys.pkl','wb') as file:
            pickle.dump(casc_keys,file)

    cfac_trials = None
    cfac_data_time = None
    if args.cfda:
        if not args.overwrite and os.path.exists(f'{subact_save_name}_CFACtrialdata.npy') and os.path.exists(f'{subact_save_name}_CFACtrialinfo.pkl'):
            print(f'Loading Counterfactuals from: {subact_save_name}_CFACtrialdata.npy')
            cfac_trials = np.load(f'{subact_save_name}_CFACtrialdata.npy')
        else:
            #generate counterfactuals from this data
            #p = 2/num_nodes
            env = NetworkCascEnv(args.p,args.p,'File',6,filename=topo_fn,cascade_type=args.cascade_type,degree=args.p)
            cfac_fns = Counterfactual_Cascade_Fns(env)
            cfac_trials,cfac_info,cfac_data_time = cfac_fns.gen_cross_subset_cfacs(allsets_trialdata,allsets_trialinfo,casc_keys,all_actions,max_ratio=100)
            np.save(save_dir + f'subact_{args.cascade_type}casc_CFACtrialdata.npy',cfac_trials)
            with open(save_dir + f'subact_{args.cascade_type}casc_CFACtrialinfo.pkl','wb') as file:
                pickle.dump(cfac_info,file)

    if args.overwrite or not os.path.exists(f'{subact_save_name}_valdata.npy'):
        train_data = casc_data #casc_data.reshape((-1,5))[:]
        if args.max_valset_trials > 0:
            if args.cfda:
                train_data = np.concatenate((train_data,cfac_trials))
            val_data = perform_val_trials(args,topo_fn,train_data[:,:-1],cycle=False)
            np.save(save_dir + f'subact_{args.cascade_type}casc_valdata.npy',val_data)
    else:
        print(f'Loading Validation Data from: {subact_save_name}_valdata.npy')

    num_cfac_data = len(cfac_trials) if cfac_trials is not None else None
    return len(casc_keys),num_cfac_data,fac_data_time,cfac_data_time

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='SL Dataset Creation Args')
    parser.add_argument("--ego_graph_size",default=10,type=int,help='Number of nodes in the top level graph.')
    parser.add_argument("--p",default=2,type=int,help='Number of nodes chosen in an action')
    parser.add_argument("--num_subact_targets",default=5,type=int,help='Number of nodes in the subaction spaces.')
    parser.add_argument("--num_subact_sets",default=10,type=int,help='How many subaction spaces to do trials for.')
    parser.add_argument("--cfda",default=False,type=bool,help='Whether to use CfDA for exploration.')
    parser.add_argument("--calc_nash_ego",default=False,type=bool,help='Whether to calculate the NashEQ for the ego graph.')
    parser.add_argument("--max_valset_trials",default=1000,type=int,help='Number of attack/defense trials per episode')
    parser.add_argument("--num_trials_sub",default=100,type=int,help='Number of attack/defense trials per episode')
    parser.add_argument("--cascade_type",default='threshold',type=str,help='Type of cascading model to use in cascading failure simulation.')
    parser.add_argument("--exploration_type",default='RandomCycle',type=str,help="Which exploration strategy to use to gen trials")
    parser.add_argument("--epsilon",default=0.99,type=int,help="Epsilon paramter for CDMExploration.")
    parser.add_argument("--load_graph_dir",default=None,type=str,help='Specifies dir to load ego topology from instead of generating a new one.')
    parser.add_argument("--top_dir",default='./data/',type=str,help='Top level directory to store created datsets in')
    parser.add_argument("--overwrite",default=False,type=bool,help='Will not overwrite directory of same name unless this flag is True')
    args = parser.parse_args()

    num_data_total, num_cfac_data,fac_data_time,cfac_data_time = create_dataset(args)
    print(f'Num Factual Data: {num_data_total}')
    print(f'Num Cfactual Data: {num_cfac_data}')
    print(f'Time to Gen Fac Data: {fac_data_time}')
    print(f'Time to Gen CFAc Data: {cfac_data_time}')