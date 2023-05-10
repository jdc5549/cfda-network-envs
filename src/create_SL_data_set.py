import os
import numpy as np

from netcasc_gym_env import NetworkCascEnv
from SL_exploration import SLExploration, RandomExploration, CDMExploration
from utils import create_random_nets,ncr,get_combinatorial_actions

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='SL Dataset Creation Args')
    parser.add_argument("--cfa",default=False,type=bool,help='Whether to use CFA for training.')
    parser.add_argument("--net_size",default=10,type=int,help='Number of nodes in the networks to generate.')
    parser.add_argument("--num_nodes_chosen",default=2,type=int,help='Number of nodes attacker and defender choose to attack/defend')
    parser.add_argument("--num_topologies",default=100,type=int,help='Number of Topologies to generate')
    parser.add_argument("--num_trials",default=10,type=int,help='Number of attack/defense trials per episode')
    parser.add_argument("--cascade_type",default='threshold',type=str,help='Type of cascading model to use in cascading failure simulation.')
    parser.add_argument("--exploration_type",default='Random',type=str,help="Which exploration strategy to use to gen trials")
    parser.add_argument("--epsilon",default=0.99,type=int,help="Epsilon paramter for CDMExploration.")
    parser.add_argument("--cfa",default=False,type=bool,help='Whether to use CFA for training.')
    args = parser.parse_args()

    #Create the Training Data
    if args.cascade_type == 'threshold' or args.cascade_type == 'shortestPath':
        gen_threshes = True
    else:
        gen_threshes = False

    if args.cfa:
        save_dir = f'./data/{args.net_size}C2/training_data/CfDA_{args.num_topologies}topo_{args.num_trials}trials_{args.cascade_type}casc/'
    else:
        save_dir = f'./data/{args.net_size}C2/training_data/{args.num_topologies}topo_{args.num_trials}trials_{args.cascade_type}casc/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir + 'topologies/')

    comb_acts = get_combinatorial_actions(args.net_size,args.num_nodes_chosen)
    if args.exploration_type == 'Random':
        exploration = RandomExploration(comb_acts)
    elif args.exploration_type == 'CDME':
        exploration = CDMExploration(comb_acts)
    else:
        print(f'Exploration type {args.exploration_type} not recognized. Exiting...')
        exit()
    create_random_nets(save_dir + 'topologies/',args.net_size,gen_threshes=gen_threshes,num2gen=args.num_topologies)
    trial_data = np.zeros([args.num_topologies,args.num_trials,5]) #last dim represents 2 attack nodes, 2 defense nodes, and attacker reward (in that order)
    p = 2/args.net_size
    for i in range(args.num_topologies):
        exploration.reset()
        env = NetworkCascEnv(args.net_size,p,p,6,'File',filename=save_dir + f'topologies/net_{i}.edgelist',cascade_type=args.cascade_type)
        for j in range(args.num_trials):
            action = exploration()
            exploration.update()
            _, reward, _, _ = env.step(action)
            trial_data[i,j,:] = np.concatenate((action[0], action[1], [reward[0]]))
    np.save(save_dir + 'trial_data.npy',trial_data)







