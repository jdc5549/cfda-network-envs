import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import os
import sys
import re
from time import perf_counter
import networkx as nx 

from torch.utils.data import DataLoader as DataLoader

from SL_validate import Validator
from NetCascDataset import NetCascDataset,NetCascDataset_Subact
from netcasc_gym_env import NetworkCascEnv
sys.path.append('./marl/')
from models import MLP_Critic
from tqdm import tqdm

if __name__ == '__main__':
	tic = perf_counter()
	import argparse
	parser = argparse.ArgumentParser(description='Netcasc SL Training Args')
	parser.add_argument("--ego_data_dir",default=None,type=str,help='Directory to load data from.')
	parser.add_argument("--subact_data_dir",default=None,type=str,help='Directory to load data from.')
	parser.add_argument("--log_dir",default='logs/subact_logs',type=str,help='Directory to store logs in.')
	parser.add_argument("--exp_name",default='my_exp',type=str,help='')
	parser.add_argument("--model_save_dir",default='./models/subact/',type=str,help='Directory to save model to.')
	parser.add_argument("--cascade_type",default='threshold',type=str,help='Type of cascading dynamics.')
	parser.add_argument("--gnn_model",default=None,type=str,help='What type of GNN model to use, if any.')
	parser.add_argument("--embed_size",default=16,type=int,help='Size of GNN embedded graph representation.')
	parser.add_argument("--num_epochs",default=1000,type=int,help='Number of training epochs to perform.')
	parser.add_argument("--learning_rate",default=0.001,type=float,help='Learning rate.')
	parser.add_argument("--sched_step",default=500,type=int,help='How often to reduce the learning rate for training NN model')
	parser.add_argument("--sched_gamma",default=0.1,type=float,help='How much to reduce the learning rate after shed_step steps')
	parser.add_argument("--mlp_hidden_size",default=64,type=int,help='Hidden layer size for MLP nets used for RL agent.')
	parser.add_argument("--mlp_hidden_depth",default=2,type=int,help='Hidden layer depth for MLP nets used for RL agent.')
	parser.add_argument("--dropout_rate",default=0.4,type=float,help='Rate of dropout to use during neural network training')
	parser.add_argument("--heuristic_features",default=False,type=bool,help='Whether to use heuristic topological features as input to the neural network.')
	parser.add_argument("--batch_size",default=64,type=int,help='Batch size for data loader.')
	parser.add_argument("--val_freq",default=100,type=int,help='Frequency (in epochs) at which to validate model.')
	parser.add_argument("--cfda",default=False,type=bool,help='Whether to use CfDA data during training.')
	parser.add_argument("--device",default='cpu',type=str,help='Device to perform training on.')
	parser.add_argument("--p",default=2,type=int,help='Number of nodes selected for an action')

	args = parser.parse_args()
	from torch.utils.tensorboard import SummaryWriter
	writer = SummaryWriter(os.path.join(args.log_dir, args.exp_name))

	#pattern = r'/(\d+)C2/'
	#match = re.search(pattern, args.ego_data_dir)
	#if match:
	G = nx.read_gpickle(f'{args.ego_data_dir}net_0.gpickle')
	num_nodes = G.number_of_nodes()#int(match.group(1))
	#else:
	#	print('Could not identify net size from ego file directory. Looking for {N}C{R} pattern in path name.')

	if 'cuda' in args.device:
		if torch.cuda.is_available():
			device = torch.device(args.device)
		else:
			device = torch.device('cpu')
			print('Specified Training device as GPU, but torch.cuda is unavailable. Training on CPU instead.')
	else:
		device = torch.device('cpu')

	gnn = args.gnn_model is not None
	if gnn:
		#initialize the embedding model
		dataset = NetCascDataset_Subact(args.ego_data_dir,args.subact_data_dir,args.cascade_type,p=args.p,gnn=True,topo_features=args.heuristic_features,cfda=args.cfda)
		(net_features,edge_index,_),_ = dataset.__getitem__(0)
		if args.gnn_model == 'GCN':
			from models import GCN_Critic
			q_model = GCN_Critic(args.embed_size,args.mlp_hidden_size,num_nodes)
			print(f'Dataset Size: {dataset.__len__()}')
			#print(f'Input Size: {args.embed_size+num_nodes}')
			print(f'Number of Model parameters: {sum(p.numel() for p in q_model.parameters())}')
			q_model.to(device)
		elif args.gnn_model == 'GAT':
			from models import GAT_Critic
			q_model = GAT_Critic(args.embed_size,args.mlp_hidden_size,num_nodes)
			print(f'Dataset Size: {dataset.__len__()}')
			#print(f'Input Size: {args.embed_size+num_nodes}')
			print(f'Number of Model parameters: {sum(p.numel() for p in q_model.parameters())}')	
			q_model.to(device)
		else:
			print(f'GNN model {args.gnn_model} not recognized.')
			exit()
		num_targets = dataset.subact_sets.shape[1]
		data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
	else:
		dataset = NetCascDataset_Subact(args.ego_data_dir,args.subact_data_dir,args.cascade_type,p=args.p,gnn=False,topo_features=args.heuristic_features,cfda=args.cfda)
		#Embedding = HeuristicActionEmbedding(dataset.topology,dataset.thresholds)
		#embedded_act = Embedding.embed_action(torch.tensor([[0,1]]))
		#act_embed_size = embedded_act.shape[1]
		#(feat_topo,_),(_,_) = dataset.__getitem__(0)
		#obs_embed_size = feat_topo.shape[0]

		q_model = MLP_Critic(args.embed_size,args.mlp_hidden_size,num_nodes,num_node_features=dataset.net_features.shape[0],p=args.p,num_mlp_layers=args.mlp_hidden_depth,dropout_rate=args.dropout_rate)
		print(f'Training Dataset Size: {dataset.__len__()}')
		#print(f'Input Size: {num_nodes+2}')
		print(f'Number of Model parameters: {sum(p.numel() for p in q_model.parameters())}')
		q_model.to(device)
		num_targets = dataset.subact_sets.shape[1]
		data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
	if not os.path.isdir(args.log_dir):
		os.mkdir(args.log_dir)
	hparams = {"training_epochs": args.num_epochs, "learning_rate": args.learning_rate, "sched_step": args.sched_step, "sched_gamma":args.sched_gamma,
				"cascade_type": args.cascade_type, "batch_size": args.batch_size,"mlp_hidden_size": args.mlp_hidden_size, "mlp_depth": args.mlp_hidden_depth,
				"net_size": num_nodes,"num_targets":num_targets, "embed_size": args.embed_size,"dropout_rate": args.dropout_rate}
	writer.add_hparams(hparams,{'dummy/dummy': 0},run_name=None)#'hparams')



	print(q_model)
	model_save_dir = args.model_save_dir + f'{num_nodes}_{num_targets}C2/{args.exp_name}/'
	if not os.path.isdir(model_save_dir):
		os.makedirs(model_save_dir)
	#init Validator
	#p = 2/num_nodes
	nash_eqs_dir = args.ego_data_dir + f'{args.cascade_type}casc_NashEQs/'
	val_data_path = args.subact_data_dir + f'subact_{args.cascade_type}casc_valdata.npy'
	if os.path.isfile(val_data_path):		
		val_dataset = NetCascDataset_Subact(args.ego_data_dir,args.subact_data_dir,args.cascade_type,p=args.p,gnn=gnn,val=True,topo_features=args.heuristic_features,cfda=False)
		print(f'Validation Dataset Size: {val_dataset.__len__()}')
	
	else: 
		val_dataset = None
	test_env = [NetworkCascEnv(args.p,args.p,'File',cascade_type=args.cascade_type,degree=args.p,
				filename = args.ego_data_dir + 'net_0.gpickle')]
	V = Validator(test_env,p=args.p,subact_sets=dataset.subact_sets,dataset=val_dataset,nash_eqs_dir=nash_eqs_dir,device=device,gnn=gnn)
	#criterion = nn.SmoothL1Loss()
	criterion = nn.BCELoss()

	optimizer = optim.Adam(q_model.parameters(), lr=args.learning_rate,weight_decay=1e-5)  # Replace with your own optimizer and learning rate
	scheduler = lr_scheduler.StepLR(optimizer, step_size=args.sched_step, gamma=args.sched_gamma)

	toc = perf_counter()
	print(f'Time until training start: {toc-tic} seconds')
	tic = perf_counter()
	# Create a progress bar for epochs
	epoch_progress_bar = tqdm(total=args.num_epochs, desc='Training Progress')
	best_div = 1e6
	best_util_err = 1e6
	best_val_err = 1e6
	nash_eq_div = 1e6
	util_err = 1e6
	val_err = 1e6
	for epoch in range(args.num_epochs):
		epoch_loss = 0
		pred_err = 0
		# Iterate over the data loader
		for i, data in enumerate(data_loader):
			if args.gnn_model is not None:
				(node_features,edge_index,actions),(reward,multi_hot_failures) = data
				edge_index = edge_index.to(device)
				#feat_topo = embed_model(net_features,edge_index)
			else:
				(node_features,actions), (reward,multi_hot_failures) = data
			B = reward.shape[0]

			node_features = node_features.to(device)
			actions = actions.to(device)
			reward = reward.to(device)
			multi_hot_failures = multi_hot_failures.to(device)

			# Zero the gradients
			optimizer.zero_grad()

			# Forward pass
			if args.gnn_model is not None:
				pred = q_model(actions,node_features,edge_index)
			else:
				pred = q_model(actions,node_features)
			# if problem_idx != -1:
			# 	print('training at problem index: ', [pred_reward[problem_idx].item(),reward[problem_idx].item()])
			# Calculate loss
			loss = criterion(pred, multi_hot_failures)

			# Backward pass
			loss.backward()
			# Update weights
			optimizer.step()

			multi_hot_pred = torch.zeros_like(pred)
			multi_hot_pred[pred > 0.1] = 1
			pred_reward = torch.mean(multi_hot_pred,dim=1)
			# Accumulate loss/error statistics
			epoch_loss += loss.item()*B/dataset.__len__()
			pred_err += np.sum(np.abs((pred_reward-reward).detach().cpu().numpy()))/dataset.__len__()
		writer.add_scalar("Training/training_loss",epoch_loss,epoch)
		writer.add_scalar("Training/training_error",pred_err,epoch)

		#Perform Validation
		if epoch % args.val_freq == args.val_freq-1 or epoch == args.num_epochs-1:
			q_model.eval()
			val_err,util_err,nash_eq_div = V.validate(q_model,device=device)
			if nash_eq_div is not None:
				writer.add_scalar("Validation/nash_eq_div",nash_eq_div,epoch)
				writer.add_scalar("Validation/util_err",util_err,epoch)
				if nash_eq_div < best_div:
					#print(f'Epoch {epoch}: Nash EQ of {nash_eq_div} is new best. Saving model.')
					model_save_path = os.path.join(model_save_dir,f'best_div.pt')
					if os.path.exists(model_save_path):
						os.remove(model_save_path)
					torch.save(q_model.state_dict(),model_save_path)
					best_div = nash_eq_div
				if util_err < best_util_err:
					#print(f'Epoch {epoch}: Util err of {util_err} is new best. Saving model.')
					model_save_path = os.path.join(model_save_dir,f'best_util_err.pt')
					if os.path.exists(model_save_path):
						os.remove(model_save_path)
					torch.save(q_model.state_dict(),model_save_path)
					best_util_err = util_err
			if val_err is not None:
				writer.add_scalar("Validation/validation_err",val_err,epoch)
				if val_err < best_val_err:
					#print(f'Epoch {epoch}: Util err of {util_err} is new best. Saving model.')
					model_save_path = os.path.join(model_save_dir,f'best_val_err.pt')
					if os.path.exists(model_save_path):
						os.remove(model_save_path)
					torch.save(q_model.state_dict(),model_save_path)	
					best_val_err = val_err	
				q_model.train()

		if nash_eq_div is not None:
			epoch_progress_bar.set_postfix({'train_loss': epoch_loss,'train_err': pred_err,'val_err': val_err,'util_err': util_err})
		else:
			epoch_progress_bar.set_postfix({'train_loss': epoch_loss,'train_err': pred_err,'val_err': val_err})			
		epoch_progress_bar.update()

		scheduler.step()
		writer.add_scalar("Learning_Parameters/learning_rate",optimizer.param_groups[0]['lr'],epoch)

	toc = perf_counter()
	epoch_progress_bar.close()
	print(f'Training completeed in {toc-tic} seconds.')
	model_save_path = os.path.join(model_save_dir,f'final.pt')
	torch.save(q_model.state_dict(),model_save_path)
	print(f'Final Model with util error of {util_err} and NashEQ div of {nash_eq_div} saved to {model_save_path}')
	if nash_eq_div is not None:
		model_save_path = os.path.join(model_save_dir,f'best_util_err.pt')
		print(f'Model with best Utility Error of {best_util_err} saved to {model_save_path}')
		model_save_path = os.path.join(model_save_dir,f'best_div.pt')
		print(f'Model with best NashEQ Div of {best_div} saved to {model_save_path}')
	if val_err is not None:
		model_save_path = os.path.join(model_save_dir,f'best_val_err.pt')
		print(f'Model with best Validation Error of {best_val_err} saved to {model_save_path}')



