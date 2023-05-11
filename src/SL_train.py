import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import os
import sys

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter

from SL_validate import Validator
from NetCascDataset import NetCascDataset
from network_gym_env import NetworkCascEnv
sys.path.append('./marl/')
from marl.model.nn.mlpnet import MultiCriticMlp
from tqdm import tqdm

if __name__ == '__main__':	
	import argparse
	parser = argparse.ArgumentParser(description='Netcasc SL Training Args')
	parser.add_argument("--train_data_dir",default=None,type=str,help='Directory to load data from.')
	parser.add_argument("--val_data_dir",default=None,type=str,help='Directory to load data from.')
	parser.add_argument("--log_dir",default='SL_logs',type=str,help='Directory to store logs in.')
	parser.add_argument("--exp_name",default='my_exp',type=str,help='')
	parser.add_argument("--model_save_dir",default='./models/SL/',type=str,help='Directory to save model to.')
	parser.add_argument("--cascade_type",default='threshold',type=str,help='Type of cascading dynamics.')
	parser.add_argument("--use_gnn",default=False,type=bool,help='Whether or not a GNN is used for feature extraction.')
	parser.add_argument("--num_epochs",default=1000,type=int,help='Number of training epochs to perform.')
	parser.add_argument("--learning_rate",default=0.001,type=float,help='Reinforcement Learning rate.')
	parser.add_argument("--sched_step",default=500,type=int,help='How often to reduce the learning rate for training NN model')
	parser.add_argument("--sched_gamma",default=0.1,type=float,help='How much to reduce the learning rate after shed_step steps')
	parser.add_argument("--mlp_hidden_size",default=64,type=int,help='Hidden layer size for MLP nets used for RL agent.')
	parser.add_argument("--batch_size",default=64,type=int,help='Batch size for data loader.')
	parser.add_argument("--val_freq",default=5,type=int,help='Frequency (in epochs) at which to validate model.')

	args = parser.parse_args()


	dataset = NetCascDataset(args.train_data_dir,args.cascade_type)
	data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

	if args.use_gnn:
		pass
	else:
		(feat_topo,_), _ = dataset.__getitem__(0)
		num_nodes, embed_size = feat_topo.shape

	if not os.path.isdir(args.log_dir):
		os.mkdir(args.log_dir)
	hparams = {"training_epochs": args.num_epochs, "learning_rate": args.learning_rate, "sched_step": args.sched_step, "sched_gamma":args.sched_gamma,
				"cascade_type": args.cascade_type, "batch_size": args.batch_size,"mlp_hidden_size": args.mlp_hidden_size, 
				"net_size": num_nodes, "embed_size": embed_size}
	writer = SummaryWriter(os.path.join(args.log_dir, args.exp_name))
	writer.add_hparams(hparams,{'dummy/dummy': 0},run_name=None)#'hparams')

	q_model = MultiCriticMlp(embed_size,embed_size*2,embed_size*2,hidden_size=args.mlp_hidden_size)  # Replace with your own model architecture
	model_save_dir = args.model_save_dir + f'{num_nodes}C2/{args.exp_name}/'
	if not os.path.isdir(model_save_dir):
		os.makedirs(model_save_dir)

	#init Validator
	p = 2/num_nodes
	topology_dir = args.val_data_dir + 'topologies/'
	nash_eqs_dir = args.val_data_dir + f'{args.cascade_type}casc_NashEQs/'
	test_envs = [NetworkCascEnv(num_nodes,p,p,'File',cascade_type=args.cascade_type,
				filename = os.path.join(topology_dir,f)) for f in os.listdir(topology_dir) if 'thresh' not in f]
	V = Validator(test_envs,nash_eqs_dir=nash_eqs_dir)

	criterion = nn.SmoothL1Loss()  # Replace with your own loss function

	optimizer = optim.Adam(q_model.parameters(), lr=args.learning_rate)  # Replace with your own optimizer and learning rate
	scheduler = lr_scheduler.StepLR(optimizer, step_size=args.sched_step, gamma=args.sched_gamma)

	# Create a progress bar for epochs
	epoch_progress_bar = tqdm(total=args.num_epochs, desc='Training Progress')
	best_div = 1e6
	best_err = 1e6
	for epoch in range(args.num_epochs):
		epoch_loss = 0.0
		# Iterate over the data loader
		for i, data in enumerate(data_loader):
			if args.use_gnn:
				pass
			else:
				(feat_topo,actions), reward = data
				B = feat_topo.shape[0]
				atk_acts = actions[:,:2]
				def_acts = actions[:,2:]
			# mod_acts = atk_acts.unsqueeze(-1).expand(-1, -1, embed_size)
			# print(mod_acts)
			# print(mod_acts.dtype)
			#select rows from featurized topology corresponding to nodes attacked
			feat_atk = torch.gather(feat_topo, 2, atk_acts.unsqueeze(-1).expand(-1, -1, embed_size).type(torch.int64))

			#flatten into 1 dimension (not including batch dim)
			feat_atk = feat_atk.view(B,-1)

			feat_def = torch.gather(feat_topo, 2, def_acts.unsqueeze(-1).expand(-1, -1, embed_size).type(torch.int64))
			feat_def = feat_def.view(B,-1)

			feat_topo_mean = torch.mean(feat_topo,dim=1)
			# Zero the gradients
			optimizer.zero_grad()

			# Forward pass
			pred_reward = q_model(feat_topo_mean,feat_atk,feat_def).squeeze()

			# # Calculate loss
			loss = criterion(pred_reward, reward)

			# Backward pass
			loss.backward()

			# Update weights
			optimizer.step()

			# Accumulate loss
			epoch_loss = (epoch_loss*i + loss.item()/len(data))/(i+1)
		
		writer.add_scalar("Training/training_loss",epoch_loss,epoch)

		#Perform Validation
		nash_eq_div,util_err = (0,0)
		# if epoch % args.val_freq == 0 or epoch == args.num_epochs-1:
		# 	q_model.eval()
		# 	nash_eq_div,util_err = V.validate(q_model)
		# 	writer.add_scalar("Validation/nash_eq_div",nash_eq_div,epoch)
		# 	writer.add_scalar("Validation/util_err",util_err,epoch)

		# 	if nash_eq_div < best_div:
		# 		#print(f'Epoch {epoch}: Nash EQ of {nash_eq_div} is new best. Saving model.')
		# 		model_save_path = os.path.join(model_save_dir,f'best_div.pt')
		# 		torch.save(q_model.state_dict(),model_save_path)
		# 		best_div = nash_eq_div
		# 	if util_err < best_err:
		# 		#print(f'Epoch {epoch}: Util err of {util_err} is new best. Saving model.')
		# 		model_save_path = os.path.join(model_save_dir,f'best_err.pt')
		# 		torch.save(q_model.state_dict(),model_save_path)	
		# 	q_model.train()

		epoch_progress_bar.set_postfix({'train_loss': epoch_loss,'util_err': util_err,'nash_eq_div': nash_eq_div})
		epoch_progress_bar.update()

		scheduler.step()
		writer.add_scalar("Learning_Parameters/learning_rate",optimizer.param_groups[0]['lr'],epoch)

	epoch_progress_bar.close()
	print('Training complete!')
	model_save_path = os.path.join(model_save_dir,f'final.pt')
	torch.save(q_model.state_dict(),model_save_path)
	print(f'Model saved to {model_save_path}')
