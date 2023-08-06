## Counterfactual Network Environments

This repository contains network environments that model the security scenario of an attack and defender each simultaneously choosing two nodes to preemptively attack/defend in a graph, where the attacker attempts to maximize the cascading failure impact and the defender attempts to miminimize impact. This repository supports two models of casading failure, namely a threshold-based model and a shortest-path-based model. In order to overcome the combinatorial action space that grows exponentially with the size of the network, the action space is broken down into smaller action subspaces with a limited number of nodes that can be targeted in each. These subspaces are not exhaustive, and so a deep learning approach is used to train on a dataset of security scenarios in these subaction spaces, and then generalize to scenarios not represented in the subspaces. Finally, this repository also implements methods for counterfactual generation, which complements the deep learning approach by producing additional training data with higher efficiency by combining aspects of cascades in the factual dataset.

## Data Generation
The following command is used to create a training dataset for the neural network model.

```python src/create_subact_dataset.py --ego_graph_size <size of main graph> --num_subact_targets <how many targets available in a subspace> --num_subact_sets <how many subspaces to create> --num_trials_sub <how many trials to perform in each subspace> --cfda <Whether to create counterfactual data> --calc_nash_ego <whether to calculate NashEQ (only for networks <50 nodes)> --max_valset_trials <size of validation dataset> --cascade_type <cascading failure model to use> --load_dir <to reuse a network topology specify path to file>```

As an example, the following command will create a 100-node graph with 500 subaction spaces that are restricted to 5 nodes to target each. 100 trials of the security game will be played out per subaction space and threshold-based cascading will be used to calculate cascading failure results.

```python src/create_subact_dataset.py --ego_graph_size 100 --num_subact_targets 5 --num_subact_sets 500 --num_trials_sub 100 --cascade_type threshold```

Data will automatically be saved to the directory ```data/Ego/<ego_graph_size>C2/ego_<validation_method>/<num_subaction_sets>sets_<num_subaction_targets>targets_<num_subact_targets>trials_<exploration_type>```

## Running Deep Learning Experiments
The following command is used to train a neural network model on a created dataset

```python src/subact_train.py --ego_data_dir <dir of network topology> --subact_data_dir <dir of datset> --exp_name <experiment name> --mlp_hidden_depth <number of hidden layers in NN> --num_epochs <number of training epochs> --learning_rate <learning rate for training> --sched_step <how often to decrease learning rate during training> --sched_gamma <by what factor to decrease learning rate during training> --val_freq <how often to validate the network using the eval method> --batch_size <data batch size for training> --mlp_hidden_size <size of NN hidden layers> --embed_size <size of action embeddding> --cascade_type <cascading failure model to use>```

```python src/subact_train.py --ego_data_dir data/Ego/25C2/ego_NashEQ/ --subact_data_dir data/Ego/25C2/ego_NashEQ/500sets_5targets_100trials_RandomCycleExpl/ --exp_name '100_5C2' --mlp_hidden_depth 3 --num_epochs 2000 --sched_step 1500 --val_freq 50 --batch_size 128 --mlp_hidden_size 64 --embed_size 64 --cascade_type 'threshold'```

