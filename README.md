# DQN 
Modular implementation of Double-DQN, Dueling-DQN, and Priority-DQN algorithm.

# Dependencies
* Python 2.7 or 3.5
* [TensorFlow](https://www.tensorflow.org/) 1.10
* [gym](https://pypi.python.org/pypi/gym) 
* [numpy](https://pypi.python.org/pypi/numpy)
* [tqdm](ihttps://pypi.python.org/pypi/tqdm) progress-bar

# Features
- Using a neural network as the function approximator for Q-learning
- Using a target network and hard-update rule to synchronoze target network with Q-network
- Using huber loss to make small but consistent updates towards optimal Q-network 

### Bonus
- The implementation is configuration based so you can try all possible combinations of DQN networks just by changing the configuration files. 

# Usage

To train a model for PongDeterministic-v4:

	$ python test_graph_dqn.py 

To view the tensorboard

	$tensorboard --logdir .

# Results

- Tensorboard Progress Bar


