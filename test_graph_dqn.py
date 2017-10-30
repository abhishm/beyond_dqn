import tensorflow as tf
import numpy as np
import gym
from tqdm import trange
from dqn_agent import DQNAgent
from sampler import Sampler
from epsilon_greedy_policy import EpsilonGreedyPolicy
from replay_buffer import ReplayBuffer
from model import q_network
import json

from atari_wrappers import wrap_dqn, ScaledFloatFrame


config = json.load(open("configuration.json"))
env = gym.make(config["env_name"])
env = ScaledFloatFrame(wrap_dqn(env))
observation_dim = env.observation_space.shape
num_actions = env.action_space.n
n_steps = config["max_step"] // config["mini_batch_size"]

session = tf.Session()
optimizer = tf.train.AdamOptimizer(learning_rate=config["learning_rate"])
writer = tf.summary.FileWriter("summary/")

dqn_agent = DQNAgent(session,
                     optimizer,
                     q_network,
                     observation_dim,
                     num_actions,
                     config["discount"],
                     config["target_update_rate"],
                     config["target_update_frequency"],
                     config["huber_loss_threshold"],
                     writer,
                     config["summary_every"],
                     config["use_double_dqn"],
                     config["use_dueling"])

exploration_policy = EpsilonGreedyPolicy(dqn_agent,
                                         num_actions,
                                         config["epsilon"])

# Initializing ReplayBuffer
replay_buffer = ReplayBuffer(config["buffer_size"])


#Initializing Sampler
training_sampler = Sampler(exploration_policy,
                           env,
                           config["mini_batch_size"],
                           writer)


for _ in trange(config["num_itr"]):
    for t in range(n_steps):
        mini_batch = training_sampler.samples()
        replay_buffer.add_batch(mini_batch)

        if config["process_after"] <= replay_buffer.count():
            random_batch = replay_buffer.sample_batch(config["sample_size"])
            dqn_agent.update_parameters(random_batch)

        if mini_batch["dones"][-1]:
            break
