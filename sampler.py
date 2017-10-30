import numpy as np
import tensorflow as tf

class Sampler(object):
    def __init__(self,
                 policy,
                 env,
                 mini_batch_size,
                 summary_writer=None):
        self.policy = policy
        self.env = env
        self.state = self.env.reset()
        self.mini_batch_size = mini_batch_size
        self.total_rewards = 0
        self.summary_writer = summary_writer


    def flush_summary(self, value, tag="reward"):
        global_step = self.policy.dqn_agent.session.run(self.policy.dqn_agent.global_step)
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        self.summary_writer.add_summary(summary, global_step)
        self.summary_writer.flush()


    def collect_mini_batch(self):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for t in range(self.mini_batch_size):
            action = self.policy.sampleAction(self.state[np.newaxis, :])
            next_state, reward, done, info = self.env.step(action)

            states.append(self.state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            self.state = next_state
            self.total_rewards += reward

            if done:
                self.state = self.env.reset()
                self.flush_summary(self.total_rewards)
                self.total_rewards = 0
                break

        return dict(states = states,
                    actions = actions,
                    rewards = rewards,
                    next_states = next_states,
                    dones = dones)


    def samples(self):
        return self.collect_mini_batch()
