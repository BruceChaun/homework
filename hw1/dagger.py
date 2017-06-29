#!/usr/bin/env python

"""
Imitation learning using DAgger
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy


class Config(object):

    obs_dim = 376
    action_dim = 17
    lr = 0.001
    max_epoch = 100
    batch_size = 32


class Data(object):

    def __init__(self, data):
        self.obs, self.action = data

    def union(self, data):
        self.obs = np.concatenate((self.obs, data[0]), axis=0)
        self.action = np.concatenate((self.action, data[1]), axis=0)

    def batchify(self, batch_size):
        for i in range(len(self.obs)//batch_size):
            idx = i * batch_size
            yield self.obs[idx:idx+batch_size], self.action[idx:idx+batch_size]


class NN(object):

    def __init__(self, config):
        self.config = config
        self.init_weight()

        self.add_placeholder()
        self.y_hat = self.forward(self.obs)
        self.loss = self.compute_loss(self.y_hat)
        self.opt = self.add_opt(self.loss)

    def init_weight(self):
        hid_size = 100
        self.weights = {
                "w_fc1": tf.Variable(
                    tf.truncated_normal(
                        [self.config.obs_dim, hid_size], stddev=0.01
                        )),
                "w_fc2": tf.Variable(
                    tf.truncated_normal(
                        [hid_size, self.config.action_dim], stddev=0.01
                        ))
                }
        self.biases = {
                "b_fc1": tf.Variable(tf.truncated_normal(
                    [hid_size], stddev=0.01)),
                "b_fc2": tf.Variable(tf.truncated_normal(
                    [self.config.action_dim], stddev=0.01))
                }

    def add_placeholder(self):
        self.obs = tf.placeholder(
                tf.float32, [None, self.config.obs_dim], "obs_placeholder")
        self.action = tf.placeholder(
                tf.float32, [None, self.config.action_dim], "action_placeholder")

    def forward(self, x):
        hid = tf.matmul(x, self.weights["w_fc1"]) + self.biases["b_fc1"]
        hid = tf.nn.relu(hid)
        y_hat = tf.matmul(hid, self.weights["w_fc2"]) + self.biases["b_fc2"]
        return y_hat

    def compute_loss(self, y_hat):
        loss = tf.reduce_mean(
                tf.squared_difference(y_hat, self.action))
        return loss

    def add_opt(self, loss):
        opt = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        return opt

    def run(self, sess, obs):
        action = sess.run(self.y_hat, {self.obs : obs})
        return action


def get_human_data(policy_fn, env, render=False):
    with tf.Session():
        tf_util.initialize()
        max_steps = env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        obs = env.reset()
        totalr = 0.
        steps = 0
        while steps < max_steps:
            action = policy_fn(obs[None,:])
            observations.append(obs)
            actions.append(action[0])
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if render:
                env.render()

        returns.append(totalr)

        return np.array(observations), np.array(actions)


def create_data(sess, model, policy_fn, env, 
        num_rollouts=20, max_steps=1000, render=False):
    observations = []
    for i in range(num_rollouts):
        obs = env.reset()
        steps = 0
        while steps < max_steps:
            action = sess.run(model.y_hat, {model.obs : obs[None,:]})
            observations.append(obs)
            obs, r, done, _ = env.step(action)
            steps += 1
            if render:
                env.render()

    return observations


def ask_expert(policy_fn, observations):
    with tf.Session() as sess:
        tf_util.initialize()
        actions = []

        for obs in observations:
            action = policy_fn(obs[None,:])
            actions.append(action[0])

        return np.array(actions)


def dagger(sess, expert_policy_file, envname, config):

    policy_fn = load_policy.load_policy(expert_policy_file)
    env = gym.make(envname)

    data = get_human_data(policy_fn, env)
    dataset = Data(data)

    nn = NN(config)
    sess.run(tf.global_variables_initializer())

    for epoch in range(config.max_epoch):
        total_loss = 0.
        n = 0

        # train policy from human data
        for step, (obs, action) in enumerate(
                dataset.batchify(config.batch_size)):
            fetch = [nn.loss, nn.opt]
            feed = {nn.obs : obs, nn.action : action}
            loss, _ = sess.run(fetch, feed)
            total_loss += loss
            n += 1

        print("Epoch {}\ttotal loss = {:2.8f}".format(
            epoch, total_loss / n / config.batch_size))

        # run policy to get dataset
        new_obs = create_data(sess, nn, policy_fn, env, max_steps=50)

        # ask expert to label
        new_action = ask_expert(policy_fn, new_obs)

        # aggregate
        dataset.union([new_obs, new_action])

    return nn


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument("--checkpoint", type=str, default="saver/model")
    args = parser.parse_args()

    config = Config()

    with tf.Session() as sess:
        #tf_util.initialize()
        nn = NN(config)
        sess.run(tf.global_variables_initializer())
        
        if args.checkpoint:
            saver = tf.train.Saver()
            saver.restore(sess, args.checkpoint)

        nn = dagger(sess, args.expert_policy_file, args.envname, config)
        saver = tf.train.Saver()
        saver.save(sess, args.checkpoint)

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            totalr = 0.
            steps = 0
            while steps < max_steps:
                action = nn.run(sess, obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))

            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}


if __name__ == '__main__':
    main()
