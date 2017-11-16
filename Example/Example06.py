# Deep Q-Network for simple path finding game
from __future__ import division

import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import scipy.misc
import os

from gridworld import gameEnv

env = gameEnv(partial=False, size=5)


class Qnetwork():
    def __init__(self, h_size):
        # scalarInput -> imageIn -> conv1 -> conv2 -> conv3 -> conv4 -> out(h_size)
        self.scalarInput = tf.placeholder(shape=[None, 21168], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 84, 84, 3])
        self.conv1 = slim.conv2d(inputs=self.imageIn, num_outputs=32,
                                 kernel_size=[8, 8], stride=[4, 4], padding='VALID',
                                 biases_initializer=None)
        self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=64,
                                kernel_size=[4, 4], stride=[2, 2], padding='VALID',
                                biases_initializer=None)
        self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs=64,
                                 kernel_size=[3, 3], stride=[1, 1], padding='VALID',
                                 biases_initializer=None)
        self.conv4 = slim.conv2d(inputs=self.conv3, num_outputs=h_size,
                                 kernel_size=[7, 7], stride=[1, 1], padding='VALID',
                                 biases_initializer=None)

        # advantage and value function
        self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()    #a kind of variance_scaling_initializer
        self.AW = tf.Variable(xavier_init([h_size//2, env.actions]))
        self.VW = tf.Variable(xavier_init([h_size//2, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)   # streamA x AW -> advantage
        self.Value = tf.matmul(self.streamV, self.VW)   # streamV x VW -> value

        # Q <= value + delta of advantage
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.Qout, axis=1)

        # calculate loss
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, env.actions, dtype=tf.float32) # make matrix from actions

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)     # summing up actually taken Qs
        self.td_error = tf.square(self.targetQ - self.Q)    # minimize square error
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)


class experience_buffer():
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:     # overflow
            self.buffer[0:(len(experience) + len(self.buffer))-self.buffer_size] = []   # wiping overflowed size
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])


def processState(states):
    return np.reshape(states, [21168])


def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars//2]):
        # tau * var + (1-tau) * other half
        op_holder.append(tfVars[idx+total_vars//2].assign(
            (var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder


def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)


# parameters
batch_size = 32 #How many experiences to use for each training step.
update_freq = 4 #How often to perform a training step.
y = .99 #Discount factor on the target Q-values
startE = 1 #Starting chance of random action
endE = 0.1 #Final chance of random action
annealing_steps = 10000. #How many steps of training to reduce startE to endE.
num_episodes = 10000 #How many episodes of game environment to train network with.
pre_train_steps = 10000 #How many steps of random actions before training begins.
max_epLength = 50 #The max allowed length of our episode.
load_model = False #Whether to load a saved model.
path = "./dqn" #The path to save our model to.
h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 #Rate to update target network toward primary network


# main training
tf.reset_default_graph()
# off-policy learning?
mainQN = Qnetwork(h_size)
targetQN = Qnetwork(h_size)

