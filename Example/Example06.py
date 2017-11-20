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
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 84, 84, 3])  # 84 x 84 pixels x rgb
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
        self.streamAC, self.streamVC = tf.split(self.conv4, 2, axis=3)
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

init = tf.global_variables_initializer()

saver = tf.train.Saver()

trainables = tf.trainable_variables()

targetOps = updateTargetGraph(trainables, tau)

myBuffer = experience_buffer()

# prob. of random action
e = startE
stepDrop = (startE-endE)/annealing_steps

# containers
jList = []  # number of stpes
rList = []  # rewards
total_steps = 0

# check path for saver
if not os.path.exists(path):
    os.makedirs(path)

# main session
with tf.Session() as sess:
    sess.run(init)
    if load_model:
        print("Loading model...")
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)

    for i in range(num_episodes):
        episodeBuffer = experience_buffer()
        s = env.reset()     # reset environment
        s = processState(s)     # make it to be flattened
        d = False   # episode is done?
        rAll = 0    #reward
        j = 0

        while j < max_epLength:
            j += 1
            if np.random.randint(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0, 4)     # randomly choose an action
            else:
                a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput:[s]})    # a_t ~ mainQN(s_t)
                a = a[0]
            s1, r, d = env.step(a)
            s1 = processState(s1)
            total_steps += 1
            episodeBuffer.add(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))

            if total_steps > pre_train_steps:
                if e > endE:    # after the pre-training phase, adjust e
                    e -= stepDrop

                if total_steps % (update_freq) == 0:
                    trainBatch = myBuffer.sample(batch_size)
                    # train networks with s1
                    Q1 = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 3])})
                    Q2, sC = sess.run([targetQN.Qout, targetQN.conv4], feed_dict={targetQN.scalarInput: np.stack(trainBatch[:, 3])})
                    end_multiplier = -(trainBatch[:, 4] - 1)    # if destination then 0, otherwise 1
                    doubleQ = Q2[range(batch_size), Q1]     # value of state in Q2
                    targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)     # reward + expected return
                    _ = sess.run(mainQN.updateModel,
                                 feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 0]),
                                            mainQN.targetQ: targetQ, mainQN.actions: trainBatch[:, 1]})
                    updateTarget(targetOps, sess)

                rAll += r
                s = s1

                if d:
                    break
        # endWhile

        myBuffer.add(episodeBuffer.buffer)
        jList.append(j)
        rList.append(rAll)

        if i % 1000 == 0:
            save_model_name = path+'/model-' + str(i) + '.ckpt'
            saver.save(sess, save_model_name)
            print("Saved Model: " + save_model_name)

        if len(rList) % 10 == 0:
            print(total_steps, np.mean(rList[-10:]), e)
    saver.save(sess, path+'/model-' + str(i) + '.ckpt')
print("precent of successful episodes: " + str(sum(rList)/num_episodes) + "%")


rMat = np.resize(np.array(rList),[len(rList)//100,100])
rMean = np.average(rMat,1)
plt.plot(rMean)