# Example of Deep Recurrent Q-Network for partially observable environment
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import os
import csv
import itertools
import tensorflow.contrib.slim as slim

from helper import *
from gridworld import gameEnv

env = gameEnv(partial=True, size=9)

# class for Q-network
class Qnetwork():
    def __init__(self, h_size, rnn_cell, myScope):
        self.scalarInput = tf.placeholder(shape=[None, 21168], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 84, 84, 3])
        self.conv1 = slim.conv2d(inputs=self.imageIn, num_outputs=32,
                                 kernel_size=[8, 8], stride=[4, 4], padding='VALID',
                                 biases_initializer=None, scope=myScope+'_conv1')
        self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=64,
                                 kernel_size=[4, 4], stride=[2, 2], padding='VALID',
                                 biases_initializer=None, scope=myScope + '_conv2')
        self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs=64,
                                 kernel_size=[3, 3], stride=[1, 1], padding='VALID',
                                 biases_initializer=None, scope=myScope + '_conv3')
        self.conv4 = slim.conv2d(inputs=self.conv3, num_outputs=h_size,
                                 kernel_size=[7, 7], stride=[1, 1], padding='VALID',
                                 biases_initializer=None, scope=myScope + '_conv4')

        self.trainLength = tf.placeholder(dtype=tf.int32)

        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])
        self.convFlat = tf.reshape(slim.flatten(self.conv4), [self.batch_size, self.trainLength, h_size])
        self.state_in = rnn_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.rnn, self.rnn_state = tf.nn.dynamic_rnn(
            inputs=self.convFlat, cell=rnn_cell, dtype=tf.float32, initial_state=self.state_in, scope=myScope+'_rnn')
        self.rnn = tf.reshape(self.rnn, shape=[-1, h_size])
        self.streamA, self.streamV = tf.split(self.rnn, 2, 1)
        self.AW = tf.Variable(tf.random_normal([h_size//2, 4]))
        self.VW = tf.Variable(tf.random_normal([h_size//2, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        self.salience = tf.gradients(self.Advantage, self.imageIn)  # d Advantage/ d imageIn
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.Qout, 1)  # Q-learning: choose next action using argmax (off policy)

        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, 4, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)     # retrieve related nodes
        self.td_error = tf.square(self.targetQ - self.Q)    # element-wise x^2

        self.maskA = tf.zeros([self.batch_size, self.trainLength//2])
        self.maskB = tf.ones([self.batch_size, self.trainLength// 2])
        self.mask = tf.concat([self.maskA, self.maskB], 1)     # only takes recent half
        self.mask = tf.reshape(self.mask, [-1])
        self.loss = tf.reduce_mean(self.td_error * self.mask)

        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)


class experience_buffer():
    def __init__(self, buffer_size=1000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:    # because we will add at least one data
            self.buffer[0:(1+len(self.buffer))-self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self, batch_size, trace_length):
        sampled_episodes = random.sample(self.buffer, batch_size)
        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode)+1-trace_length)   # randomly select a trace in an episode
            sampledTraces.append(episode[point:point+trace_length])
        sampledTraces = np.array(sampledTraces)     # list to narray
        return np.reshape(sampledTraces, [batch_size * trace_length, 5])


# PARAMETERS: Setting the training parameters
batch_size = 4      # How many experience traces to use for each training step.
trace_length = 8    # How long each experience trace will be when training
update_freq = 5     # How often to perform a training step.
y = .99             # Discount factor on the target Q-values
startE = 1          # Starting chance of random action
endE = 0.1          # Final chance of random action
anneling_steps = 10000  # How many steps of training to reduce startE to endE.
num_episodes = 10000    # How many episodes of game environment to train network with.
pre_train_steps = 10000     # How many steps of random actions before training begins.
load_model = False      # W hether to load a saved model.
path = "./drqn_ex7"     # The path to save our model to.
h_size = 512            # The size of the final convolutional layer before splitting it into Advantage and Value streams.
max_epLength = 50   # The max allowed length of our episode.
time_per_step = 1   # Length of each step used in gif creation
summaryLength = 100     # Number of epidoes to periodically save for analysis
tau = 0.001

# main training
tf.reset_default_graph()
cell = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
cellT = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
mainQN = Qnetwork(h_size, cell, 'main')
targetQN = Qnetwork(h_size, cellT, 'target')

init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=5)
trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables, tau)
myBuffer = experience_buffer()

e = startE      # epsilong
stepDrop = (startE - endE)/anneling_steps

jList = []
rList = []
total_steps = 0

if not os.path.exists(path):
    os.makedirs(path)

with open('./Center/log.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(['Episode', 'Length', 'Reward', 'IMG', 'LOG', 'SAL'])

with tf.Session() as sess:
    if load_model:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt is not None:    # if it can be loaded
            saver.restore(sess, ckpt.model_checkpoint_path)

    print("Start training")
    sess.run(init)

    updateTarget(targetOps, sess) # Set the target network to be equal to the primary network.
    for i in range(num_episodes):
        episodeBuffer = []

        sP = env.reset()
        s = processState(sP)
        d = False
        rAll = 0
        j = 0
        state = (np.zeros([1, h_size]), np.zeros([1, h_size]))  # recurrent layer's hidden states

        while j < max_epLength:
            j += 1

            # if epsilon-greedy or it is in early phase for training
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                state1 = sess.run(mainQN.rnn_state,
                                  feed_dict={mainQN.scalarInput: [s/255.0], mainQN.trainLength: 1,
                                             mainQN.state_in: state, mainQN.batch_size: 1})
                a = np.random.randint(0, 4)
            else:
                a, state1 = sess.run([mainQN.predict, mainQN.rnn_state],
                                     feed_dict={mainQN.scalarInput: [s/255.0], mainQN.trainLength: 1,
                                                mainQN.state_in: state, mainQN.batch_size: 1})
                a = a[0]
            s1P, r, d = env.step(a)     # state, reward, is done
            s1 = processState(s1P)      # flattening
            total_steps += 1
            episodeBuffer.append(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))    # form as a narray [1,5] dim

            if total_steps > pre_train_steps:
                if e > endE:
                    e -= stepDrop   # update epsilon

                if total_steps % (update_freq) == 0:    # if update phase
                    updateTarget(targetOps, sess)
                    state_train = (np.zeros([batch_size, h_size]), np.zeros([batch_size, h_size]))

                    trainBatch = myBuffer.sample(batch_size, trace_length)  # get batch for training

                    Q1 = sess.run(mainQN.predict, feed_dict={
                        mainQN.scalarInput: np.vstack(trainBatch[:, 3]/255.0),
                        mainQN.state_in: state_train,   # initial state of rnn
                        mainQN.trainLength: trace_length, mainQN.batch_size: batch_size})
                    Q2 = sess.run(targetQN.Qout, feed_dict={
                        targetQN.scalarInput: np.stack(trainBatch[:, 3]/255.0),
                        targetQN.state_in: state_train,
                        targetQN.trainLength: trace_length, targetQN.batch_size: batch_size
                    })

                    end_multiplier = -(trainBatch[:, 4] - 1)    # it will be zero if done
                    doubleQ = Q2[range(batch_size*trace_length), Q1]
                    targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)     # target value of Q-running

                    sess.run(mainQN.updateModel, feed_dict={
                        mainQN.scalarInput: np.vstack(trainBatch[:, 0]/255.0), mainQN.targetQ: targetQ,
                        mainQN.state_in: state_train,
                        mainQN.actions: trainBatch[:, 1], mainQN.trainLength: trace_length,
                        mainQN.batch_size: batch_size
                    })

            rAll += r   # total reward
            s = s1      # flattened state
            sP = s1P    # original state
            state = state1  # rnn state
            if d:
                break

        # per episode updates
        bufferArray = np.array(episodeBuffer)
        episodeBuffer = list(zip(bufferArray))
        myBuffer.add(episodeBuffer)
        jList.append(j)
        rList.append(rAll)

        # print("" + str(i) + "/" + str(total_steps))

        if i % 1000 == 0 and i != 0:
            saver.save(sess, path+'/model-'+str(i)+'.cptk')
            print("Saved Model:" + path+'/model-'+str(i)+'.cptk')

        if len(rList) % summaryLength == 0 and len(rList) != 0:
            print(total_steps, np.mean(rList[-summaryLength:]), e)
            saveToCenter(i, rList, jList, np.reshape(np.array(episodeBuffer), [len(episodeBuffer), 5]),
                         summaryLength, h_size, sess, mainQN, time_per_step)
    saver.save(sess, path+'/model-'+str(i)+'.cptk')

