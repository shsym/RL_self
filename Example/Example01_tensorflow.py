#Tensorflow version of Q-learning for FrozenLake problem

import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

#setup environment
env = gym.make('FrozenLake-v0')

tf.reset_default_graph()

#basic tensors for the network
inputs = tf.placeholder(shape=[1,16], dtype=tf.float32) #1 x 16 matrix with float32 data
W = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))    #16 x 4 matrix filled with a random value in [0, 0.01]
Qout = tf.matmul(inputs, W)  #input x weight => 1 x 4 matrix
predict = tf.argmax(Qout, 1)    #use axis 1 for argmax (4 elements), find the action that has maximum estimation

#Calculate the loss
nextQ = tf.placeholder(shape=[1, 4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))   #reduce the matrix of the sum on element-wise squared error
#define trainer and objective: what to minimize
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

#Start training
init = tf.initialize_all_variables()

#Set learning parameters
y = .99
e = 0.1
n_episodes = 2000

jList = []
rList = []
#Start tensorflow session
with tf.Session() as sess:
    sess.run(init)
    for i in range(n_episodes):
        #reset environment
        state = env.reset()
        rAll = 0
        d = False
        j = 0
        #The Q-Network
        while  j < 99:
            j += 1
            #implement epsilon-greedy action election
            # feed inputs as current state = state-th vector of identity matrix
            # Flows: action <= predict, allQ <= Qout
            action, allQ = sess.run([predict, Qout], feed_dict={inputs: np.identity(16)[state:state+1]})

            #predict has an index of action that has the maximum expected reward
            if np.random.rand(1) < e:   #epsilon part
                action[0] = env.action_space.sample()   #random sampling of the action: random exploration

            #get new state
            next_state, reward, d, _ = env.step(action[0])

            # calculate maximum value of Q value at the new state
            nextQ_value = sess.run(Qout, feed_dict={inputs: np.identity(16)[next_state:next_state+1]})
            maxQ = np.max(nextQ_value)
            # update current Q
            targetQ = allQ  #1 x 4 matrix
            targetQ[0, action[0]] = reward + y*maxQ #update the Q-value of the corresponding action
            _, nextW = sess.run([updateModel, W], feed_dict={inputs:np.identity(16)[state:state+1], nextQ:targetQ})

            #update reward and state
            rAll += reward
            state = next_state
            if d == True:   #destination
                e = 1./((i/50) + 10)
                break

        jList.append(j) #how fast did we reach the goal
        rList.append(rAll)

print("Percent of succesful episodes: "+ str(sum(rList)/n_episodes*100.) + "%")
