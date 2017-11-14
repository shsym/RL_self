# Example for model-based RL
""" Instead of explore the physical world every time,
    model-based RL will learn a model from the environment first,
    then the policy model will learn from both real world and the model"""

import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import math

# python 3 does not have xrange
import sys
if sys.version_info.major > 2:
    xrange = range
del sys

import gym
env = gym.make('CartPole-v0')

h_size = 8  # number of hidden nodes
learning_rate = 1e-2
gamma = 0.99    #discount factor
decay_rate = 0.99
resume = False

# batch sizes
model_bs = 3
real_bs = 3

s_size = 4 # dimension of input_state


# === policy network === #
tf.reset_default_graph()
observations = tf.placeholder(dtype=tf.float32, shape=[None, 4], name="input_x")    # input place holder
W1 = tf.get_variable("W1", shape=[4, h_size],
                     initializer=tf.contrib.layers.xavier_initializer())
# input to hidden layer connection with 'relu' activation ftn: [None, 4] x [4, h_size] => [None, h_size]
layer1 = tf.nn.relu(tf.matmul(observations, W1))

W2 = tf.get_variable("W2", shape=[h_size,1],
                     initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1, W2)   # estimated score would be => hidden x weight2 => [None, 1]
probability = tf.nn.sigmoid(score)  # from the score, we get probability using sigmoid function => [None,]

tvars = tf.trainable_variables()
input_y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="input_y")
advantages = tf.placeholder(dtype=tf.float32, name="reward_signal")
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
W1Grad = tf.placeholder(dtype=tf.float32, name="batch_grad1")
W2Grad = tf.placeholder(dtype=tf.float32, name="batch_grad2")
batchGrad = [W1Grad, W2Grad]    # gradient as inputs
loglik = tf.log(input_y * (input_y - probability) + (1 - input_y) * (input_y + probability))
loss = -tf.reduce_mean(loglik * advantages)
newGrad = tf.gradients(loss, tvars)     # gradient of loss for W1, W2
updateGrads = adam.apply_gradients(zip(batchGrad, tvars))


# === model layer === #
model_size = 256

input_data = tf.placeholder(dtype=tf.float32, shape=[None, 5])  # 5 as input?
with tf.variable_scope('rnnlm'):
    softmax_w = tf.get_variable("softmax_w", [model_size, 50])
    softmax_b = tf.get_variable("softmax_b", [50])

previous_state = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="previous_state")
# weight and bias for input - hidden layer connection
W1M = tf.get_variable("W1M", shape=[5, model_size],
                      initializer=tf.contrib.layers.xavier_initializer())
B1M = tf.Variable(tf.zeros([model_size]), name="B1M")
layer1M = tf.nn.relu(tf.matmul(previous_state, W1M) + B1M)  # input * weight + bias

W2M = tf.get_variable("W2M", shape=[model_size, model_size],
                      initializer=tf.contrib.layers.xavier_initializer())
B2M = tf.Variable(tf.zeros([model_size]), name="B2M")
layer2M = tf.nn.relu(tf.matmul(layer1M, W2M) + B2M) #final output from the model

# weight: model to output
wO = tf.get_variable("w0", shape=[model_size, 4],
                     initializer=tf.contrib.layers.xavier_initializer())
# weight: model to reward
wR = tf.get_variable("wR", shape=[model_size, 1],
                     initializer=tf.contrib.layers.xavier_initializer())
# weight: model to done (finished)
wD = tf.get_variable("wD", shape=[model_size, 1],
                     initializer=tf.contrib.layers.xavier_initializer())

# bias: model to output
bO = tf.Variable(tf.zeros([4]), name="bO")
# bias: model to reward
bR = tf.Variable(tf.zeros([1]), name="bR")
# bias: model to done
bD = tf.Variable(tf.ones([1]), name="bD")

# connection between model to output(=observation), reward, done
predicted_observation = tf.matmul(layer2M, wO, name="predicted_observation") + bO
predicted_reward = tf.matmul(layer2M, wR, name="predicted_reward") + bR
predicted_done = tf.sigmoid(tf.matmul(layer2M, wD, name="predicted_done") + bD)

# place holders for true observation
true_observation = tf.placeholder(dtype=tf.float32, shape=[None, 4], name="true_observation")
true_reward = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="true_reward")
true_done = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="true_done")

# make a state using the predicted values
predicted_state = tf.concat([predicted_observation, predicted_reward, predicted_done], 1)
# losses: error between true and predicted values
observation_loss = tf.square(true_observation - predicted_observation)
reward_loss = tf.square(true_reward - predicted_reward)
done_loss = tf.multiply(predicted_done, true_done) + tf.multiply(1-predicted_done, 1-true_reward)
done_loss = -tf.log(done_loss)
model_loss = tf.reduce_mean(observation_loss + done_loss + reward_loss) # total loss as a sum

modelAdam = tf.train.AdamOptimizer(learning_rate=learning_rate)
updateModel = modelAdam.minimize(model_loss)


# === helper functions === #
def reset_grad_buffer(grad_buffer):
    for ix, grad in enumerate(grad_buffer):
        grad_buffer[ix] = grad * 0
    return grad_buffer


def discount_reward(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):   # staring from the last to the first
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def step_model(sess, xs, action):
    # xs => history of observations
    toFeed = np.reshape(np.hstack([xs[-1][0], np.array(action)]), [1, 5])    # state = [predicted_state, action]
    myPredict = sess.run([predicted_state], feed_dict={previous_state: toFeed})
    reward = myPredict[0][:, 4]
    observation = myPredict[0][:, 0:4]
    observation[:, 0] = np.clip(observation[:, 0], -2.4, 2.4)
    observation[:, 2] = np.clip(observation[:, 2], -0.4, 0.4)
    doneP = np.clip(myPredict[0][:, 5], 0, 1)
    if doneP > 0.1 or len(xs) >= 300:
        done = True
    else:
        done = False
    return observation, reward, done


xs, drs, ys, ds = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 1
real_episodes = 1
init = tf.global_variables_initializer()
batch_size = real_bs

drawFromModel = False
trainTheModel = True
trainThePolicy = False
switch_point = 1

with tf.Session() as sess:
    rendering = False
    sess.run(init)
    observation = env.reset()
    x = observation
    gradBuffer = sess.run(tvars)    # shape of the grad buffer
    gradBuffer = reset_grad_buffer(gradBuffer)  # reset grad buffer

    while episode_number <= 5000:
        # draw environment only when the performance is higher than the threshold
        if (reward_sum/batch_size > 150 and drawFromModel is False) or rendering is True:
            env.render()
            rendering = True

        x = np.reshape(observation, [1, 4])
        tfprob = sess.run(probability, feed_dict={observations: x})
        action = 1 if np.random.uniform() < tfprob else 0

        # observation and action
        xs.append(x)
        y = 1 if action == 0 else 0
        ys.append(y)

        if drawFromModel is False:
            observation, reward, done, info = env.step(action)
        else:
            observation, reward, done = step_model(sess, xs, action)
        reward_sum += reward
        ds.append(done*1)
        drs.append(reward)

        if done:

            if drawFromModel is False:
                real_episodes += 1
            episode_number += 1

            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            epd = np.vstack(ds)
            xs, drs, ys, ds = [], [], [], []

            if trainTheModel is True:
                actions = np.array([np.abs(y-1) for y in epy][:-1])  # y => action conversion
                state_prevs = epx[:-1, :]
                state_prevs = np.hstack([state_prevs, actions])
                state_nexts = epx[1:, :]
                rewards = np.array(epr[1:, :])
                dones = np.array(epd[1:, :])
                state_nextsAll = np.hstack([state_nexts, rewards, dones])
                feed_dict = {previous_state: state_prevs, true_observation: state_nexts,
                             true_done: dones, true_reward: rewards}
                loss, pState, _ = sess.run([model_loss, predicted_state, updateModel], feed_dict=feed_dict)

            if trainThePolicy is True:
                discounted_epr = discount_reward(epr).astype('float32')
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)
                tGrad = sess.run(newGrad, feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})

                if np.sum(tGrad[0] == tGrad[0]) == 0:
                    break
                for ix, grad in enumerate(tGrad):
                    gradBuffer[ix] += grad

            if switch_point + batch_size == episode_number:
                switch_point = episode_number
                if trainThePolicy is True:
                    sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
                    gradBuffer = reset_grad_buffer(gradBuffer)

                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01

                if drawFromModel is False:
                    print('World Perf: Episode %f. Reward %f. action: %f. mean reward %f.' % (
                        real_episodes, reward_sum / real_bs, action, running_reward / real_bs))
                    if reward_sum / batch_size > 200:
                        break
                reward_sum = 0

                if episode_number > 100:
                    # swap training model
                    drawFromModel = not drawFromModel
                    trainTheModel = not trainTheModel
                    trainThePolicy = not trainThePolicy

            if drawFromModel == True:
                observation = np.random.uniform(-0.1, 0.1, [4])
                batch_size = model_bs
            else:
                observation = env.reset()
                batch_size = real_bs

print(real_episodes)


# figure
plt.figure(figsize=(8, 12))
for i in range(6):
    plt.subplot(6, 2, 2*i + 1)
    plt.plot(pState[:,i])
    plt.subplot(6,2,2*i+1)
    plt.plot(state_nextsAll[:,i])
plt.tight_layout()
