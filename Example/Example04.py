# Example for pole balancing problem
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import matplotlib.pyplot as plt

try:
    xrange = xrange
except:
    xrange = range

env = gym.make('CartPole-v0')

gamma = 0.99


def discount_reward(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r) # same shape with r but set to zeros
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]    #summing up delayed rewards
        discounted_r[t] = running_add   #cumulative reward
    return discounted_r


class Agent:
    def __init__(self, lr, s_size, a_size, h_size):
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)

        # make hidden layer [state (s_size) -> hidden (h_size)]
        hidden = slim.fully_connected(self.state_in, h_size, biases_initializer=None, activation_fn=tf.nn.relu)

        # make output layer [hidden (h_size) -> output (a_size)]
        self.output = slim.fully_connected(hidden, a_size, activation_fn=tf.nn.softmax, biases_initializer=None)

        # final output action
        self.chosen_action = tf.argmax(self.output, 1)

        # The following statements are for training
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        # Get index of output (h_size x a_size) =maybe=> [0, a_size, 2 * a_size, ... , h_size * a_size)
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder

        # Flatten output then pick-ups numbers pointed from
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        # Loss function
        self.loss = tf.reduce_mean(-tf.log(self.responsible_outputs) * self.reward_holder)

        tvars = tf.trainable_variables()    # get the all declared trainable variables
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))


# main script
tf.reset_default_graph() # reset Tensorflow grpah

myAgent = Agent(lr=1e-2, s_size=4, a_size=2, h_size=8)

total_episodes = 1000
max_ep = 999
update_frequency = 5

init = tf.global_variables_initializer()

# Run the Tersorflow session
with tf.Session() as sess:
    sess.run(init)
    i = 0
    total_reward = []
    total_length = []

    gradBuffer = sess.run(tf.trainable_variables())
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    while i < total_episodes:
        s = env.reset()
        running_reward = 0
        ep_history = []

        for j in range(max_ep):     # maximum step of a episode
            a_dist = sess.run(myAgent.output, feed_dict={myAgent.state_in:[s]})
            a = np.random.choice(a_dist[0], p=a_dist[0])
            a = np.argmax(a_dist == a)  # find an action 'a' from a_dist

            s1, r, d, _ = env.step(a) # get the reward
            ep_history.append([s, a, r, s1])    # tracking history (s_t, action, reward, s_t+1,)
            s = s1
            running_reward += r

            # Training with the delayed reward
            if d == True:
                ep_history = np.array(ep_history)   # convert ep_history as np.array
                ep_history[:, 2] = discount_reward(ep_history[:, 2])    # discount reward (index == 2)
                feed_dict = {myAgent.reward_holder: ep_history[:, 2],
                             myAgent.action_holder: ep_history[:, 1],
                             myAgent.state_in: np.vstack(ep_history[:, 0])}
                grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
                for idx, grad in enumerate(grads):
                    gradBuffer[idx] += grad

                if i % update_frequency == 0 and i != 0:
                    feed_dict = dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                    _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                    for ix, grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0

                total_reward.append(running_reward)
                total_length.append(j)
                break

        i += 1
        if i % 100 == 0:
            print(np.mean(total_reward[-100:]))


