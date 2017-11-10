# Example for contextual bandits
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


# Define contextual bandit
class ContextualBandit():
        def __init__(self):
            self.state = 0  # initial state
            self.bandits = np.array([[0.2, 0, -0.0, -5],[0.1, -5, 1, 0.25],[-5, 5, 5, 5]])  # different kinds of bandits
            self.n_bandits = self.bandits.shape[0]
            self.n_actions = self.bandits.shape[1]  # number of actions

        def get_bandit(self):
            self.state = np.random.randint(0, len(self.bandits))    # randomly update internal state
            return self.state

        def pull_arm(self, action):
            # get the reward from given action (+internal state)
            bandit = self.bandits[self.state, action]
            result = np.random.randn(1)
            if result[0] > bandit:  # small bandit has higher probability to return positive reward
                return 1
            else:
                return -1


# Define agent
class agent():
    def __init__(self, lr, s_size, a_size):  # gamma, size of state, size of action
        self.state_in = tf.placeholder(shape=[1], dtype=tf.int32)   # int 32 type encoding of state
        state_in_oh = slim.one_hot_encoding(self.state_in, s_size)
        # make fully connected NN layer: state -(weight)-> action
        output = slim.fully_connected(state_in_oh, a_size,      # number of inputs and number of ouputs
                                      biases_initializer=None,
                                      activation_fn=tf.nn.sigmoid,
                                      weights_initializer=tf.ones_initializer())
        # output may contain weight matrix

        self.output = tf.reshape(output, [-1])  # reshape output, [-1] means flattening input
        self.chosen_action = tf.argmax(self.output, 0)  # return index maximizing the first dimension(0) of given output

        # define input holders and connects networks
        self.reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
        # from the fully connected layer, self.output, get the responsible W
        self.responsible_W = tf.slice(self.output, self.action_holder, [1]) # input, begin, size
        self.loss = -(tf.log(self.responsible_W)*self.reward_holder)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        self.update = optimizer.minimize(self.loss)


# Main function
tf.reset_default_graph()    # clean up Tensorflow graph

cBandit = ContextualBandit()
myAgent = agent(lr=0.001, s_size=cBandit.n_bandits, a_size=cBandit.n_actions)   # initialize agent
weights = tf.trainable_variables()[0]   # only for investigate network

total_episode = 10000
total_reward = np.zeros([cBandit.n_bandits, cBandit.n_actions])
e = 0.1

init = tf.initialize_all_variables()

# launch the Tensorflow session
with tf.Session() as sess:
    sess.run(init)  # initialize variables
    i = 0
    while i < total_episode:
        s = cBandit.get_bandit()    # get the next state

        # epsilon-based gradient
        if np.random.rand(1) < e:
            action = np.random.randint(cBandit.n_actions)
        else:
            action = sess.run(myAgent.chosen_action, feed_dict={myAgent.state_in:[s]})
        # get the reward
        reward = cBandit.pull_arm(action)
        # update the network
        feed_dict={myAgent.action_holder: [action], myAgent.reward_holder: [reward], myAgent.state_in: [s]}
        _, ww = sess.run([myAgent.update, weights], feed_dict=feed_dict)
        # update reward
        total_reward[s, action] += reward
        i += 1
        if i % 500 == 0:
            print("Mean reward for each of the " + str(cBandit.n_bandits) +
                  " bandits: " + str(np.mean(total_reward, axis=1)))


for a in range(cBandit.n_bandits):
    print("The agent thinks action " + str(np.argmax(ww[a]) + 1) + " for bandit " + str(a + 1)
          + " is the most promising....")
    if np.argmax(ww[a]) == np.argmin(cBandit.bandits[a]):
        print("...and it was right!")
    else:
        print("...and it was wrong!")