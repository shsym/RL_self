# Example for n-armed bandit problem

import tensorflow as tf
import numpy as np

# make bandits
n_bandits = 10
# bandits = np.multiply(np.random.randn(n_bandits), 3)    # Low number = low threshold will give high reward
bandits = [-5.04006619, -3.31135104, -3.52528687, -2.1492237, 1.70333692, -0.23218338, 1.83488429,  0.8272563,
           -4.99673734, 3.01440605]


def pull_bandit(bandit):
    rand_num = np.random.randn(1)   # 1 dimensional random number (mean=0, variance=1)
    return rand_num[0] - bandit
    # if rand_num > bandit:
    #     return 1
    # else:
    #     return -1


tf. reset_default_graph()

W = tf.Variable(tf.ones([n_bandits]))   # weight: initialized with 1, [n_bandits dimensional matrix]
chosen_action = tf.argmax(W, 0)    # Find the max weight in the first (and only) dimention

reward = tf.placeholder(shape=[1], dtype=tf.float32)
action = tf.placeholder(shape=[1], dtype=tf.int32)

responsible_W = tf.slice(W, action, [1])    # get the corresponding weight that cause this action
loss = -(tf.log(responsible_W) * reward)    # Loss = -log(weight) * reward(=advantage)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
updateModel = optimizer.minimize(loss)

# environment setup and launch
total_episodes = 1000
total_reward = np.zeros(n_bandits)
epsilon = 0.1

init = tf.initialize_all_variables()

print("Bandits:")
print(bandits)

# Tensorflow session
with tf.Session() as sess:
    sess.run(init)  # initialize
    i = 0   # id for episode
    while i < total_episodes:
        # epsilon
        if np.random.rand(1) < epsilon:
            a = np.random.randint(n_bandits)
        else:
            a = sess.run(chosen_action)    # Choose action based on weight

        r = pull_bandit(bandits[a])    # Get the reward by pull action-th bandit

        _, resp, ww = sess.run([updateModel, responsible_W, W], feed_dict={reward: [r], action: [a]})

        total_reward[a] += r

        i += 1

        if i % 50 == 0:
            print(str(i) + "|Running reward for the " + str(n_bandits) + " bandits: " + str(ww))

print("The agent thinks bandit " + str(np.argmax(ww)) + " is the most promising....")
if np.argmax(ww) == np.argmax(-np.array(bandits)):
    print("...and it was right!")
else:
    print("...and it was wrong!")

