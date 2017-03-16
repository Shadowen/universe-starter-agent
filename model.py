import numpy as np
import tensorflow as tf


def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)


class LSTMPolicy(object):
    def __init__(self, ob_space, ac_space):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))

        conv1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=(5, 5), padding='same', activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=2)
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2, 2), strides=2)
        fc1 = tf.layers.dense(inputs=flatten(pool2), units=1024, activation=tf.nn.relu)

        self.logits = tf.layers.dense(inputs=fc1, units=ac_space, activation=None, name='action')
        self.vf = tf.reshape(tf.layers.dense(inputs=fc1, units=1, name='value'), [-1])
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        """ Gets the initial state of the LSTM """
        return self.state_init

    def act(self, ob):
        """
        Sample an action from the policy. Also returns the value of the current state.
        :param ob: observation
        :return: (action, value)
        action - action sampled from multinomial distribution
        value - value function at this state
        """
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf], {self.x: [ob]})

    def value(self, ob, c, h):
        """ Calculate the estimated value of a given state. """
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob]})[0]
