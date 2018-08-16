
'''
This script provides ResVggNet and VggNet model.

'''

import tensorflow as tf

class Model:

    def __init__(self, class_num, pool_size, is_training):
        self.class_num = class_num
        self.pool_size = pool_size
        self.is_training = is_training

    def conv(self, x_tensor, conv_num_outputs, conv_ksize=3, conv_strides=1, conv_padding='SAME', name=None):
        res = tf.layers.conv2d(x_tensor, conv_num_outputs, conv_ksize, strides=conv_strides, padding=conv_padding,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(), name=name)
        return res

    def maxpool(self, x_tensor, pool_strides=(2, 2)):
        res = tf.nn.max_pool(x_tensor, ksize=[1, self.pool_size[0], self.pool_size[1], 1],
                             strides=[1, pool_strides[0], pool_strides[1], 1], padding='SAME')
        return res

    def avgpool(self, x_tensor, pool_strides=(2, 2)):
        res = tf.nn.avg_pool(x_tensor, ksize=[1, self.pool_size[0], self.pool_size[1], 1],
                             strides=[1, pool_strides[0], pool_strides[1], 1], padding='SAME')
        return res

    def fc(self, x_tensor, num_outputs, active=None, name=None):
        std_dev = x_tensor.shape[-1].value ** -0.5
        weight = tf.Variable(tf.random_normal([x_tensor.shape[-1].value, num_outputs], stddev=std_dev))
        bias = tf.Variable(tf.zeros([num_outputs]))
        res = tf.add(tf.matmul(x_tensor, weight), bias, name=name)
        if active == 'relu':
            res = tf.nn.relu(res)
        return res

    def conv_with_batch_norm(self, X, size):
        net = self.conv(X, size)
        net = tf.layers.batch_normalization(net, training=self.is_training)
        return net

    def basic_residual_block(self, X, size, type=1):
        if type == 1:
            residual = X
        elif type == 2:
            residual = self.conv_with_batch_norm(X, size)
        net = self.conv_with_batch_norm(X, size)
        net = tf.nn.relu(net)
        net = self.conv_with_batch_norm(net, size)
        return residual + net

    def residual_block(self, X, size, is_reduce=True):
        net = self.basic_residual_block(X, size, type=2)
        net = tf.nn.relu(net)
        net = self.basic_residual_block(net, size, type=1)
        if is_reduce:
            net = self.maxpool(net)
        net = tf.nn.relu(net)
        return net

    def ResVggNet(self, input_op):
        net = self.conv(input_op, 64, name="input_node")
        net = self.conv(net, 64)
        net = self.maxpool(net)
        net = tf.nn.relu(net)
        net = self.residual_block(net, 64)
        net = self.residual_block(net, 128)
        net = self.residual_block(net, 128)
        net = self.residual_block(net, 256, is_reduce=False)
        net = self.avgpool(net)
        net = tf.contrib.layers.flatten(net)
        net = self.fc(net, 768, active='relu')
        logits = self.fc(net, self.class_num, name="output_node")

        return logits

    def VggNet(self,X):
        net = self.conv(X, 64)
        net = self.conv(net, 64)
        net = self.maxpool(net)
        net = self.conv(net, 128)
        net = self.conv(net, 128)
        net = self.maxpool(net)
        net = self.conv(net, 256)
        net = self.conv(net, 256)
        net = self.conv(net, 256)
        net = self.maxpool(net)
        net = self.conv(net, 512)
        net = self.conv(net, 512)
        net = self.conv(net, 512)
        net = self.maxpool(net)
        net = self.conv(net, 512)
        net = self.conv(net, 512)
        net = self.conv(net, 512)
        net = self.maxpool(net)
        net = tf.contrib.layers.flatten(net)
        net = self.fc(net, 768, active='relu')
        tf.layers.batch_normalization(net, training=self.is_training)
        net = self.fc(net, 768, active='relu')
        tf.layers.batch_normalization(net, training=self.is_training)
        logits = self.fc(net, self.class_num, name="output_node")

        return logits
