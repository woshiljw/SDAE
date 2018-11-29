import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from Noise import Noise

def init_weights(shape):
    '''
    define the weights
    :param shape: weights' shape
    :return: tensorflow variable
    '''
    weights = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(weights)

def init_bias(shape):
    bias = tf.constant(0.1,shape=shape)
    return tf.Variable(bias)

def conv2d(x,W,name):
    return tf.nn.conv2d(x,
                        filter=W,
                        strides=[1,1,1,1],
                        padding='SAME',
                        name=name)

def conv2d_transpose(x,W,output_shape,name):
    return tf.nn.conv2d_transpose(x,
                                  filter=W,
                                  output_shape=output_shape,
                                  strides=[1,1,1,1],
                                  padding='SAME',
                                  name=name)


def max_pool_2x2(x,name='pool_layer'):
    return tf.nn.max_pool(x,
                          ksize=[1,2,2,1],
                          strides=[1,2,2,1],
                          padding='SAME',
                          name=name)

def unpooling(x,W,output_shape,name='unpool_layer'):
    W = np.zeros(W,np.float32)
    print(W.shape[2],W.shape[3])
    W[0,0,:,:] = np.eye(W.shape[2],W.shape[3])
    W = tf.constant(W)
    return tf.nn.conv2d_transpose(x,
                                  filter=W,
                                  output_shape=output_shape,
                                  strides=[1,2,2,1],
                                  padding='SAME',
                                  name=name)

def conv_layer(x,shape,name='conv_layer'):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(x,W,name=name)+b)

def deconv_layer(x,shape,output_shape,name='deconv_layer'):
    W = init_weights(shape)
    b = init_bias([shape[2]])
    print('b:{}'.format(b.shape))

    return tf.nn.conv2d_transpose(x,
                               W,
                               output_shape,
                               strides=[1,1,1,1],
                               padding="SAME",name=name)+b



def full_layer(input,size):
    in_size = int(input.get_shape()[1])

    W = init_weights([in_size,size])
    b = init_bias([size])
    return tf.nn.relu(tf.matmul(input,W)+b)

x = tf.placeholder(dtype=tf.float32,shape=[None,784])
y_i = tf.placeholder(dtype=tf.float32,shape=[None,784])
y_image = tf.reshape(y_i,[-1,28,28,1])

filters={
    "conv1/deconv1":[3,3,1,16],
    "conv2/deconv2":[3,3,16,8]
}

def mkModel(x):
    x_image = tf.reshape(x,[-1,28,28,1])

    #encode
    conv1 = conv_layer(x_image,filters["conv1/deconv1"],name='conv1')
    conv1_pool = max_pool_2x2(conv1,name='conv1_pool')

    conv2 = conv_layer(conv1_pool,filters["conv2/deconv2"],name='conv2')
    conv2_pool = max_pool_2x2(conv2,name='conv2_pool')

    encode = full_layer(tf.reshape(conv2_pool,[-1,7*7*8]),20)

    decode = full_layer(encode,7*7*8)

    unpool1 = unpooling(tf.reshape(decode,[-1,7,7,8]),[2,2,8,8],tf.shape(conv2),name='unpool1')
    print("unpool:{}".format(unpool1.shape))
    deconv1 = tf.nn.relu(deconv_layer(unpool1,filters["conv2/deconv2"],tf.shape(conv1_pool),name='deconv1'))
    print("deconv1:{}".format(deconv1.shape))

    unpool2 = unpooling(deconv1,[2,2,16,16],tf.shape(conv1),name='unpool2')
    print("unpool2:{}".format(unpool2.shape))
    decode = tf.nn.sigmoid(deconv_layer(unpool2,filters["conv1/deconv1"],tf.shape(x_image),name='decode2'))
    print("decode2:{}".format(decode.shape))

    return x_image,decode

x_image,decode=mkModel(x)

cost = tf.reduce_mean(tf.square(y_image-decode))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)

mnist = input_data.read_data_sets('../data',one_hot=True)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(5000):
        x_,y = mnist.train.next_batch(64)
        x_noise = Noise(x_)
        if i % 10 == 0:
            print('In {} step,cost: {}'.format(i,sess.run(cost,feed_dict={x:x_noise.GaussianNoise(0.1),y_i:x_})))
        sess.run(train_step,feed_dict={x:x_noise.GaussianNoise(0.1),y_i:x_})

    x_test = mnist.test.images.reshape(10, 1000, 784)

    x_, result = sess.run([x_image, decode], feed_dict={x: Noise(x_test[0]).GaussianNoise(0.2),y_i:Noise(x_test[0]).GaussianNoise(0.2)})
    print(x_[0].shape)
    plt.imshow(np.reshape(x_[0],[28,28]))
    plt.show()
    plt.imshow(np.reshape(result[0],[28,28]))
    plt.show()