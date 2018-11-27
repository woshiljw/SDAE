import tensorflow as tf


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
                                  name=name)


def max_pool_2x2(x,name='pool_layer'):
    return tf.nn.max_pool(x,
                          ksize=[1,2,2,1],
                          strides=[1,2,2,1],
                          padding='SAME',
                          name=name)

def unpooling(x,)

def conv_layer(x,shape,name='conv_layer'):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(x,W,name=name)+b)

def full_layer(input,size):
    in_size = int(input.get_shape()[1])
    W = init_weights([in_size,size])
    b = init_bias(size)
    return tf.matmul(input,W)+b

x = tf.placeholder(dtype=tf.float32,shape=[None,784])

def encoder(x):
    x_image = tf.reshape(x,[-1,28,28,1])

    conv1 = conv_layer(x_image,[3,3,1,16],name='conv_layer_1')
    conv1_pool = max_pool_2x2(conv1,name='pool_layer_1')

    conv2 = conv_layer(conv1_pool,[3,3,16,8],name='conv_layer_2')
    conv2_pool = max_pool_2x2(conv2,name='pool_layer_2')

    return conv2_pool

def decoder(feature):
    pass

feature = encoder(x)
