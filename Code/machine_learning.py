# @Author: Andrea F. Daniele <afdaniele>
# @Date:   Tuesday, January 23rd 2018
# @Email:  afdaniele@ttic.edu
# @Last modified by:   afdaniele
# @Last modified time: Wednesday, January 31st 2018

import tensorflow as tf
from utils import ProgressBar

def get_model( timesteps, num_features, rnn_state_size, num_classes, forward_only=True, learning_rate=0.001 ):
    # define placeholders for the input X
    X = tf.placeholder( tf.float32, [timesteps, None, num_features] )
    # pick LSTM as type of RNN cell to use
    lstm_cell = tf.contrib.rnn.BasicLSTMCell( rnn_state_size )
    # define Softmax projection matrix and bias factor
    softmax_W = tf.get_variable( "softmax_W",
        [rnn_state_size, num_classes],
        tf.float32,
        initializer=tf.contrib.layers.xavier_initializer()
    )
    softmax_b = tf.get_variable( "softmax_b",
        [num_classes],
        tf.float32,
        initializer=tf.contrib.layers.xavier_initializer()
    )
    # create container for output logits
    output_logits = []
    # define initial state and initial memory for the LSTM cell
    zero_state = tf.placeholder( tf.float32, [None, rnn_state_size] )
    hidden_state = zero_state
    current_state = zero_state
    # the LSTM object takes the initial state as a tuple (hidden, input)
    lstm_state = ( hidden_state, current_state )
    # create neural network
    pbar = ProgressBar( maxVal=timesteps )
    print '\nCreating RNNs... ',
    with tf.variable_scope('RNN/'):
        for t in range(timesteps):
            # Update the state of the LSTM chain after processing the input at time t
            lstm_cell = tf.contrib.rnn.BasicLSTMCell( rnn_state_size, reuse=t>0 )
            lstm_output, lstm_state = lstm_cell( X[t], lstm_state )
            # append the logit signal coming out of the LSTM to the output vector
            output_logits.append( lstm_output )
            # update progress bar
            pbar.next()

    # use only the output of the last RNN cell instead of computing the mean of all the outputs
    mean_logits = output_logits[-1]

    # project the logits so that the last dimension equals the number of classes
    logits = tf.matmul( mean_logits, softmax_W ) + softmax_b
    # convert logits into a probability distribution over classes
    Y = tf.nn.softmax( logits )

    # return pointers to probability distribution if in forward_only mode
    if( forward_only ):
        return X, Y, zero_state

    # => this section creates the backpropagation (needed only for training)
    # define placeholders for the supervision (labels)
    groundtruth = tf.placeholder( tf.int32, [None,] )
    labels = tf.one_hot(groundtruth, num_classes)

    # compute the log-loss (aka cross-entropy loss)
    loss = tf.nn.softmax_cross_entropy_with_logits( logits=logits, labels=labels )
    batch_loss = tf.reduce_mean( loss )

    # create trainer operation
    train_op = tf.train.AdamOptimizer(learning_rate).minimize( batch_loss )
    # return pointers
    return X, Y, groundtruth, zero_state, batch_loss, train_op
