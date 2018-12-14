import tensorflow as tf
sess = tf.InteractiveSession()

NUM_INPUTS = 3
NUM_HIDDEN = 5
NUM_OUTPUTS = 3

do_training = 1
save_trained = 1
save_file = "./tf-intro.sess"

x = tf.placeholder(tf.float32, shape=[None, NUM_INPUTS], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, NUM_INPUTS], name='y_')

# Input layer to hidden layer
# Weights (fully connected)
W_fc1 = tf.truncated_normal([NUM_INPUTS, NUM_HIDDEN], mean=0.5, stddev=0.707)
W_fc1 = tf.Variable(W_fc1, name='W_fc1')

# Biases (fully connected)
b_fc1 = tf.truncated_normal([NUM_HIDDEN], mean=0.5, stddev=0.707)
b_fc1 = tf.Variable(b_fc1, name='b_fc1')

# Hidden units
h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

# Hidden layer to output layer
# Weights (fully connected)
W_fc2 = tf.truncated_normal([NUM_HIDDEN, NUM_OUTPUTS], mean=0.5, stddev=0.707)
W_fc2 = tf.Variable(W_fc2, name='W_fc2')

# Biases (fully connected)
b_fc2 = tf.truncated_normal([NUM_OUTPUTS], mean=0.5, stddev=0.707)
b_fc2 = tf.Variable(b_fc2, name='b_fc2')

# Output
y = tf.matmul(h_fc1, W_fc2) + b_fc2

results = tf.sigmoid(y, name='results')
cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_))

# Training
train_step = tf.train.RMSPropOptimizer(0.25, momentum=0.5).minimize(cross_entropy)
inputvals =  [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1],
              [1, 1, 0], [1, 1, 1]]
targetvals = [[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0],
              [1, 1, 1], [0, 0, 0]]

if do_training == 1:
    sess.run(tf.global_variables_initializer())
    for i in range(10001):
        if i%100 == 0:
            train_error = cross_entropy.eval(feed_dict={x:inputvals, y_:targetvals})
            print("step %d, training error %g"%(i, train_error))
            if train_error < 0.0005:
                break
        sess.run(train_step, feed_dict={x:inputvals, y_:targetvals})

    if save_trained == 1:
        print("Saving neural network to %s.*"%(save_file))
        saver = tf.train.Saver()
        saver.save(sess, save_file)
else:
    print ("Loading neural network from %s"%(save_file))
    saver = tf.train.Saver()
    saver.restore(sess, save_file)
    # Note: the restore both loads and initializes the variables

print ('\nCounting starting with: 0 0 0')
res = sess.run(results, feed_dict={x: [[0,0,0]]})
print('%g %g %g'%(res[0][0], res[0][1], res[0][2]))
for i in range(8):
    res = sess.run(results, feed_dict={x: res})
    print('%g %g %g'%(res[0][0], res[0][1], res[0][2]))
