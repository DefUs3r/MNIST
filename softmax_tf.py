import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x=tf.placeholder(tf.float32,[None,784])
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(x,W)+b)
y_prime=tf.placeholder(tf.float32,[None,10])
cross_entropy=tf.reduce_mean(-1*tf.reduce_sum(y_prime*tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess=tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(100):
    batch_xs, batch_ys=mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_prime:batch_ys})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_prime, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_prime: mnist.test.labels}))
