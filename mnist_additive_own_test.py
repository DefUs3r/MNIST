import numpy as np
import cv2
from scipy import ndimage
import tensorflow as tf
import sys
import os
import math
import time
import matplotlib.pyplot as plt
from intro_noise import noisy as n
import procimg as _input_
#Additive Test for new brand new pics!!
"""
-----------------------------------------------SWITCHES-------------------------------------------------------
"""
#Important Switches
reuse = True
run_training = True
save_it = False
visualise = False
"""
---------------------------------------------IMAGE INPUT AND PROCESSING---------------------------------------------------------
"""
train_images=np.zeros((10,784))
train_correct_vals=np.zeros((10,10))

cv_images=np.zeros((10,784))
cv_correct_vals=np.zeros((10,10))

test_images=np.zeros((10,784))
test_correct_vals=np.zeros((10,10))

label=0
for number in [0,1,2,3,4,5,6,7,8,9]:
    train_images[label]=_input_.inp_img("train",number)
    cv_images[label]= _input_.inp_img("cv",number)
    test_images[label]= _input_.inp_img("test",number)
    train_correct_val = np.zeros((10))
    cv_correct_val = np.zeros((10))
    test_correct_val = np.zeros((10))
    train_correct_val[number]=1
    cv_correct_val[number] = 1
    test_correct_val[number] = 1
    train_correct_vals[label] = train_correct_val
    cv_correct_vals[label] = cv_correct_val
    test_correct_vals[label] = test_correct_val
    label+=1
"""
-----------------------------ARCHITECTURE PLANNING AND CONSTRUCTION ZONE-----------------------------------------------------------------
"""
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()

def var_summary(var):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean',mean)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
    tf.summary.scalar('stddev',stddev)
    tf.summary.scalar('max',tf.reduce_max(var))
    tf.summary.scalar('min',tf.reduce_min(var))
    tf.summary.histogram('hist',var)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#conv=>stride=1,zero padded,pool=>max(2X2)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
if reuse is True:
    dat_dict = np.load("var_store.npz")
    W = tf.convert_to_tensor(dat_dict['W'],dtype=tf.float32)
    var_summary(W)
    b = tf.convert_to_tensor(dat_dict['b'],dtype=tf.float32)
    var_summary(b)
    W_conv1 = tf.convert_to_tensor(dat_dict['W_conv1'],dtype=tf.float32)
    var_summary(W_conv1)
    b_conv1 = tf.convert_to_tensor(dat_dict['b_conv1'],dtype=tf.float32)
    var_summary(b_conv1)
    W_conv2 = tf.convert_to_tensor(dat_dict['W_conv2'],dtype=tf.float32)
    var_summary(W_conv2)
    b_conv2 = tf.convert_to_tensor(dat_dict['b_conv2'],dtype=tf.float32)
    var_summary(b_conv2)
    W_fc1 = tf.convert_to_tensor(dat_dict['W_fc1'],dtype=tf.float32)
    var_summary(W_fc1)
    b_fc1 = tf.convert_to_tensor(dat_dict['b_fc1'],dtype=tf.float32)
    var_summary(b_fc1)
    W_fc2 = tf.convert_to_tensor(dat_dict['W_fc2'],dtype=tf.float32)
    var_summary(W_fc2)
    b_fc2 = tf.convert_to_tensor(dat_dict['b_fc2'],dtype=tf.float32)
    var_summary(b_fc2)
    print("Variables restored")
else:
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    
sess.run(tf.global_variables_initializer())

#base architecture conv1-pool1-conv2-pool2-fc1-dropout-readout

x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
tf.summary.histogram('hidden fc1',h_fc1)
keep_prob = tf.placeholder(tf.float32)
tf.summary.scalar('dropout keep_prob',keep_prob)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
tf.summary.histogram('conv1 op',y_conv)
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
tf.summary.histogram('conv2 op',y_conv)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
#Using Adam Optimization
if reuse is False:
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy of net', accuracy)
sess.run(tf.global_variables_initializer())
"""
--------------------------------------VISUALISING WHAT CNN IS LEARNING------------------------------
"""
def getActivation(layer,inp):
    units = sess.run(layer, feed_dict={x:np.reshape(inp,[1,784],order='F'),keep_prob:1.0})
    plotNN(units)

def plotNN(units):
    filters = units.shape[3]
    plt.figure(1,figsize=(20,20))
    n_col = 6
    n_row = math.ceil(filters/n_col)+1
    for i in range(filters):
        plt.subplot(n_row,n_col,i+1)
        plt.title("Template Learnt #"+str(i))
        plt.imshow(units[0,:,:,i],interpolation="nearest",cmap="gray")

"""
------------------------------------------TRAINING ZONE----------------------------------------------------------------------
"""


if run_training is True:
    start_time = time.time()
    for i in range(20000):
        batch = mnist.train.next_batch(50)  # mini batch sampling
        if reuse is True:
            break
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            curr_time = time.time()
            print("Epoch:- %d\n Training Accuracy => %g\n Time Elapsed => %g" %(i, train_accuracy, curr_time-start_time))
        if reuse is False:
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        if visualise is True:
            visualiseImg = batch[0]
            plt.imshow(np.reshape(visualiseImg,[28,28]),interpolation="nearest",cmap="gray")
            getActivation(h_conv1,visualiseImg)
            getActivation(h_pool1,visualiseImg)
            getActivation(h_conv2,visualiseImg)
            getActivation(h_pool2,visualiseImg)
            getActivation(h_fc1,visualiseImg)
            
    curr_time = time.time()
    train_accuracy = accuracy.eval(feed_dict={x: train_images, y_:train_correct_vals, keep_prob:1.0})
    if reuse is False:
        train_step.run(feed_dict={x: train_images, y_:train_correct_vals, keep_prob:0.5})
        print("Epoch:- Personal Dataset, Training Accuracy => %g, Time Elapsed => %g" %(train_accuracy, curr_time-start_time))
    if visualise is True:
        visualiseImg = mnist.test.images[0]
        plt.imshow(np.reshape(visualiseImg,[28, 28]), interpolation="nearest", cmap="gray")
        getActivation(h_conv1, visualiseImg)
        getActivation(h_pool1, visualiseImg)
        getActivation(h_conv2, visualiseImg)
        getActivation(h_pool2, visualiseImg)
        getActivation(h_fc1, visualiseImg)
    if save_it is True:
        tot_dict = {}
        tot_dict = {'W': W.eval(), 'b': b.eval(), 'W_conv1': W_conv1.eval(), 'W_conv2': W_conv2.eval(), 'b_conv1': b_conv1.eval(),'b_conv2': b_conv2.eval(), 'W_fc1': W_fc1.eval(), 'W_fc2': W_fc2.eval(), 'b_fc1': b_fc1.eval(), 'b_fc2': b_fc2.eval()}
        np.savez("var_store", **tot_dict)
    print("MNIST test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    cv_accuracy = accuracy.eval(feed_dict={x: cv_images, y_: cv_correct_vals, keep_prob: 1.0})
    print("My own Cross validation accuracy %g" % cv_accuracy)
    if (cv_accuracy > 0.8):
        test_accuracy = accuracy.eval(feed_dict={x: test_images, y_:test_correct_vals, keep_prob:1.0})
        print("Test accuracy on own dataset %g", test_accuracy)
    else:
        print("Network is not learning. Need better hyperparameters.")
else:
    print("This is a debugging session. Not trained. Exiting program.\n")
