#testing regular and hebbian networks on XOR

import tensorflow as tf
import numpy as np
import random

def test0():
    x_ = tf.placeholder(tf.float32, shape=[2,2], name="x-input")
    y_ = tf.placeholder(tf.float32, shape=[2,1], name="y-input")
    
    Theta1 = tf.Variable(tf.random_uniform([2,2], -1, 1), name="Theta1")
    Theta2 = tf.Variable(tf.random_uniform([2,2], -1, 1), name="Theta2")
    Theta3 = tf.Variable(tf.random_uniform([2,1], -1, 1), name="Theta3")
    
    Bias1 = tf.Variable(tf.zeros([2]), name="Bias1")
    Bias2 = tf.Variable(tf.zeros([2]), name="Bias2")
    Bias3 = tf.Variable(tf.zeros([1]), name="Bias3")
    
    A2 = tf.nn.relu(tf.matmul(x_, Theta1) + Bias1)
    A3 = tf.nn.relu(tf.matmul(A2, Theta2) + Bias2)
    Hypothesis = tf.sigmoid(tf.matmul(A3, Theta3) + Bias3)
    
    cost = tf.reduce_mean(( (y_ * tf.log(Hypothesis)) + 
        ((1 - y_) * tf.log(1.0 - Hypothesis)) ) * -1)
    
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
    
    XOR_X1 = [[0,0],[0,1]]
    XOR_Y1 = [[0],[1]]
    XOR_X2 = [[1,0],[1,1]]
    XOR_Y2 = [[1],[0]]

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    tracking = np.ndarray((50,1))
    for i in range(5000):
            flip = random.randint(1,2)
            if (flip == 1):
                sess.run(train_step, feed_dict={x_: XOR_X1, y_: XOR_Y1})
                sess.run(train_step, feed_dict={x_: XOR_X2, y_: XOR_Y2})
            else:
                sess.run(train_step, feed_dict={x_: XOR_X2, y_: XOR_Y2})
                sess.run(train_step, feed_dict={x_: XOR_X1, y_: XOR_Y1})
            if i % 100 == 0:
                print('Epoch ', i)
                if (flip == 1):
                    hyp1 = sess.run(Hypothesis, feed_dict={x_: XOR_X1, y_: XOR_Y1})
                    hyp2 = sess.run(Hypothesis, feed_dict={x_: XOR_X2, y_: XOR_Y2})
                    print('Hypothesis 1', hyp1)
                    print('Hypothesis 2', hyp2)
                else:
                    hyp1 = sess.run(Hypothesis, feed_dict={x_: XOR_X1, y_: XOR_Y1})
                    hyp2 = sess.run(Hypothesis, feed_dict={x_: XOR_X2, y_: XOR_Y2})
                    print('Hypothesis 2', hyp2)
                    print('Hypothesis 1', hyp1)
                c1 = sess.run(cost, feed_dict={x_: XOR_X1, y_: XOR_Y1})
                c2 = sess.run(cost, feed_dict={x_: XOR_X2, y_: XOR_Y2})
                c = (c1 + c2)/2
                print('cost ', c)
                tracking[(i//100), 0] = c
    np.save("baseXOR_20", tracking)
    

def test1():
    x_ = tf.placeholder(tf.float32, shape=[4,2], name="x-input")
    y_ = tf.placeholder(tf.float32, shape=[4,1], name="y-input")

    Theta1 = tf.Variable(tf.random_uniform([2,2], -1, 1), name="Theta1")
    Theta2 = tf.Variable(tf.random_uniform([2,1], -1, 1), name="Theta2")

    Bias1 = tf.Variable(tf.zeros([2]), name="Bias1")
    Bias2 = tf.Variable(tf.zeros([1]), name="Bias2")
    
    Alpha = tf.Variable(0.1*tf.ones([2,2]))
    Hebb = tf.Variable(tf.zeros([2,2]))
 
    A2 = tf.sigmoid(tf.matmul(x_, (Theta1 + tf.matmul(Alpha,Hebb))) + Bias1)
    Hypothesis = tf.sigmoid(tf.matmul(A2, Theta2) + Bias2)

    cost = tf.reduce_mean(( tf.square(y_ - Hypothesis) + tf.square(Hypothesis - y_)))

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    XOR_X = [[0,0],[0,1],[1,0],[1,1]]
    XOR_Y = [[0],[1],[1],[0]]

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    prevH = Theta1
    for i in range(10000):
            sess.run(train_step, feed_dict={x_: XOR_X, y_: XOR_Y})
            t1 = sess.run(Theta1, feed_dict={x_: XOR_X, y_: XOR_Y})
            t2 = sess.run(Theta2, feed_dict={x_: XOR_X, y_: XOR_Y})
            Hebb = Hebb + (t2)*(prevH - t2*Hebb)
            prevH = t1
            if i % 10 == 0:
                print('Epoch ', i)
                print('Hypothesis ', sess.run(Hypothesis, feed_dict={x_: XOR_X, y_: XOR_Y}))
                print('cost ', sess.run(cost, feed_dict={x_: XOR_X, y_: XOR_Y}))
                print('hebb', Hebb)
                
def test2():
    x_ = tf.placeholder(tf.float32, shape=[2,2], name="x-input")
    y_ = tf.placeholder(tf.float32, shape=[2,1], name="y-input")
    
    w1 = tf.Variable(tf.random_uniform([2,2], -1,1), name="w1")
    w2 = tf.Variable(tf.random_uniform([2,2], -1,1), name="w2")
    w3 = tf.Variable(tf.random_uniform([2,1], -1,1), name="w3")
    
    b1 = tf.Variable(tf.zeros([2]), name="b1")
    b2 = tf.Variable(tf.zeros([2]), name="b2")
    b3 = tf.Variable(tf.zeros([1]), name="b3")
    
    a1 = tf.Variable(0.3*tf.ones([2,2]), name="a1")
    a2 = tf.Variable(0.3*tf.ones([2,2]), name="a2")
    
    h1 = tf.Variable(tf.zeros([2,2]), name="h1", trainable=False)
    h2 = tf.Variable(tf.zeros([2,2]), name="h2", trainable=False)
    
    l1 = tf.nn.relu(tf.matmul(x_, (w1 + tf.matmul(a1,h1))) + b1)
    l2 = tf.nn.relu(tf.matmul(l1, (w2 + tf.matmul(a2,h2))) + b2)
    l3 = tf.nn.softmax(tf.matmul(l2, w3) + b3, dim=0)
    
    XOR_X1 = [[0,0],[0,1]]
    XOR_Y1 = [[0],[1]]
    XOR_X2 = [[1,0],[1,1]]
    XOR_Y2 = [[1],[0]]
    
    cost = tf.reduce_mean(( (y_ * tf.log(l3)) + 
        ((1 - y_) * tf.log(1.0 - l3)) ) * -1)
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)


    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    pH1 = w1
    pH2 = w2
    n = 0.25
    tracking = np.ndarray((50,3,2,1))
    for i in range(5000):
            #reset Hebbian trace for episode
            h1 = tf.zeros([2,2])
            h2 = tf.zeros([2,2])
            #first half of training data
            flip = random.randint(1,2)
            if (flip == 1):
                sess.run(train_step, feed_dict={x_: XOR_X1, y_: XOR_Y1})
                h1 = n*(tf.matmul(tf.transpose(l1), l2)) + (1-n)*h1
                h2 = n*(tf.matmul(tf.transpose(l2), l3)) + (1-n)*h2
                sess.run(train_step, feed_dict={x_: XOR_X2, y_: XOR_Y2})
            else:
                sess.run(train_step, feed_dict={x_: XOR_X2, y_: XOR_Y2})
                h1 = n*(tf.matmul(tf.transpose(l1), l2)) + (1-n)*h1
                h2 = n*(tf.matmul(tf.transpose(l2), l3)) + (1-n)*h2
                sess.run(train_step, feed_dict={x_: XOR_X1, y_: XOR_Y1})
            if i % 100 == 0:
                print('Epoch ', i)
                if (flip == 1):
                    hyp1 = sess.run(l3, feed_dict={x_: XOR_X1, y_: XOR_Y1})
                    hyp2 = sess.run(l3, feed_dict={x_: XOR_X2, y_: XOR_Y2})
                    print('Hypothesis 1', hyp1)
                    print('Hypothesis 2', hyp2)
                else:
                    hyp1 = sess.run(l3, feed_dict={x_: XOR_X1, y_: XOR_Y1})
                    hyp2 = sess.run(l3, feed_dict={x_: XOR_X2, y_: XOR_Y2})
                    print('Hypothesis 2', hyp2)
                    print('Hypothesis 1', hyp1)
                c1 = sess.run(cost, feed_dict={x_: XOR_X1, y_: XOR_Y1})
                c2 = sess.run(cost, feed_dict={x_: XOR_X2, y_: XOR_Y2})
                c = (c1 + c2)/2
                print('cost ', c)
                tracking[(i//100), 0] = c
                tracking[(i//100), 1] = hyp1
                tracking[(i//100), 2] = hyp2
    np.save("semi_random_hebianXOR_0.25", tracking)
    
  
def addingTest():
    x_IN = np.ndarray((10,2), dtype="float32")
    y_SUM = np.ndarray((1), dtype="float32")
    
    x_ = tf.placeholder(tf.float32, shape=[2])
    y_ = tf.placeholder(tf.float32, shape=[1])
    
    w1 = tf.Variable(tf.random_uniform([2], 0,1), name="w1")
    w2 = tf.Variable(tf.random_uniform([2], 0,1), name="w2")
    w3 = tf.Variable(tf.random_uniform([1], 0,1), name="w3")
    
    b1 = tf.Variable(tf.zeros([2]), name="b1")
    b2 = tf.Variable(tf.zeros([2]), name="b2")
    b3 = tf.Variable(tf.zeros([1]), name="b3")
    
    a1 = tf.Variable(0.3*tf.ones([2]), name="a1")
    a2 = tf.Variable(0.3*tf.ones([2]), name="a2")
    
    h1 = tf.Variable(tf.zeros([2]), name="h1", trainable=False)
    h2 = tf.Variable(tf.zeros([2]), name="h2", trainable=False)
    
    l1 = tf.nn.relu((x_ * (w1 + a1*h1)) + b1)
    l2 = tf.nn.relu(l1 * (w2 + a2*h2) + b2)
    l3 = tf.nn.relu((l2 * w3) + b3)
    
    cost = tf.reduce_sum(tf.abs(y_SUM - l3))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    n = 0.5
    #100 epochs
    for i in range(100):
        #initialize artificial training sample, set label from (0,1,2)
        for j in range(10):
            x_IN[j,0] = random.randint(0,1)
            x_IN[j,1] = 0
        i1 = random.randint(0,9)
        i2 = random.randint(0,9)
        while (i1 == i2):
            i2 = random.randint(0,9)
        y_SUM[0] = x_IN[i1,0] + x_IN[i2,0]
        x_IN[i1,1] = x_IN[i2,1] = 1
        #set Hebbian memory to zero
        h1 = tf.zeros([2,2])
        h2 = tf.zeros([2,2])
        for j in range(9):
            vals = np.ndarray((2))
            vals[0] = x_IN[j,0]
            vals[1] = x_IN[j,1]
            hyp = sess.run(l3, feed_dict={x_: vals})
            h1 = n*(l1*l2) + (1-n)*h1
            h2 = n*(l2*l3) + (1-n)*h2
        vals[0] = x_IN[9,0]
        vals[0] = x_IN[9,1]
        sess.run(train_step, feed_dict={x_: vals, y_: y_SUM})
        hyp = sess.run(l3, feed_dict={x_: vals})
        cPrint = sess.run(cost, feed_dict={x_: vals, y_: y_SUM})
        
        print("-----Epoch " + str(i) + "-----")
        print("Prediction: " + str(hyp))
        print("Actual: " + str(y_SUM))
        print("Cost: " + str(cPrint))
    
    
addingTest()   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
