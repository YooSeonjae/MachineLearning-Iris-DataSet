import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

tf.set_random_seed(777)  # for reproducibility
x = np.loadtxt('iris.csv', delimiter=',', usecols=[0, 1, 2, 3])  # 자료중에 특성만 저장
y = np.loadtxt('iris.csv', delimiter=',', dtype=np.str, usecols=[4])  # 종류저장

x_data = x[:, 0:]
for i in range(len(y)):
    if y[i] == 'Iris-setosa':
        y[i] = 0
    else:
        y[i] = 1
y_data = np.array([[y] for y in y])

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, random_state=0)  # 학습,테스트 데이터 나누기

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([4, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(-tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, a, _ = sess.run([cost, accuracy, train], feed_dict={X: X_train, Y: y_train})
        if step % 200 == 0:
            print(a)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: X_test, Y: y_test})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
