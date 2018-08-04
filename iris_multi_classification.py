'''from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import mglearn
import pandas as pd

iris_dataset = load_iris()

print("iris_dataset의 키: \n{}".format(iris_dataset.keys()))

print(iris_dataset['DESCR'][:193] + "\n...")

print("타깃의 이름: {}".format(iris_dataset['target_names']))

print("특성의 이름: \n{}".format(iris_dataset['feature_names'])) #꽃잎의 길이,폭, 꽃받침의 길이,폭

print("data의 타입: {}".format(type(iris_dataset['data'])))

print("data의 크기: {}".format(iris_dataset['data'].shape))

print("data의 처음 다섯 행:\n{}".format(iris_dataset['data'][:-1])) #전체 데이터

print("target의 타입: {}".format(type(iris_dataset['target'])))

print("target의 크기: {}".format(iris_dataset['target'].shape))

print("타깃:\n{}".format(iris_dataset['target'])) #타켓(종류)은 0~2

X_train, X_test, y_train, y_test = train_test_split(
iris_dataset['data'], iris_dataset['target'], random_state=0)

print("X_train 크기: {}".format(X_train.shape)) #3/4는 학습용
print("y_train 크기: {}".format(y_train.shape))
print("X_test 크기: {}".format(X_test.shape)) #1/4는 테스트용
print("y_test 크기: {}".format(y_test.shape))

# X_train 데이터를 사용해서 데이터프레임을 만듭니다.
# 열의 이름은 iris_dataset.feature_names에 있는 문자열을 사용합니다.

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# 데이터프레임을 사용해 y_train에 따라 색으로 구분된 산점도 행렬을 만듭니다.
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)'''

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#그래프
iris = pd.read_csv("iris.csv", names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', "Species"])
iris.head()
iris.info()
sns.pairplot(iris, hue="Species")
plt.show()
#그래프끝

#분류
tf.set_random_seed(777)  # for reproducibility
x = np.loadtxt('iris.csv', delimiter=',', usecols=[0,1,2,3]) #자료중에 특성만 저장
y = np.loadtxt('iris.csv', delimiter=',',dtype=np.str , usecols=[4]) #종류저장

x_data = x[:,0:]
for i in range(len(y)):
    if y[i] == 'Iris-setosa':
        y[i] = 0
    elif y[i] == 'Iris-versicolor':
        y[i] = 1
    elif y[i] == 'Iris-virginica':
        y[i] = 2
y_data = np.array([[y] for y in y])

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data,test_size=0.0065, random_state=0) #학습,테스트 데이터 나누기

nb_classes = 3

X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.int32, [None, 1])

Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot사용하면 한차원이 더늘어남
print("one_hot", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes]) #원하는 모양으로 shape변경
print("reshape", Y_one_hot)

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# softmax
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

# Cross entropy cost/loss
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot) #softmax_cross_entropy_with_logits사용시 logits를 넣어준다 hypo가 아니다
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        sess.run(optimizer, feed_dict={X: X_train, Y: y_train})
        if step % 200 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X: X_train, Y: y_train})
            print("Step: {:5}\tLoss: {:.3f}\tAccuracy: {:.2%}".format(step, loss, acc))
            #print(acc)

    # Let's see if we can predict
    pred, ac = sess.run([prediction, accuracy], feed_dict={X: X_test, Y:y_test})
    #print(ac)
    # y_data: (N,1) = flatten => (N, ) matches pred.shape
    #for p, y in zip(pred, y_test.flatten()):
        # print("[{}] Prediction: {} Real Y: {}".format(p == int(y), p, int(y)))
