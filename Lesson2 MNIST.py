from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("./data/MNIST/", one_hot=True)
# #定义图中op
# x为输入
x = tf.placeholder("float", [None, 784])
# w,b为参数
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# softmax模型,y为预测值
y = tf.nn.softmax(tf.matmul(x, w) + b)
# #训练模型
# y_为真实值
y_ = tf.placeholder("float", [None, 10])
# 定义损失函数，使用交叉熵损失
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# 选择优化算法，这里使用梯度下降
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# #开始启动session
# 初始化所有的变量，必须的
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# 模型迭代1000次
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# #评估模型
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
