"""
这句话就是导入了官网所说的input_data.py文件
嫌麻烦，可以把这个input_data.py文件内容复制到一个自己新建的.py文件中
这样导入语句会更加简洁（当然只有这个功能而已）
注意：A,py和B.py在同一个目录时，若要在A.py中执行import B
先要，在project view中，“右击”A，B所在文件夹，选择“Mark Dictionary as”-->“Source Root”
这样A.py中的"import B"才不会提示错误
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("./data/MNIST/", one_hot=True)
# --------------------定义op
# x为输入，两种参数"float"与tf.float32都可以(受Lesson1影响，我试了一下，个人倾向于tf.float32)
# x = tf.placeholder("float", [None, 784])
x = tf.placeholder(tf.float32, [None, 784])
# y_为真实值
y_ = tf.placeholder("float", [None, 10])

# w,b为softmax模型参数
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# softmax模型,预测值输出y
y = tf.nn.softmax(tf.matmul(x, w) + b)

# 定义损失函数，使用交叉熵损失
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# 选择优化算法，这里使用梯度下降
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# --------------------开启Session
# 初始化所有变量，必须的op
init = tf.global_variables_initializer()
# 使用with语句的时候，下面所有代码都得缩进Tab，保证这些op在Session中
with tf.Session() as sess:
    sess.run(init)
    # 模型迭代1000次
    for i in range(1000):
        # 每次循环，随机抓取训练数据中的100个数据点
        # 用这些训练数据点，替换之前的x,y_占位符，来运行"train_step"op
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
# #评估模型
    # 比较预测标签和实际标签是否一致，返回的是tf.equal返回的是bool类型，如[False，True，..]
    # tf.cast把[False，True，..]--->[0,1,...]
    # tf.reduce_mean对[0,1,...]，即总的分类结果求均值，算得accuracy
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # 此时x,y_由测试点的数据来替换了
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
