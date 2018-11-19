import tensorflow as tf

# -----------------------------------1.矩阵和常量-------------------------
'''
matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2], [2]])
# tf.matmul:矩阵乘法，tf.multiply:数乘
# 当tf.multiply参数为矩阵时，执行矩阵数乘
product = tf.matmul(matrix1, matrix2)
a = tf.constant(1)
b = tf.constant(2)
product2 = tf.multiply(a, b)
with tf.Session() as sess:
    # run获取返回的结果，可以一次获取多个结果，多个参数用[]括好
    result = sess.run([product, product2])
    print(result)

    # 或者分开获取结果    
    # result1 = sess.run(product)
    # result2 = sess.run(product2)
    # print(result1, result2)
'''

# -----------------------------------2.变量和常量-------------------------
'''
# 创建一个变量, 初始化为标量 0
state = tf.Variable(0, name='counter')

# 创建一个 op, 其作用是使 state 增加 1
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)
init_op = tf.global_variables_initializer()

# 启动图, 运行 op
with tf.Session() as sess:
    sess.run(init_op)
    # 打印 'state' 的初始值
    print(sess.run(state))
    for _ in range(3):
        # 运行 op, 更新 'state', 并打印 'state'
        sess.run(update)
        print(sess.run(state))
'''

# -----------------------------------3.Feed与tf.placeholder占位符-------------------------
'''
# 其实就是只是定义了一个类型的op，没有赋值，等到执行的时候才把值feed给那个op
# 于是op开始就只是占了一个位置
# feed必须是run的一个参数
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)
with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: 3.0, input2: 4.0}))
'''