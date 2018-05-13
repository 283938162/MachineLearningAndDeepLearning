import tensorflow as tf  # 赋予Python访问TensorFlow类(classes)，方法（methods），符号(symbols)

"""
TensorFlow核心程序由2个独立部分组成：
    a:Building the computational graph构建计算图
    b:Running the computational graph运行计算图
"""
node1 = tf.constant(3.0, dtype = tf.float32)
node2 = tf.constant(4.0)  # also tf.float32 impicitly

print(node1, node2)

"""
session封装了TensorFlow运行时的控制和状态
"""

session = tf.Session()

print(session.run([node1, node2]))

# 以组合Tensor节点操作(操作仍然是一个节点)来构造更加复杂的计算，
node3 = tf.add(node1, node2)
print('node3 :', node3)
print('session.run(node3) :', session.run(node3))

path = r'F:\DataResposity\tensorboard'

def tensorboardTest():
	with tf.name_scope('input1'):
		input1 = tf.constant([1.0, 2.0, 3.0], name = "input1")
	with tf.name_scope('input2'):
		input2 = tf.Variable(tf.random_uniform([3]), name = "input2")
	output = tf.add_n([input1, input2], name = "add")
	writer = tf.summary.FileWriter(path, tf.get_default_graph())
	# writer1 = tf.summary.FileWriter(r"F:\数据仓库\tensorboard", tf.get_default_graph())
	writer.close()


tensorboardTest()

def tensorboardTest2():
	a = tf.constant(5, name = 'input_a')
	b = tf.constant(3, name = 'input_b')
	# c = tf.mul(a, b, name = 'mul_c')
	d = tf.add(a, b, name = 'add_d')
	# e = tf.add(c, d, name = 'add_e')

	sess = tf.Session()
	output = sess.run(d)
	writer = tf.summary.FileWriter(r'F:\数据仓库\tensorboard', sess.graph)

	writer.close()
	sess.close()


# tensorboardTest2()
