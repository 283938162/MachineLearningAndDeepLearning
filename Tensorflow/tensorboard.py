import tensorflow as tf

path = r'F:\DataResposity\tensorboard\logfile'

# 定义一个简单的计算图，实现向量加法的操作。
input1 = tf.constant([1.0, 2.0, 3.0], name = 'input1')
input2 = tf.Variable(tf.random_uniform([3]), name = 'input2')
output = tf.add_n([input1, input2], name = 'add')


print('input1 = ',input1)
print('input2 = ',input2)
print('output = ',output)

# 生成一个写日志的writer，并将当前的tensorflow计算图写入日志。
# tensorflow提供了多种写日志文件的API
writer = tf.summary.FileWriter(path, tf.get_default_graph())
writer.close()


def tensorboardTest():
	with tf.name_scope('input1'):
		input1 = tf.constant([1.0, 2.0, 3.0], name = "input1")
	with tf.name_scope('input2'):
		input2 = tf.Variable(tf.random_uniform([3]), name = "input2")
	output = tf.add_n([input1, input2], name = "add")
	writer = tf.summary.FileWriter(path, tf.get_default_graph())
	# writer1 = tf.summary.FileWriter(r"F:\数据仓库\tensorboard", tf.get_default_graph())
	writer.close()


# tensorboardTest()

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
