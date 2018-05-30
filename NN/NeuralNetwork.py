import numpy as np

"""
python 实现的一个简单的神经网络

tanh()以及其导数(derivative)
https://baike.baidu.com/item/tanh/19711736?fr=aladdin
"""


# 双曲函数
def tanh(x):
	return np.tanh(x)


# 因为更全权重的时候使用到双曲函数的导数，所以这里也直接定义
def tanh_deriv(x):
	# return 1 - np.tanh(x)*np.tanh(x) #两种写法等价
	return 1 - np.tanh(x) ** 2


def logistic(x):
	return 1 / (1 + np.exp(-x))  # 以e为底的对数是 np.exp(x)


def logistic_deriv(x):
	return logistic(x) * (1 - logistic(x))


class NeuralNetwork:
	def __init__(self, layers, activation = 'tanh'):  # 构造函数 self相当于c中的指针，java中的this
		"""
		:param layers:  A list containing the number of units in each layer
						should be at two valules
		:param activation:The activation function to be used. can be 'logistic' or 'tanh'
		"""
		if activation == 'logistic':
			self.activation = logistic
			self.activation_deriv = logistic_deriv
		elif activation == 'tanh':
			self.activation = tanh
			self.activation_deriv = tanh_deriv

		self.weights = []
		for i in range(1, len(layers) - 1):  # i对层数进行循环  这里i是第二层 i-1 与前一层神经元之间的w  i+1与后一层
			self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1) - 1)) * 0.25)  # -0.25<w<0.25
			self.weights.append((2 * np.random.random((layers[i] + 1, layers[i + 1] + 1) - 1)) * 0.25)  # -0.25<w<0.25
			print(str(self.weights))

	def fit(self, x, y, learning_rate = 0.2, epochs = 10000):
		"""
		:param self 指引当前类的指针
		:param x: 数据集 二维矩阵   多少行就有多少实例，每一列都是一个特征维度 特征值
		:param y: class lable 分类标签
		:param learn_rate: 学习率 大小一般在01之间，太大太小都容易错误极值
		:param epochs: 迭代次数
		:return:
		"""
		x = np.atleast_2d(x)  # 确认输入实例至少是二维的
		temp = np.ones([x.shape[0], x.shape[1] + 1])
		temp[:, 0:-1] = x  # adding the bias unit to input layer
		x = temp
		y = np.array(y)

		for k in range(epochs):
			i = np.random.randint(x.shape[0])  # x.shape[0] is the number of the trainingset samples
			a = [x[i]]  # choose a sample randomly to train the model
			print(str(a))
			for l in range(len(self.weights)):
				# print("a["+str(l)+"]; "+str(a[l])+"  WEIGHT "+str(self.weights[l])+str(len(self.weights)))
				a.append(self.activation(np.dot(a[l], self.weights[l])))
			error = y[i] - a[-1]
			deltas = [error * self.activation_deriv(a[-1])]

			for l in range(len(a) - 2, 0, -1):
				deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))
			deltas.reverse()

			for i in range(len(self.weights)):
				layer = np.atleast_2d(a[i])
				delta = np.atleast_2d(deltas[i])
				self.weights[i] += learning_rate * layer.T.dot(delta)

	def predict(self, x):
		x = np.array(x)
		temp = np.ones(x.shape[0] + 1)
		temp[0:-1] = x
		a = temp
		for l in range(0, len(self.weights)):
			a = self.activation(np.dot(a, self.weights[l]))
		return a
