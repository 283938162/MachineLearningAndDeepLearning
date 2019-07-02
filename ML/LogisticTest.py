# 我们将建立一个逻辑回归模型来预测一个学生是否被大学录取。假设你是一个大学系的管理员，
# 你想根据两次考试的结果来决定每个申请人的录取机会。你有以前的申请人的历史数据，你可以用它作为逻辑回归的训练集。
# 对于每一个培训例子，你有两个考试的申请人的分数和录取决定。为了做到这一点，
# 我们将建立一个分类模型，根据考试成绩估计入学概率。

# 数据分析三大件
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pdData = pd.read_csv('../zData/LogiReg_data.txt', header=None, names=['Exam 1', 'Exam 2', 'Admitted'])

# 傻b了 这不是Jupyter  需要print才有数据打印
print(pdData.head())

# 查看数据维度
print(pdData.shape)

# 另外一中读取文本的方法
# data = pd.read_table('../zData/LogiReg_data.txt',header=None, encoding='gb2312', sep=',')
# header=None:没有每列的column name，可以自己设定
# encoding='gb2312':其他编码中文显示错误
# sep=',': ','隔开
# data.head()


#       Exam 1     Exam 2  Admitted
# 0  34.623660  78.024693         0
# 1  30.286711  43.894998         0
# 2  35.847409  72.902198         0
# 3  60.182599  86.308552         1
# 4  79.032736  75.344376         1

positive = pdData[pdData['Admitted'] == 1]  # 取出被录取的样本
# print(positive)
negative = pdData[pdData['Admitted'] == 0]  # 取出未被录取的样本

# figsize 指定画图域的长和宽
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=30, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=30, c='r', marker='x', label='Not Admitted')
ax.legend()

ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')


# plt.show()

# 目标：建立分类器（求解出三个参数 theta_0   theta_1  theta_2 ,
#
# 设定阈值，根据阈值判断录取结果   比如大于0.5 被录取 小于则未被录取
#
# ### 要完成的模块
# sigmoid : 映射到概率的函数
# model : 返回预测结果值
# cost : 根据参数计算损失
# gradient: 计算每个参数的梯度方向
# descent : 进行参数更新
# accuracy: 计算精度"

# 对应上面案例 theta0是偏置项  theta1对应exam1考试成绩 theta2 是exam2成绩


# 映射到概率的函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 返回预测结果值  预测函数 也就是h theta x
def model(X, theta):
    return sigmoid(np.dot(X, theta.T))


# 根据参数计算损失
# 参数  X 数据  y 标签 theta 参数
def cost(X, y, theta):
    left = np.multiply(-y, np.log(model(X, theta)))
    right = np.multiply(1 - y, np.log(1 - model(X, theta)))
    # 求平均损失
    return np.sum(left - right) / (len(X))


# 计算每个参数的梯度方向 计算梯度 ravel()  降维 (矩阵降到向量)
def gradient(X, y, theta):
    grad = np.zeros(theta.shape)  # 有多少个特征/维度 就有多少个梯度  zero是为了初始化占位
    error = (model(X, theta) - y).ravel()  # 将符号提到里面 所以是 thatax - y

    # 求偏导是先对 theta0 求偏导 然后在对theta1 和 theta2求偏导 同样是对特征值求偏导
    for j in range(len(theta.ravel())):  # for each parameter
        term = np.multiply(error, X[:, j])  # 所有样本的第j列
        grad[0, j] = np.sum(term) / len(X)  # 更新梯度

    return grad


# 梯度下降 进行参数更新
#
# data 元素数据 矩阵类型
# theta 权重参数
# batchSize 批次
# stopType 停止类型
# thresh 阈值
# alpha  学习率

import time


def descent(data, theta, batchSize, stopType, thresh, alpha):
    # 梯度下降求解
    init_time = time.time()
    i = 0  # 迭代次数
    k = 0  # batch

    X, y = shuffleData(data)
    grad = np.zeros(theta.shape)  # 梯度初始化 占位
    costs = [cost(X, y, theta)]  # 损失值 初始化损失值集合 其中 X,y是确定的 theta是迭代更新的

    while True:
        grad = gradient(X[k:k + batchSize], y[k:k + batchSize], theta)
        k += batchSize  # 取batch数量个数据
        if k >= n:  # 这个是为啥?
            k = 0
            X, y = shuffleData(data)  # 重新洗牌

        theta = theta - alpha * grad  # 梯度下降->参数更新
        costs.append(cost(X, y, theta))  # 计算新的损失值 append 向集合中添加元素
        i += 1

        if stopType == STOP_ITER:
            value = i
        elif stopType == STOP_COST:
            value = costs
        elif stopType == STOP_GRID:
            value = grad

        if stopCriterion(stopType, value, thresh): break
    return theta, i - 1, costs, grad, time.time() - init_time


# accuracy
def accuracy():
    return


# 画图展示 sigmoid函数
nums = np.arange(-10, 10, step=1)  # create a vector containing 20 equally spaced values from -10 to 10
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(nums, sigmoid(nums), 'r')
# plt.show()


# in a try / except structure so as not to return an error if the block si executed several times
# Insert column into DataFrame at specified location.
pdData.insert(0, 'Ones', 1)

print(pdData.head())

# 对数据进行分割
# set X (training data) and y (target variable)

# convert the Pandas representation of the data to an array useful for further computations
orig_data = pdData.as_matrix()
# print(orig_data)

# 取出列数 这个根据通用性  尽量不要直接去数 3 ,5之类的
col_nums = orig_data.shape[1]

X = orig_data[:, 0:col_nums - 1]
print(X[:5])

y = orig_data[:, col_nums - 1: col_nums]
# print(y[:5])

# convert to numpy arrays and initalize the parameter array theta
# X = np.matrix(X.values)\n",
# y = np.matrix(data.iloc[:,3:4].values) #np.array(y.values)\n",

theta = np.zeros([1, 3])
# print(theta)
# print(theta.T)

print(X.shape)
print(y.shape)
print(theta.shape)

# 0.6931471805599453
print(cost(X, y, theta))

# 比较三种不同梯度下降方法
STOP_ITER = 0  # 根据迭代次数
STOP_COST = 1  # 根据损失值 相近两次目标函数损失值相差很小 也可以判定停止
STOP_GRID = 2  # 梯度变化很小了


# 返回结果是boolean
def stopCriterion(type, value, threshold):
    # 设定三种不同的停止策略
    # (1) 指定停止次数
    if type == STOP_ITER:
        return value > threshold
    # (2) 指定阈值
    elif type == STOP_COST:
        return abs(value[-1] - value[-2] < threshold)

    # (3) 指定阈值 np.linalg.norm 求范数 矩阵整体元素平方和开根号，不保留矩阵二维特性
    elif type == STOP_GRID:
        return np.linalg.norm(value) < threshold


import numpy.random


# 洗牌  为了使模型的泛化能力更强 首先将数据的顺序打乱
# numpy提供了一个shuffle操作
def shuffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:, 0:cols - 1]
    y = data[:, cols - 1:]
    return X, y


# 功能型函数 便于画图展示
# 根据参数选择梯度下降的方式和停止策略

# orig_data 数据矩阵
# theta  权重参数 使用zeros初始化
# n batchSize
# STOP_ITER 停止策略
# thresh 阈值
# alpha  学习率


def runExpe(data, theta, batchSize, stopType, thresh, alpha):
    # 核心代码 执行一次梯度更新(执行一次梯度下降得到一个最终结果)
    theta, iter, costs, grad, dur = descent(data, theta, batchSize, stopType, thresh, alpha)

    name = "Original" if (data[:, 1] > 2).sum() > 1 else "Scaled"
    name += "data - learning rate: {} - ".format(alpha)
    if batchSize == n:
        strDescType = "Gradient"
    elif batchSize == 1:
        strDescType = "Stochastic"
    else:
        strDescType = "Mini - batch({})".format(batchSize)

    name += strDescType + " descent - Stop: "
    if stopType == STOP_ITER:
        strStop = "{} iterations".format(thresh)
    elif stopType == STOP_COST:
        strStop = "costs change < {}".format(thresh)
    else:
        strStop = "gradient norm < {}".format(thresh)
    name += strStop

    print("***{}\nTheta: {} - Iter:{} -Last cost: {:03.2f} - Duration: {:03.2f}s".format(name, theta, iter, costs[-1],
                                                                                         dur))
    # 画图
    fig, ax = plt.subplots(figsize=(12, 4))
    # 损失值的折线图
    ax.plot(np.arange(len(costs)), costs, 'r')

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Cost")
    ax.set_title(name.upper() + ' - Error vs. Iteration')
    # plt.show()
    return theta


# 设定迭代次数 停止策略是 STOP_ITER 迭代5000次停止
# 选择的梯度下降方法是基于所有样本的

n = 100  # 因为数据样本一共就有100个 也就是对整体进行一个梯度下降 基于所有样本做一个梯度下降
# runExpe(orig_data, theta, n, STOP_ITER, thresh=5000, alpha=0.000001)

# 最终cost达到0.63

# 根据损失值停止  设定阈值1E-6,差不多需要110000次迭代
# runExpe(orig_data, theta, n, STOP_COST, thresh=0.000001, alpha=0.001)


# 根据梯度变化停止 设定阈值0.05 差不多需要40000次迭代
# runExpe(orig_data, theta, n, STOP_GRID, thresh=0.05, alpha=0.001)


## 对比不同的梯度下降方法
# runExpe(orig_data, theta, 1, STOP_ITER, thresh=5000, alpha=0.001)  # 只迭代一个样本
# 结果不收敛 不稳定,一些列的上下浮动 ; 如何优化呢? 增加迭代次数 同时降低 学习率
# runExpe(orig_data, theta, 1, STOP_ITER, thresh=15000, alpha=0.000002)  # 只迭代一个样本
# 这个效果好多了,但是收敛效果没那么多


# Mini-batch descent
# runExpe(orig_data, theta, 16, STOP_ITER, thresh=15000, alpha=0.001)
# 结果不收敛 不稳定,一些列的上下浮动,调小学习率可以实现较好的收敛效果

# 但是可以尝试使用下面的方法
#
# 浮动仍然比较大，我们来尝试下对数据进行标准化\n",
# 将数据按其属性(按列进行)减去其均值，然后除以其方差。最后得到的结果是，对每个属性/每列来说所有数据都聚集在0附近，方差值为1"

from sklearn import preprocessing as pp

scaled_data = orig_data.copy()
scaled_data[:, 1:3] = pp.scale(orig_data[:, 1:3])


# runExpe(scaled_data, theta, 16, STOP_ITER, thresh=15000, alpha=0.001)

# 收敛结果很好  它好多了！原始数据，只能达到达到0.61，而我们得到了0.38个在这里
# 所以对数据做预处理是非常重要的"

# 如果数据结果浮动很明显 (1) 数据预处理 (2) 调整学习率


# runExpe(scaled_data, theta, 16, STOP_GRID, thresh=0.002*2, alpha=0.001)
# 预处理效果不错


# 精度
# 设定阈值
def predict(X, theta):
    return [1 if x > 0.5 else 0 for x in model(X, theta)]


scaled_X = scaled_data[:, :3]
y = scaled_data[:, 3]
predictions = predict(scaled_X, theta)
print(predictions)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print("accuracy = {0}%".format(accuracy))














