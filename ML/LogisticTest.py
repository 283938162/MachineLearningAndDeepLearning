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


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 画图展示 sigmoid函数
nums = np.arange(-10, 10, step=1)  # create a vector containing 20 equally spaced values from -10 to 10
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(nums, sigmoid(nums), 'r')
plt.show()
