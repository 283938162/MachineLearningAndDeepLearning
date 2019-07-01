import numpy as np

# numpy 关于矩阵运算的库
#https://www.jianshu.com/p/358948fbbc6e

# 一维向量 二维矩阵 多维张量
# 数组 集合
# The numpy.array() function can take a list or list of lists as input. When we input a list, we get a one-dimensional array as a result
vector = np.array([1, 2, 3, 4]) # []为啥存在 语法意义是 array() 参数是一个list
print("------------------------------")
# Numpy 值判断 ==
# it will compare the second value to each element in the vector\n
# If the values are equal, the Python interpreter returns True; otherwise, it returns False

# 如果第一个ndarray进行一个等值操作 就相当于对ndarray中的每个值都进行一次等值判断 矩阵的优势 不需要for循环 为啥矩阵效率高?可见一斑!
print(vector == 2)  # [False  True False False]  matrix 一样的道理
print(vector[vector == 2])  # 这个用法有点乖  直接拿一个布尔值当索引  结果返回就是一个真实值

# When we input a list of lists, we get a matrix as a result
matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(vector)
print(matrix)

# shape #
# we can use ndarray shape property to figure out how many elements are in the array.
print(vector.shape)
# For matrices, the shape property contains a tuple with 2 elements
print(matrix.shape)

# axis
print(vector.sum())

# The axis dictates which dimension we perform the operation on
# 1 means that we want to perform the operation on each row, and 0 means on each column
print(matrix.sum(axis=1))
print(matrix.sum(axis=0))

# dtype   astype()#
# Each value in a Numpy array has to the same data type
# NumPy will automatically figure out an appropriate data type when reading in data or converting lists to arrays.
# You can check the data type of a NumPy array using the dtype property.

print(vector.dtype)
print(vector.astype(float).dtype)
print(vector.astype(float))

members = np.array([1, 2, 3, 4.0])
print(members.dtype)  # float64

members = np.array([1, 2, 3, "4"])
print(members.dtype)  # <U11

# 读取外部数据源转换成 numpy.ndarray
# PyCharm   crtl + Q 查看函数文档
# 第一行是标识头 处理数据的时候需要跳过
world_clcohol = np.genfromtxt("../zData/world_alcohol.txt", delimiter=",", dtype=str, skip_header=1)
print(type(world_clcohol))
print(world_clcohol)

print("------------------------------")

# 想取某个值 index
# 通过索引去矩阵中的值  行集合列的index都是从0开始的  注意索引是  逗号 隔开的!
print(world_clcohol[2, 4])  # 0.5
print(world_clcohol[3, 2])  # Cte d'Ivoire

# 想取 某行某列某块  slice
# 切片  slice #   注意切片 分割符 是 冒号 : 或者 :+,
# 逗号前为行,后为列,如果逗号前后只有冒号.表示取所有行/列;如果是一维数组是没有逗号隔开的,只有行;
# 取值规则:左闭右开 #

print(world_clcohol[:, 0])  # 取第一列
print(world_clcohol[0:2, 0:2])

# 怎么取指定列?  比如想取第1行和第二行 , 第0列 与 第2列与第三列

print(world_clcohol[1:3, [[0], [2], [3]]])
print(world_clcohol[1:3, [[0], [2], [3]]].shape)
print(world_clcohol[1:3, [[0], [2], [3]]].reshape(2, 3))  #

# np.arrange  reshape   构造与变形

print(np.arange(15))
a = np.arange(15).reshape(3, 5)
print(a)
print(a.shape)
# the number of axis (dimensions) of the array
print(a.ndim)
print(a.dtype.name)
print(a.size)

# np.zeros 初始化一个空矩阵
# 对于二维矩阵 zeros参数是一个元组 如果没有指定类型,默认是float类型的'
# 无论是zeros还是ones random 参数都是元组  代表每一个维度
print(np.zeros((3, 4)))

# np.ones 构造单位矩阵 并指定元素类型
print(np.ones((2, 3, 4), dtype=np.int32))

# To create sequences of numbers  序列
# np.arange(start,end,step)

print(np.arange(10, 30, 5))
print(np.arange(1, 2, 0.2))

# np.random 随机初始化
# np.random 进入random模块  之后在调用random函数 函数参数跟zeros一样
# 如果是一个二维矩阵 参数一是一个元组 默认-1到1区间的值
print(np.random.random((2, 3)))  # 2行3列的矩阵

# np.linspace    np.linspace(start,end,nums)  在start和end区间 平均取nums个值
print(np.linspace(0, 10, 10))

# Numpy arrays 数学运算
a = np.array([5, 6, 7, 8])
b = np.arange(4)

print(a)
print(b)

# 对于两个array运算是对应位置执行互操作
print(a + b)
print(a - b)

# array中的每个元素对单个元素执行
print(a - 1)

print(b ** 2)  # ** 平方
print(b > 2)

# The matrix product can be performed using the dot function or method

A = np.array([
    [1, 1],
    [0, 1]
])

B = np.array([
    [2, 0],
    [3, 4]
])

# *成
print(A * B)  # 对应位置的相乘
# [[2 0]
#  [0 4]]

# 点乘
print(A.dot(B))  # 矩阵的乘法 笛卡尔积 #行与列想乘
print(np.dot(A, B))  # 矩阵的乘法 笛卡尔积 #行与列想乘
# [[5 4]
#  [3 4]]


# exp，高等数学里以自然常数e为底的指数函数
# Exp：返回e的n次方，e是一个常数为2.71828
# Exp 函数 返回 e（自然对数的底）的幂次方。
# 次幂exp  开根号sqrt
C = np.arange(3)
print(C)

print(np.exp(C))
print(np.sqrt(C))

# 矩阵操作

# np.floor Return the floor of the input  向下取整

a = 10 * np.random.random((3, 4))
af = np.floor(a)  # 注 floor 不是 float
print(a)
print(af)
# [[5.03404289 5.4316545  9.45317135 1.3562172 ]
#  [4.83825453 7.09626535 1.06119177 6.99280841]
#  [6.32773684 5.5438614  2.94784341 2.83658639]]
# [[5. 5. 9. 1.]
#  [4. 7. 1. 6.]
#  [6. 5. 2. 2.]]


# ravel()   ravel()方法 压平array数组
# flatten the array  拉平成一个向量

print(af.ravel())  # [1. 4. 9. 1. 9. 4. 6. 7. 4. 8. 3. 8.]

# 直接给ndarray的shape属性赋值就等价于 reshape
af.shape = (6, 2)
print(af)

# T 转置 行和列变换
print(af.T)

# If a dimension is given as -1 in a reshaping operation, the other dimensions are automatically calculated
# 对于一个二维矩阵 一旦确定了一个维度 其实另外一个维度也是已经确定了的  可以直接使用-1代替
# 对于三维,确定两维 另外一维也可以确定
print(af.reshape(3, -1))
print(af.reshape(2, -1))

print(af.reshape(2, 2, -1))

#############
# 矩阵的拼接 #
#############

a = np.floor(10 * np.random.random((2, 2)))
b = np.floor(10 * np.random.random((2, 2)))
print(a)
print(b)

# hstack  Stack arrays in sequence horizontally (column wise).  横竖 水平方向 horizontally
# 原来a 有两个特征 b 也有两个特征 我现在想要组合在一起 还是两行 有4个特征
print(np.hstack((a, b)))  # 查看其函数文档 参数值是一个tuple
print(np.stack((a, b)))
# [[5. 0. 7. 1.]
#  [7. 7. 8. 9.]]

# vstack     Stack arrays in sequence vertically (row wise).   竖着 垂直方向 vertically
# 竖着去拼 增加样本数量
# print(np.vstack((a,b)))
# [[4. 0.]
#  [6. 5.]
#  [8. 2.]
#  [6. 4.]]


# stack 指定axis 与 hstack和vstack 效果是一样的
# todo

# 拆分  hsplit  水平方向拆  vsplit 垂直方向拆
a = np.floor(10 * np.random.random((2, 12)))
print(a)

# [[7. 2. 2. 9. 9. 0. 9. 6. 6. 8. 7. 5.]
#  [8. 2. 5. 4. 8. 2. 8. 7. 7. 0. 6. 0.]]


print(np.hsplit(a, 3))  # a 是待切分的array
# [array([[7., 2., 2., 9.],
#        [8., 2., 5., 4.]]), array([[9., 0., 9., 6.],
#        [8., 2., 8., 7.]]), array([[6., 8., 7., 5.],
#        [7., 0., 6., 0.]])]

print(np.hsplit(a, (3, 4)))  # 传元组表示指定一个切分的位置 切第三列  随机
# [array([[7., 2., 2.],
#        [8., 2., 5.]]),
# array([[9.],
#        [4.]]),
# array([[9., 0., 9., 6., 6., 8., 7., 5.],
#        [8., 2., 8., 7., 7., 0., 6., 0.]])]


b = np.floor(10 * np.random.random((12, 2)))
print(b)
print(np.vsplit(b, 3))  # 平均切3分
# [[8. 9.]
#  [8. 9.]
#  [7. 9.]
#  [5. 6.]
#  [8. 0.]
#  [4. 8.]
#  [3. 7.]
#  [6. 8.]
#  [9. 2.]
#  [2. 3.]
#  [0. 0.]
#  [9. 0.]]
# [array([[8., 9.],
#        [8., 9.],
#        [7., 9.],
#        [5., 6.]]),
#  array([[8., 0.],
#        [4., 8.],
#        [3., 7.],
#        [6., 8.]]),
#  array([[9., 2.],
#        [2., 3.],
#        [0., 0.],
#        [9., 0.]])]


# Simple assignments make no copy of array objects or of their data.
a = np.arange(12)
b = a

# a and b are two names for the same ndarray object
# 如果对b进行修改 a也会随之改变  判断两个对象是否相等的方法 is   id(object)
# ab虽然名字不同但是却指向内存的同一块空间
print(b is a)

b.shape = 3, 4
print(a.shape)
print(id(a))
print(id(b))

# True
# (3, 4)
# 1609705302656
# 1609705302656

# 浅赋值  view  #The view method creates a new array object that looks at the same data.
# 虽然ab指向内存空间地址不同但是公用的是一套值.
c = a.view()
print(c is a)
c.shape = 2, 6
print(a.shape)
c[0, 4] = 1234  # 指定元素赋值  , 隔开只是取值 不切片
print(a)
print(id(a))
print(id(c))

# False
# (3, 4)
# [[   0    1    2    3]
#  [1234    5    6    7]
#  [   8    9   10   11]]
# 2211427887744
# 2211427888784

# 还特么撇不清关系了
# copy
# The copy method makes a complete copy of the array and its data.\n
# 位置不一样 值也不一样
d = a.copy()
print(d is a)

d[0, 0] = 9999
print(d)
print(a)
# False
# [[9999    1    2    3]
#  [1234    5    6    7]
#  [   8    9   10   11]]
# [[   0    1    2    3]
#  [1234    5    6    7]
#  [   8    9   10   11]]


# 排序和索引
# 找到每一列的最大值 返回其索引
data = np.sin(np.arange(20).reshape(5, 4))
print(data)
# Indexes of the maximum values along an axis.
# axis=0 行方向 实际上按列来推进的
# axis=1 列方向 实际上按行来推进的
ind = data.argmax(axis=0)
print(ind)

print(data.shape)
print(data.shape[1])
print(range(data.shape[1]))

data_max = data[ind, range(data.shape[1])]  # 矩阵的元素操作,两个参数一个是向量 一个是range,取相应的index构建一个联合索引
print(data_max)

# [[ 0.          0.84147098  0.90929743  0.14112001]
#  [-0.7568025  -0.95892427 -0.2794155   0.6569866 ]
#  [ 0.98935825  0.41211849 -0.54402111 -0.99999021]
#  [-0.53657292  0.42016704  0.99060736  0.65028784]
#  [-0.28790332 -0.96139749 -0.75098725  0.14987721]]
# [2 0 3 1]
# (5, 4)
# 4
# range(0, 4)
# [0.98935825 0.84147098 0.99060736 0.6569866 ]


# tile 扩展
a = np.arange(0, 40, 10)
print(a)
b = np.tile(a, (2, 2))  # 对a进行一个扩展,(2,2)是一个shape 行和列分别扩展为原来的2倍
print(b)
# [ 0 10 20 30]
# [[ 0 10 20 30  0 10 20 30]
#  [ 0 10 20 30  0 10 20 30]]

# 排序
a = np.array([[4, 3, 5], [1, 2, 1]])
print(a)
b = np.sort(a, axis=1)  # axis=1 列方向 实际上按行来推进的
print(b)
b = np.sort(a, axis=0)
print(b)

a = np.array([4, 3, 1, 2])
i = np.argsort(a)  # 返回向量重小到大值的索引
print(i)

print(a[i])  # 向量做变量 批量操作数据 -- flag 基础操作
# [[4 3 5]
#  [1 2 1]]

# [[3 4 5]
#  [1 1 2]]
# [[1 2 1]
#  [4 3 5]]

# [2 3 1 0]
# [1 2 3 4]


# 棋盘矩阵
z = np.zeros((8, 8), dtype=int)
print(z)
# [[0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0]]

# 其中1、3、5、7行&&0、2、4、6列的元素置为1   1 ,3，5，7列&&0,2,4,6行也是1
z[1::2, ::2] = 1
z[::2, 1::2] = 1
print(z)

# [[0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]
#  [0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]
#  [0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]
#  [0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]]


# min()与 max()函数

m = 10 * np.random.random((3, 3))
print(m)
min, max = m.min(), m.max()
print(min)
print(max)

# [[0.59624115 0.07872599 2.78363442]
#  [8.91227078 6.80103315 2.40046635]
#  [1.72074955 7.63249044 9.08252408]]
# 0.07872599198319175
# 9.08252407619325


#归一化，将矩阵规格化到0～1，即最小的变成0，最大的变成1，最小与最大之间的等比缩放
#这种归一化的计算方式有待考证
z=10 * np.random.random((5,5))
print(z)
zmin,zmax = z.min(),z.max()
z = (z-zmin)/(zmax -zmin)
print(z)


#生成0~10之间均匀分布的11个数，包括0和10
# 离散 均匀分布
z = np.linspace(0,10,11,endpoint=True,retstep=True)
print(z)

# (array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.]), 1.0)

#交换矩阵的其中两行

a = np.array([[1,2,3],[2,3,4],[1,6,5], [9,3,4]])   # 元素嵌套 a[1] = [A,B,C] A=[1,2,3]
tmp = np.copy(a[1])
a[1] = a[2]
a[2] = tmp
# array([[1, 2, 3],
#        [1, 6, 5],
#        [2, 3, 4],
#        [9, 3, 4]])

# 其实有更简单的方法  A[[i, j], :] = A[[j, i], :] # 实现了第i行与第j行的互换
a = np.arange(6).reshape(3,2)
print(a)
a[[0,1]] = a[[1,0]]
print(a)

# [[0 1]
#  [2 3]
#  [4 5]]
# [[2 3]
#  [0 1]
#  [4 5]]


#找出数组中与给定值最接近的数  abs 绝对值
# a.argmin(axis=None, out=None) : Return indices of the minimum values along the given axis of `a`.
# [indices]	index 的复数形式之一
z = np.array([[0,1,2,3],[4,5,6,7]])
a = 5.1
print(np.abs(z-a).argmin())
# 5

# 判断二维矩阵中有没有一整列数为0？
# randint(low, high=None, size=None, dtype='l'): Return random integers from `low` (inclusive) to `high` (exclusive).
# a.any(axis=None, out=None, keepdims=False): Returns True if any of the elements of `a` evaluate to True.
z = np.random.randint(0,3,(2,10))#随机生产成一个2行10列的二维矩阵,大小在0到3之间,左闭右开原则.
print(z)
print(z.any(axis=0))  # 如果对应一列都为0 则false
#
# [[2 2 1 1 1 0 2 1 2 2]
#  [1 1 1 1 0 0 0 1 2 1]]
# [ True  True  True  True  True False  True  True  True  True]
help(np.random.randint)