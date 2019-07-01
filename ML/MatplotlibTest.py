import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 失业率   时间-失业人口占总人口比例
unrate = pd.read_csv('../zData/UNRATE.csv')
print(unrate)

# to_datetime  Convert argument to datetime.
# 可以将xxxx/xx/xx 的形式转换成 xxxx-xx-xx的形式

# 给unrate这一列重写赋值
unrate['DATE'] = pd.to_datetime(unrate['DATE'])  # to_datetime  Convert argument to datetime.
print(unrate.head(12))

# Using the different pyplot functions, we can create, customize, and display a plot. For example, we can use 2 functions to :\n",
# plt.plot()  # 画图操作
# plt.show()  # 显示画图

first_twelve = unrate[0:12]
# plt.plot(x,y)  x代表图的x轴,y代表y轴
plt.plot(first_twelve['DATE'], first_twelve['VALUE'])  # 一个折线图
# While the y-axis looks fine, the x-axis tick labels are too close together and are unreadable\n",
# We can rotate the x-axis tick labels by 90 degrees so they don't overlap\n",
# We can specify degrees of rotation using a float or integer value.\n",

# 对横坐标水平显示的值 做45°的旋转显示 (主要是避免x刻度太密集 导致显示文字重叠 影响可读性)
# rotation 指定角度
plt.xticks(rotation=45)

# 指定 x y 轴代表什么含义 以及图的title
# xlabel(): accepts a string value, which gets set as the x-axis label.\n",
# ylabel(): accepts a string value, which is set as the y-axis label.\n",
# title(): accepts a string value, which is set as the plot title.\n",

plt.xlabel("Month")
plt.ylabel("Unemployment Rate")
plt.title("Monthly Unemployment Trends,1948")
# plt.show()

# 画子图
# 一个图要统计多个指标这些指标没办法放到一个图之后

# fig = plt.figure()  # 默认640 x 480
# 初始化画布
fig = plt.figure(figsize=(6, 6))  # 指定画布大小 长x宽 也就是600x600

# 添加画板
ax1 = fig.add_subplot(4, 3, 1)
ax2 = fig.add_subplot(4, 3, 2)
ax3 = fig.add_subplot(4, 3, 6)

# 画板画图
ax1.plot(np.random.randint(1, 5, 5), np.arange(5))
ax2.plot(np.arange(10) * 3, np.arange(10))
ax3.plot(np.array([1, 3, 5, 7]), np.array([2, 4, 6, 8]))

# 显示
# plt.show()

# 画多条折线
unrate['MONTH'] = unrate['DATE'].dt.month
fig = plt.figure()
plt.plot(unrate[0:12]['MONTH'], unrate[0:12]['VALUE'], c='red')
plt.plot(unrate[12:24]['MONTH'], unrate[12:24]['VALUE'], c='blue')
plt.plot(unrate[24:36]['MONTH'], unrate[24:36]['VALUE'], c='yellow')

# plt.show()


# 使用循坏来实现

fig = plt.figure(figsize=(10, 10))
colors = ['red', 'blue', 'green', 'orange', 'black']
for i in range(5):
    start_index = i * 12
    end_index = (i + 1) * 12
    subset = unrate[start_index:end_index]

    # 为每根线添加label
    label = str(1948 + i)
    plt.plot(subset['MONTH'], subset['VALUE'], c=colors[i], label=label)

# best 自动放到一个合适的位置 也可以自定义
print(help(plt.legend))
plt.legend(loc='best')

# plt.show()

# 使用书电影评分的数据集 来画柱状图和散点图 首先读入数据集
reviews = pd.read_csv("../zData/fandango_scores.csv")
# 取列集合
cols = ['FILM', 'RT_user_norm', 'Metacritic_user_nom', 'IMDB_norm', 'Fandango_Ratingvalue', 'Fandango_Stars']
norm_reviews = reviews[cols]

# 1取第一行
print(norm_reviews[:1])

# The Axes.bar() method has 2 required parameters, left and height. \n",
# We use the left parameter to specify the x coordinates of the left sides of the bar. \n",
# We use the height parameter to specify the height of each bar\n",

num_cols = ['RT_user_norm', 'Metacritic_user_nom', 'IMDB_norm', 'Fandango_Ratingvalue', 'Fandango_Stars']

# ix 什么意思
# loc——通过行标签索引行数据
# iloc——通过行号索引行数据
# ix——通过行标签或者行号索引行数据（基于loc和iloc 的混合）
# 同理，索引列数据也是如此！

# pandas与numpy最大的区别是自带索引列 使用vaules属性可以取值
bar_heights = norm_reviews.ix[0, num_cols].values  # 等价与 norm_reviews.loc[0,num_cols]
print(bar_heights)

# 确定各个柱离0值的距离
bar_positions = np.arange(5) + 0.75
tick_position = range(1, 6)  # 左闭右开 取1到5

fig, ax = plt.subplots()

# 0.3 代表柱的宽度
ax.bar(bar_positions, bar_heights, 0.3)  # 默认是竖着画
# ax.barh(bar_positions,bar_heights,0.3)  # 当然也可以直着画 只需要改此一处即可

ax.set_xticks(tick_position)  # Set the x ticks with list of *ticks*  ticks记号 标号
ax.set_xticklabels(num_cols, rotation=45)

ax.set_xlabel("Rating Source")
ax.set_ylabel("Average Rating")
ax.set_title("Average User Rating For Avengers: Age of Ultron (2015)")

# plt.show()

# 散点图
fig, ax = plt.subplots()
ax.scatter(norm_reviews['Fandango_Ratingvalue'], norm_reviews['RT_user_norm'])
ax.set_xlabel("Fandango")
ax.set_ylabel("Rotten Tommatoes")
plt.show()
