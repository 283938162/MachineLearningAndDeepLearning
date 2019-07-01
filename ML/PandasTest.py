import pandas as pd
import numpy as np

# pandas 关于数据处理的库
# 对Numpy做了封装 更方便

food_info = pd.read_csv("../zData/food_info.csv")
# print(type(food_info))
# <class 'pandas.core.frame.DataFrame'>

# print(food_info.dtypes)  # 打印当前读进数据的dtype

# NDB_No               int64  #整型
# Shrt_Desc           object  # pandas把String类型称之为object
# Water_(g)          float64
# Energ_Kcal           int64
# Protein_(g)        float64
# Lipid_Tot_(g)      float64

# print(food_info)
# Return the first n rows.  n : int, default 5  如果不指定head参数,默认显示前5行
# print(food_info.head())
# first_rows = food_info.head()
# print(first_rows)
#
# print(food_info.tail())
#
# print(food_info.columns)  # 了解这个数据的整体指标

# [5 rows x 36 columns]
# Index(['NDB_No', 'Shrt_Desc', 'Water_(g)', 'Energ_Kcal', 'Protein_(g)',
#        'Lipid_Tot_(g)', 'Ash_(g)', 'Carbohydrt_(g)', 'Fiber_TD_(g)',
#        'Sugar_Tot_(g)', 'Calcium_(mg)', 'Iron_(mg)', 'Magnesium_(mg)',
#        'Phosphorus_(mg)', 'Potassium_(mg)', 'Sodium_(mg)', 'Zinc_(mg)',
#        'Copper_(mg)', 'Manganese_(mg)', 'Selenium_(mcg)', 'Vit_C_(mg)',
#        'Thiamin_(mg)', 'Riboflavin_(mg)', 'Niacin_(mg)', 'Vit_B6_(mg)',
#        'Vit_B12_(mcg)', 'Vit_A_IU', 'Vit_A_RAE', 'Vit_E_(mg)', 'Vit_D_mcg',
#        'Vit_D_IU', 'Vit_K_(mcg)', 'FA_Sat_(g)', 'FA_Mono_(g)', 'FA_Poly_(g)',
#        'Cholestrl_(mg)'],
#       dtype='object')
# print(food_info.shape)  # (8618, 36)

# pandas 取数据 按行取 loc[index]
# pandas uses zero-indexing
# Series object representing the row at index 0.
print(food_info.loc[0])  # loc[row_index]
print(food_info.loc[0:10])  # loc[row_index_start:end] 包含start和end

# objetc - For string values
# int - For integer values
# float - For float values
# datetime - For time values
# bool - For Boolean values


# Return a DataFrame containing the rows at indexs
# 切片
print(food_info.loc[0:3])
# 选取指定行
print(food_info.loc[[1, 2, 10]])

# pandans按列取  df[col_name]

ndb_no = food_info["NDB_No"]
print(ndb_no)

# 选取指定列
columns = ["Vit_D_mcg", "Vit_D_IU"]  # list结构
print(food_info[columns])

# 查看以g结尾的字段
col_names = food_info.columns.tolist()
print(col_names)
gram_columns = []

for c in col_names:
    if c.endswith("(g)"):
        gram_columns.append(c)

gram_df = food_info[gram_columns]  #
print(gram_df.head(3))

# 维度一样时 想对应位置的元素做运算
water_energy = food_info["Water_(g)"] * food_info["Energ_Kcal"]
print(water_energy)

iron_grams = food_info["Iron_(mg)"] / 1000
print(iron_grams)

print(food_info.shape)
# 将新列作为一个特征值加入dataframe  DF[new_col_name] = value
food_info["Iron_(g)"] = iron_grams
print(food_info.shape)

# min max mean 先找到列 再调用相应的函数
min_calories, max_calories, mean_calories = food_info["Energ_Kcal"].min(), food_info["Energ_Kcal"].max(), food_info[
    "Energ_Kcal"].mean()
print(min_calories)
print(max_calories)
print(mean_calories)

# 排序 默认升序
food_info.sort_values("Sodium_(mg)", inplace=True)
print(food_info["Sodium_(mg)"])

# 指定倒排
food_info.sort_values("Sodium_(mg)", inplace=True, ascending=False)
print(food_info["Sodium_(mg)"])

titanic_survival = pd.read_csv("../zData/titanic_train.csv")
print(titanic_survival.head())

age = titanic_survival["Age"]
print(age.loc[0:10])

# 判断前10行里面有没有缺失值
age_is_null = pd.isnull(age.loc[0:10])
print(age_is_null)

# 0     False
# 1     False
# 2     False
# 3     False
# 4     False
# 5      True   # 说明第6行 age的值为NaN 代表这个是个缺失值
# 6     False
# 7     False
# 8     False
# 9     False
# 10    False
# Name: Age, dtype: bool


# 根绝 True False的list获取真实值
# 可以把True或False当成一个索引 来获取真实值!!! flag
age_null_true = age.loc[0:10][age_is_null]
print(age_null_true)

# 5   NaN
# Name: Age, dtype: float64


age_null_count = len(age_null_true)
print(age_null_count)

# 对缺失值的处理  脏数据
# The result of this is that mean_age would be nan. This is because any calculations we do with a null value also result in a null value
# 如果不处理
mean_age = sum(titanic_survival["Age"]) / len(titanic_survival["Age"])
print(mean_age)
# nan


# we have to filter out the missing values before we calculate the mean.
good_ages = titanic_survival["Age"].loc[0:10][age_is_null == False]  # 有意思
correct_mean_age = sum(good_ages) / len(good_ages)
print(correct_mean_age)

#  pandas 自带求均值
# missing data is so common that many pandas methods automatically filter for it
correct_mean_age = titanic_survival["Age"].mean()
print(correct_mean_age)

# 求每个仓位的平均价格 分 1 2 3 等仓
passenger_classes = [1, 2, 3]
fares_by_class = {}
for this_class in passenger_classes:
    # 找出1一等舱的所有乘客所在的行
    pclass_rows = titanic_survival[titanic_survival["Pclass"] == this_class]  # 有意思 又是 布尔索引
    # print(pclass_rows)
    # break
    # 根据拿到的那一行定位目标列
    pclass_fares = pclass_rows["Fare"]
    fare_for_class = pclass_fares.mean()

    # 组装字典
    fares_by_class[this_class] = fare_for_class
print(fares_by_class)

# 计算各个舱位乘客获救的几率


# povit 基准 你要统计的东西要以那个特征为基准
# index tells the method which column to group by
# values is the column that we want to apply the calculation to
# aggfunc specifies the calculation we want to perform  如果未指定 默认求均值
#
# 注:mena  不是mean()

passenger_survival = titanic_survival.pivot_table(index="Pclass", values="Survived", aggfunc=np.mean)
print(passenger_survival)

# Pclass  Survived
# 1       0.629630
# 2       0.472826
# 3       0.242363


# 计算每个舱位乘客的平均年龄
passenger_age = titanic_survival.pivot_table(index="Pclass", values="Age")
print(passenger_age)

# Pclass   Age
# 1       38.233441
# 2       29.877630
# 3       25.140620

# 想同时看一个量跟其他两个量之间的关系.
# 查看登船口与船票和生存几率之间的一个关系

port_stats = titanic_survival.pivot_table(index="Embarked", values=["Fare", "Survived"], aggfunc=np.sum)
print(port_stats)

# Embarked    Fare           Survived
# C         10072.2962        93
# Q          1022.2543        30
# S         17439.3988       217


# dropna 丢掉缺失值
# specifying axis=1 or axis='columns' will drop any columns that have null values

drop_na_columns = titanic_survival.dropna(axis=1)
new_titanic_survival = titanic_survival.dropna(axis=0, subset=["Age", "Sex"])  # 把subset里面包含缺失值的行给去掉
print(new_titanic_survival)

# 如何定位到一个具体的值 而不是一个具体的样本
row_index_83_age = titanic_survival.loc[83, "Age"]

# 排序
# 倒序排序 每一行行首显示的是是这一行原来的index
new_titanic_survival = titanic_survival.sort_values("Age", ascending=False)
print(new_titanic_survival[0:10])
#      PassengerId  Survived  Pclass                                  Name  \
# 630          631         1       1  Barkworth, Mr. Algernon Henry Wilson
# 851          852         0       3                   Svensson, Mr. Johan
# 493          494         0       1               Artagaveytia, Mr. Ramon
# 96            97         0       1             Goldschmidt, Mr. George B
# 116          117         0       3                  Connors, Mr. Patrick
# 672          673         0       2           Mitchell, Mr. Henry Michael
# 745          746         0       1          Crosby, Capt. Edward Gifford
# 33            34         0       2                 Wheadon, Mr. Edward H
# 54            55         0       1        Ostby, Mr. Engelhart Cornelius
# 280          281         0       3                      Duane, Mr. Frank

# 怎么同时实现index也实现排序?
titanic_resetIndex = new_titanic_survival.reset_index(drop=True)  # drop掉原来的index值
print(titanic_resetIndex.loc[0:10])


#     PassengerId  Survived  Pclass                                  Name   Sex  \
# 0           631         1       1  Barkworth, Mr. Algernon Henry Wilson  male
# 1           852         0       3                   Svensson, Mr. Johan  male
# 2           494         0       1               Artagaveytia, Mr. Ramon  male
# 3            97         0       1             Goldschmidt, Mr. George B  male
# 4           117         0       3                  Connors, Mr. Patrick  male
# 5           673         0       2           Mitchell, Mr. Henry Michael  male
# 6           746         0       1          Crosby, Capt. Edward Gifford  male
# 7            34         0       2                 Wheadon, Mr. Edward H  male
# 8            55         0       1        Ostby, Mr. Engelhart Cornelius  male
# 9           281         0       3                      Duane, Mr. Frank  male
# 10          457         0       1             Millet, Mr. Francis Davis  male


# pandas 自定义函数

# 返回第一百行数据
# This function returns the hundredth item from a series
def hundredth_row(column):
    # Extract the hundredth item
    return column.loc[99]


# apply传入的另外一个函数的名字   # 所以有时候mean()  会写成mean

# Return the hundredth item from each column
print(titanic_survival.apply(hundredth_row))


# PassengerId                  100
# Survived                       0
# Pclass                         2
# Name           Kantor, Mr. Sinai
# Sex                         male
# Age                           34
# SibSp                          1
# Parch                          0
# Ticket                    244367
# Fare                          26
# Cabin                        NaN
# Embarked                       S
# dtype: object


# 计算每一个列缺失值的个数

def not_null_count(column):
    column_null = pd.isnull(column)
    null = column[column_null]
    return len(null)


column_null_count = titanic_survival.apply(not_null_count)
print(column_null_count)


# PassengerId      0
# Survived         0
# Pclass           0
# Name             0
# Sex              0
# Age            177
# SibSp            0
# Parch            0
# Ticket           0
# Fare             0
# Cabin          687
# Embarked         2
# dtype: int64


# 离散化
def is_minor(row):
    if row["Age"] < 18:
        return True
    else:
        return False


monors = titanic_survival[0:10].apply(is_minor,axis=1)  # axis ?
print(monors)
#
# 0    False
# 1    False
# 2    False
# 3    False
# 4    False
# 5    False
# 6    False
# 7     True
# 8    False
# 9     True
# dtype: bool


# 连续值做成了一个离散化的操作

def generate_age_label(row):
    age = row["Age"]
    if pd.isnull(age):
        return "unknown"
    elif age < 18:
        return "monor"
    else:
        return "adult"

age_labels = titanic_survival.apply(generate_age_label,axis=1)
print(age_labels)


# 0      adult
# 1      adult
# 2      adult
# 3      adult
# 4      adult
# 5    unknown
# 6      adult
# 7      monor
# 8      adult
# 9      monor
# dtype: object

# 计算获救人数与成年未成年的关系
titanic_survival["age_labels"] = age_labels
age_group_survival = titanic_survival.pivot_table(index="age_labels",values="Survived")
print(age_group_survival)

#
# age_labels  Survived
# adult       0.381032
# monor       0.539823
# unknown     0.293785



# pandas 两种数据结构 一种dataframe结构 另外一种是Series结构 上面讲的都是
# 是df结构  下面是series

#Series (collection of values) 对DF的分解 DF的一行或者一列就是一个Series结构
#DataFrame (collection of Series objects) DF是由一些列的series组成的
#Panel (collection of DataFrame objects)


fandango = pd.read_csv("../zData/fandango_scores.csv")
series_film = fandango['FILM']
print(type(series_film))  # 取的这一列的数据结构称之为 series
print(series_film[0:5]) # 取前5

file_names = series_film.values
print(type(file_names))  # <class 'numpy.ndarray'>
print(file_names) # ndarray结构


# DF由series组成 series又是由ndarray构成 可见DF是做的一个封装
# 还可以指定所以 默认所以都是index 可以指定str类型的做索引















