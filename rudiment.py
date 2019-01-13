#机器学习入门 http://www.mamicode.com/info-detail-2239720.html

import pandas as pd #数据分析、处理
import numpy as np #科学计算包
import matplotlib.pyplot as plt #画图

from sklearn.datasets import load_iris
iris_dataset = load_iris() #sklearn已经整理了Iris数据集，使用load_iris函数可以直接下载，使用；

# print(iris_dataset)#发现数据集整理成了一个大字典；

print("Keys of iris_dataset:\n{}".format(iris_dataset.keys()))#有5个键；我们逐个看看 output:

print("DESCR of iris_dataset:\n{}".format(iris_dataset["DESCR"]))#数据集的描述信息；
#我们知道有150条记录（每类50条，一共有3类）；
#属性：
#4个数值型，用来预测的属性：sepal 长、宽；petal长、宽
#一个类别标签：三类Setosa，Versicolour，Virginica；

print('data of iris_dataset:\n{}'.format(iris_dataset['data'][:5]))#看数据的前5条；
print('shape of iris_dataset:\n{}'.format(iris_dataset['data'].shape))#data形状：150*4；150条记录，没错
print('target_names of iris_dataset:\n{}'.format(iris_dataset['target_names']))#3类

print('target of iris_dataset:\n{}'.format(iris_dataset['target'][:5]))#全是0;数据是按照类别进行排序的；全是0，全是1，全是2；
print('target shape of iris_dataset:\n{}'.format(iris_dataset['target'].shape))#说明有150个标签，一维数组；

#划分一下数据集，方便对训练后的模型进行评测？可信否？
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],
                                                 test_size=0.25,random_state=0)
#第一个参数：数据；第二个参数：标签；第三个参数：测试集所占比例；第四个参数：random_state=0：确保无论这条代码，运行多少次，
#产生出来的训练集和测试集都是一模一样的，减少不必要的影响；

#观察一下划分后数据：
print('shape of X_train:{}'.format(X_train.shape))
print('shape of y_train:{}'.format(y_train.shape))
print('='*64)
print('shape of X_test:{}'.format(X_test.shape))
print('shape of y_test:{}'.format(y_test.shape))

#画图观察一下数据：问题是否棘手？
#一般画图使用scatter plot 散点图，但是有一个缺点：只能观察2维的数据情况；如果想观察多个特征之间的数据情况，scatter plot并不可行；
#用pair plot 可以观察到任意两个特征之间的关系图（对角线为直方图）；恰巧：pandas的 scatter_matrix函数能画pair plots。
#所以，我们先把训练集转换成DataFrame形式，方便画图；
iris_dataframe = pd.DataFrame(X_train,columns=iris_dataset.feature_names)

grr = pd.plotting.scatter_matrix(iris_dataframe,c=y_train,figsize=(15,15),marker='o',hist_kwds={'bins':20},s=60,alpha=.8)
#不同颜色代表不同的分类；
# plt.show() #加上这个才会显示图形，否则不显示

#模型训练
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)#设置为最近邻；
knn.fit(X_train,y_train)

# 评估模型
# 方法一：手动计算
y_pred = knn.predict(X_test)#预测
print('Test Set Score:{:.2f}'.format(np.mean(y_test == y_pred)))#自己计算得分情况；准确率

# 方法二：score函数
print('Test Set Score:{:.2f}'.format(knn.score(X_test,y_test)))
#用测试集去打个分，看看得分情况，确定分类器是否可信；
#Socore为97%：说明在测试集上有97%的记录都被正确分类了；分类效果很好，值得信赖！

# 模型应用
#我们可以用训练好的模型去应用了：unseen data
X_new = np.array([[5,2.9,1,0.2]]) #新数据 为什么定为2维的？ 因为sklearn 总是期望收到二维的numpy数组.
result = knn.predict(X_new)
print('Prediction:{}'.format(result))
print('Predicted target name:{}'.format(iris_dataset['target_names'][result]))