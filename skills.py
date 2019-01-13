# 常用机器学习实践技巧  https://zhuanlan.zhihu.com/p/50444108
from sklearn import datasets
import pandas as pd #数据分析、处理
import numpy as np #科学计算包
import matplotlib.pyplot as plt #画图
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

import random
data = datasets.load_iris()
col_name = data.feature_names
X = data.data
y = data.target

# print(X)
# print(y)

# 数据预览
# print(X[:10])
# print(X.shape)
# print(X.dtypes)
# print(X.describe())
# X.sample(n=10)

# 先把训练集转换成DataFrame形式
iris_dataframe = pd.DataFrame(data.data,columns=col_name)
print(iris_dataframe.head())
print(iris_dataframe.sample(n=10))
print(iris_dataframe.dtypes)
print(iris_dataframe.describe())

# 绘制箱型图
iris_dataframe.plot(kind="box",subplots=True,layout=(1,4),figsize=(12,5))
plt.show()

# 绘制条形图
iris_dataframe.hist(figsize=(12,5),xlabelsize=1,ylabelsize=1)
plt.show()

# 绘制密度图
iris_dataframe.plot(kind="density",subplots=True,layout=(1,4),figsize=(12,5))
plt.show()

# 绘制特征相关图
pd.plotting.scatter_matrix(iris_dataframe,figsize=(10,10))
plt.show()

# 绘制特征相关性的热力图
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
cax = ax.matshow(iris_dataframe.corr(),vmin=-1,vmax=1,interpolation="none")
fig.colorbar(cax)
ticks = np.arange(0,4,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(col_name)
ax.set_yticklabels(col_name)
plt.show()

# 查找最优模型
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

models = []
models.append(("AB",AdaBoostClassifier()))
models.append(("GBM",GradientBoostingClassifier()))
models.append(("RF",RandomForestClassifier()))
models.append(("ET",ExtraTreesClassifier()))
models.append(("SVC",SVC()))
models.append(("KNN",KNeighborsClassifier()))
models.append(("LR",LogisticRegression()))
models.append(("GNB",GaussianNB()))
models.append(("LDA",LinearDiscriminantAnalysis()))

names = []
results = []

for name,model in models:
    kfold = KFold(n_splits=5,random_state=42)
    result = cross_val_score(model,X,y,scoring="accuracy",cv=kfold)
    names.append(name)
    results.append(result)
    print("{}  Mean:{:.4f}(Std{:.4f})".format(name,result.mean(),result.std()))

pipeline = []
pipeline.append(("ScalerET", Pipeline([("Scaler",StandardScaler()),
                                       ("ET",ExtraTreesClassifier())])))
pipeline.append(("ScalerGBM", Pipeline([("Scaler",StandardScaler()),
                                        ("GBM",GradientBoostingClassifier())])))
pipeline.append(("ScalerRF", Pipeline([("Scaler",StandardScaler()),
                                       ("RF",RandomForestClassifier())])))

names = []
results = []
for name,model in pipeline:
    kfold = KFold(n_splits=5,random_state=42)
    result = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")
    results.append(result)
    names.append(name)
    print("{}:  Error Mean:{:.4f} (Error Std:{:.4f})".format(
        name,result.mean(),result.std()))

#模型调节 网格搜索对模型参数进行批量调节
param_grid = {
    "C":[0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0],
    "kernel":['linear', 'poly', 'rbf', 'sigmoid']
}
model = SVC()
kfold = KFold(n_splits=5, random_state=42)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring="accuracy", cv=kfold)
grid_result = grid.fit(X, y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
