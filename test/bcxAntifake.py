import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC  # "Support vector classifier"
from sklearn.preprocessing import StandardScaler

# 定义函数plot_svc_decision_function用于绘制分割超平面和其两侧的辅助超平面
def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 创建网格用于评价模型
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    #绘制超平面
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    #标识出支持向量
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=200, linewidth=1,  edgecolors='blue', facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

#读取数据文件
data = pd.read_csv("data.csv")
print(data)
print(data["imgPath"])

#转换为numpy数组
np_data = data.values
print(np_data)
#切分变量和目标
X_ = np_data[:, 1:15]
Y_ = np_data[:, 16]

X_=X_.astype(float)
Y_=Y_.astype(int)


# 特征点、eavBlur、模糊、、运动模糊、ssim
X = np.vstack((X_[:, 8], X_[:, 13]))
# X = np.vstack((X_[:, 8], X_[:, 4],X_[:, 10], X_[:, 11], X_[:, 13]))
X = X.transpose()
# 对数据进行标准化
# X = StandardScaler().fit_transform(X)
Y = Y_

# 将样本数据绘制在直角坐标中
plt.scatter(X[:, 0], X[:, 1], c=Y, s=20, cmap='autumn')
plt.show()

# ax = plt.subplot(projection='3d')
# ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=Y, s=50, cmap='autumn');
# ax.set_xlabel('特征点')
# ax.set_ylabel('ssim')
# ax.set_zlabel('blur')
# plt.show()

# 用线性核函数的SVM来对样本进行分类
#惩罚系数 默认
# model = SVC(kernel='linear')
#惩罚系数 10
# model = SVC(kernel='linear', C=10.0)
# model = SVC(kernel='rbf', C=10, gamma=0.1)
# model.fit(X, Y)

# 网格搜索寻找最佳参数
grid = GridSearchCV(SVC(), param_grid={'C': [0.05, 0.1, 1, 10], 'gamma': [1, 0.1, 0.01, 0.05]}, cv=4)
grid.fit(X, Y)
print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
model = grid.best_estimator_

# 在直角坐标中绘制出分割超平面、辅助超平面和支持向量
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='autumn')
plot_svc_decision_function(model)
plt.show()

