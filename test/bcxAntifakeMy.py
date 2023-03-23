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
                   s=100, linewidth=1,  edgecolors='blue', facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

#读取数据文件
data = pd.read_csv("myData去采集.csv")
print(data)
print(data["imgPath"])

#转换为numpy数组
np_data = data[["scanId","key1","key2","keyRatio","knnRatio","ssim","lightRatio","duibiduRatio","laplacianRatio","brennerRatio","tenengardRatio","energyRatio","eavRatio","blurNoiseRatio","target"
]].values
# 转换数据类型
np_data = np_data.astype(float)

# np_data_s = StandardScaler().fit_transform(np_data)

# 放大数据
# scale=np.array([1000,100,100,1])
# np_data = np_data*scale

np_data_genuine = np_data[np.where(np_data[:, 14] == 1)]
np_data_fake = np_data[np.where(np_data[:, 14] == -1)]
np_data_copy = np_data[np.where(np_data[:, 14] == -2)]

ax = plt.subplot(projection='3d')
ax.scatter3D(np_data_genuine[:, 4], np_data_genuine[:, 5], np_data_genuine[:, 7], c='green', s=10, cmap='RdYlGn');
plt.show()

ax = plt.subplot(projection='3d')
ax.scatter3D(np_data_fake[:, 4], np_data_fake[:, 5], np_data_fake[:, 7], c='red', s=10, cmap='RdYlGn');
plt.show()

ax = plt.subplot(projection='3d')
ax.scatter3D(np_data_copy[:, 4], np_data_copy[:, 5], np_data_copy[:, 7], c='orangered', s=10, cmap='RdYlGn');
plt.show()

ax = plt.subplot(projection='3d')
ax.scatter3D(np_data[:, 4], np_data[:, 5], np_data[:, 7], c=np_data[:, 14], s=10, cmap='RdYlGn', alpha=0.6);
plt.show()


from plotly.offline import plot
from plotly import subplots
# 加载数据
import plotly.graph_objs as go
# 准备数据
dataframe = np_data

# # 构造3D散点图trace1
# trace1 = go.Scatter3d(
#     x=data.knnRatio,
#     y=data.ssim,
#     z=data.duibiduRatio,
#     hovertext = data.scanId,
#     mode='markers',
#     marker=dict(
#         size=5,
#         color=data.target,  # RGB颜色对照表可参考：https://tool.oschina.net/commons?type=3
#         colorscale = 'RdYlGn', # 选择颜色
#         opacity=0.68  # 透明度
#     )
# )
#
# dataTrace = [trace1]
# layout = go.Layout(
#     margin=dict(
#         l=0,
#         r=0,
#         b=0,
#         t=0
#     )
# )
# fig = go.Figure(data=dataTrace, layout=layout)
# plot(fig)


# 两个基本参数：设置行、列
# fig = subplots.make_subplots(rows=2, cols=2,
#                     specs=[[{'is_3d': True}, {'is_3d': True}],
#                            [{'is_3d': True}, {'is_3d': True}]],
#                              print_grid=False)
# # # 构造3D散点图trace1

# # # 构造3D散点图trace1
# trace2 = go.Scatter3d(
#     x=data.knnRatio,
#     y=data.ssim,
#     z=data.duibiduRatio,
#     hovertext = data.scanId,
#     mode='markers',
#     marker=dict(
#         size=5,
#         color=data.target,  # RGB颜色对照表可参考：https://tool.oschina.net/commons?type=3
#         colorscale = 'RdYlGn', # 选择颜色
#         opacity=0.68  # 透明度
#     )
# )
# fig.add_trace(trace1, 1, 1)
# fig.add_trace(trace2, 1, 2)
# fig.add_trace(trace2, 2, 1)
# fig.add_trace(trace2, 2, 2)
# # 设置图形的宽高和标题
# fig.update_layout(height=1000,
#                   width=1000,
#                   title_text="子图制作")
# plot(fig)

arr =["keyRatio","ssim","lightRatio","duibiduRatio","laplacianRatio","brennerRatio","tenengardRatio","energyRatio","eavRatio","blurNoiseRatio"]
# 两个基本参数：设置行、列
fig = subplots.make_subplots(rows=len(arr), cols=1,subplot_titles=arr)  # 1行2列
# 添加两个数据轨迹
for i in range(len(arr)):
    fig.add_trace(go.Scatter(
        x=data.knnRatio,
        y=data[arr[i]],
        hovertext=data.scanId,
        mode = "markers",
        marker = dict(size=5,color=data.target,colorscale = 'RdYlGn',opacity=0.68),
    ), row=i+1,col=1)
# 设置图形的宽高和标题
fig.update_layout(
    height=2500,
    # width=800,
    title_text="子图制作")
plot(fig)

#切分变量和目标
np_data = np_data[np.where(np_data[:, 14] > -2)]
X_ = np_data[:, 0:14]
Y_ = np_data[:, 14]

X = np.vstack((X_[:, 4], X_[:, 5]))
X = X.transpose()
# X = StandardScaler().fit_transform(X)
Y = Y_

# 将样本数据绘制在直角坐标中
# plt.scatter(X[:, 0], X[:, 1], c=Y, s=20, cmap='autumn')
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
print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
model = grid.best_estimator_

# 在直角坐标中绘制出分割超平面、辅助超平面和支持向量
plt.scatter(X[:, 0], X[:, 1], c=Y, s=10, cmap='autumn')
plot_svc_decision_function(model)
plt.show()

