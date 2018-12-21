import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# 生成样本数据
X = np.sort(5 * np.random.rand(40, 1), axis=0)
# y = np.ravel(2*X + 3)
# y = np.polyval([2,3,5,2], X).ravel()
y = np.sin(X).ravel()

# 加入部分噪音
y[::5] += 3 * (0.5 - np.random.rand(8))

# 调用模型
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)

# 可视化结果
lw = 2
plt.scatter(X, y, color='darkorange', label='data')
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()