import numpy as np
import matplotlib.pyplot as plt
# size of the point dataset
m = 20
# Points x-coordinate and dummy value(x0,x1)
# x轴的点和虚拟数据
X0 = np.ones((m, 1))
X1 = np.arange(1, m+1).reshape(m, 1)
X = np.hstack((X0, X1))

# Points y-coordinate
Y = np.array([
    3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
    11, 13, 13, 16, 17, 18, 17, 19, 21
]).reshape(m, 1)

# The Learning Rate alpha.
alpha = 0.01


# 以矩阵向量的形式定义代价函数和代价函数的梯度
# 代价函数
def error_function(theta, X, Y):
    # '''Error function J definition.'''
    diff = np.dot(X, theta) - Y
    return (1./2*m) * np.dot(np.transpose(diff), diff)


# 代价函数的梯度
def gradient_function(theta, X, Y):
    # '''Gradient of the function J definition.'''
    diff = np.dot(X, theta) - Y
    return (1./m) * np.dot(np.transpose(X), diff)


# 梯度下降迭代计算
def gradient_descent(X, Y, alpha):
    # '''Perform gradient descent.'''
    theta = np.array([1, 1]).reshape(2, 1)
    gradient = gradient_function(theta, X, Y)
    print('theta:', theta)
    print('gradient:', gradient)
    while not np.all(np.absolute(gradient) <= 1e-5):
        theta = theta - alpha * gradient
        gradient = gradient_function(theta, X, Y)
        print('gradient:', gradient)
    return theta


if __name__ == '__main__':
    print('X:', X)
    print('Y:', Y)
    print('alpha:', alpha)

    optimal = gradient_descent(X, Y, alpha)
    print('optimal:', optimal)
    print('error function:', error_function(optimal, X, Y)[0, 0])

    t = np.arange(0., 20., 0.1)
    plt.plot(X1, Y, 'o')
    plt.plot(t, optimal[0]+optimal[1]*t)

    plt.show()
