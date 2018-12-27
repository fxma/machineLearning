from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
import pandas as pd

# 导入数据
data = pd.read_csv('../data/msft_stockprices_dataset.csv', delimiter=',')
used_features = ["High Price", "Low Price","Open Price","Volume"]
# used_features = ["Volume"]
X = data[used_features].values
scores = data["Close Price"].values

X_train = X[:700]
X_test = X[700:]

# print(X_train)
# print(X_test)

y_train = scores[:700]
y_test = scores[700:]

if __name__ == '__main__':
    # 创建线性回归模型
    regr = LinearRegression()
    # 创建逻辑回归模型，可是调用的时候报错
    # regr = LogisticRegression(C=1e5)
    # 创建SVR回归模型,正确率为0
    # regr = SVR(kernel='rbf', C=1e3, gamma=0.1)
    # 创建SVR回归模型，线性内核，运行没反应
    # regr = SVR(kernel='linear', C=1e3)

    # 用训练集训练模型
    regr.fit(X_train, y_train)

    # 用训练得出的模型进行预测
    y_predict = regr.predict(X_test)

    # |预测值-真实值| / 真实值 <= ErrorTolerance 时
    errorTolerance = 0.05;
    # 正确数
    correctNum = 0;

    for i in range(len(y_predict)):
        predict = y_predict[i];
        original = y_test[i];
        sub = abs(predict-original);
        accuracy = sub/original;
        print("序列号:%d,预测值:%s, 真实值:%s,精度:%s" % (700+1+i, y_predict[i], y_test[i], accuracy))
        if accuracy<errorTolerance:
            correctNum = correctNum+1;

    correctRate = correctNum/len(y_predict)*100
    print("预测准确率:%s,预测正确样本数:%s, 总测试样本数:%s" % (correctRate, correctNum, len(y_predict)))