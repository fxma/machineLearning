from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import pandas as pd

# Importing dataset
data = pd.read_csv('quiz.csv', delimiter=',')

used_features = ["Last Score", "Hours Spent"]
X = data[used_features].values
scores = data["Score"].values

X_train = X[:11]
X_test = X[11:]

# Logistic Regression - Multiple Classification
level = []

for i in range(len(scores)):
    if scores[i] >= 85:
        level.append(2)
    elif scores[i] >= 60:
        level.append(1)
    else:
        level.append(0)

y_train = level[:11]
y_test = level[11:]

classifier = LogisticRegression(C=1e5)
classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)
print(y_predict)
