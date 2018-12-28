import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

#先手动做数据清理，清理的数据放入新的csv文件
#将清理过的数据做数值化
# degree数值化为1 2 3,
# education先转换为985 211 normal 然后数值化为 1 2 3,
# skills使用二进制的位运算,比如,C++在第二位的标记为1,00000010,否则为0，00000000,然后转换为十进制2和0
# position数值化为1 2 3,
#用朴素贝叶斯分类器模型训练
#使用训练过的模型预测数据
#输出准确率

# data = pd.read_csv("../data/employees_dataset.csv.csv")
# 手动清理出来的文件，目前只是清理了学位和学校
data = pd.read_csv("../data/employees_dataset_cleaned.csv")

degree = data["degree"].values
education = data["education"].values
skills = data["skills"].values
workingExperience = data["working_experience"].values
position = data["position"].values

degreeSet = set()
educationSet = set()
skillsSet = set()
workingExperienceSet = set()
positionSet = set()
for val in degree:
    degreeSet.add(val)
for val in education:
    educationSet.add(val)
for val in skills:
    skillsSet.add(val)
for val in workingExperience:
    workingExperienceSet.add(val)
for val in position:
    positionSet.add(val)
print(degreeSet)
print(educationSet)
# print(skillsSet)
# print(workingExperienceSet)
print(positionSet)

# 特征值数值化
data["education_cleaned"] = np.where(data["education"] == "985", 1,
                                     np.where(data["education"] == "211", 2,
                                              np.where(data["education"] == "normal", 3, 4)
                                              ))
data["degree_cleaned"] = np.where(data["degree"] == "bachlor", 1,
                                     np.where(data["degree"] == "master", 2,
                                              np.where(data["degree"] == "phd", 3, 4)
                                              )
                                     )
data["position_cleaned"] = np.where(data["position"] == "qa", 1,
                                  np.where(data["position"] == "manager", 2,
                                           np.where(data["position"] == "dev", 3, 4)
                                           )
                                  )
# 将数据切分为训练集合测试集
X_train, X_test = train_test_split(data, test_size=0.1, random_state=int(time.time()))

print("训练集=\n", X_train)
print("测试集=\n", X_test)

#  贝叶斯分类器初始化
gnb = GaussianNB()
used_features = ["education_cleaned", "degree_cleaned"]
#  训练模型
gnb.fit(X_train[used_features].values, X_train["position_cleaned"])

#  预测
y_pred = gnb.predict(X_test[used_features])

#  打印预测
print("职位预测=\n", y_pred)

print("测试数据个数： {}，预测失败个数: {}, 正确率 {:05.2f}%".format(
    X_test.shape[0],
    (X_test["position_cleaned"] != y_pred).sum(),
    100*(1-(X_test["position_cleaned"] != y_pred).sum()/X_test.shape[0])
))



# dict['Age'] = 8;


# data["education_cleaned"]=np.where(data["education"]=="bachlor",1,
#                                    np.where(data["education"]=="master",2,
#                                             np.where(data["education"]=="phd",3,4)
#                                             )
#                                    )
#
# names = ['Bob','Jessica','Mary','John','dd']
# births = [968,155,77,578,973]
# DataSet = list(zip(names,births))
# DataSet
# df = pd.DataFrame(data = DataSet ,columns=['Names','Births'])
# df
# df.to_csv('births1880.csv', index=False, header=False )