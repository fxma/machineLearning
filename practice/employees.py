import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

#先做数据清理
#清理的数据放入新的csv文件
# degree使用0 1 2 3,
# education使用985 211 normal 分别对应 0 1 2,
# skills使用二进制的位运算,比如,C++在第二位的标记为1,00000010,否则为0，00000000,然后转换为十进制2和0
#读取csv文件运用分类器模型训练
#预测数据

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
degreeDict = {}
degree_cleaned=[]
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

print(degree_cleaned)
for idx, val in (enumerate(degreeSet)):
    degreeDict[val] = idx
    # print(idx)
    # print(val)
print(degreeDict)


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