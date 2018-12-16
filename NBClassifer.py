import pandas as pd
import numpy as np
import time
from sklearn.model_selection import  train_test_split
from sklearn.naive_bayes import GaussianNB

# Importing dataset.
# Please refer to the 【Data】 part after the code for the data file.

data = pd.read_csv("./career_data.csv")

# Convert categorical variable to numeric
data["985_cleaned"] = np.where(data["985"] == "Yes", 1, 0)
data["education_cleaned"] = np.where(data["education"] == "bachlor", 1,
                                     np.where(data["education"] == "master", 2,
                                              np.where(data["education"] == "phd", 3, 4)
                                              )
                                     )
data["skill_cleaned"]=np.where(data["skill"] == "c++", 1,
                               np.where(data["skill"] == "java", 2, 3
                                        )
                               )
data["enrolled_cleaned"] = np.where(data["enrolled"] == "Yes", 1, 0)

# Split dataset in training and test datasets
X_train, X_test = train_test_split(data, test_size=0.1, random_state=int(time.time()))

print("X_train=", X_train)
print("X_test=", X_test)

# Instantiate the classifier
gnb = GaussianNB()
used_features = ["985_cleaned", "education_cleaned", "skill_cleaned"]

gnb.fit(X_train[used_features].values, X_train["enrolled_cleaned"])


y_pred = gnb.predict(X_test[used_features])

print("y_pred=", y_pred)

# Print results
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%".format(
    X_test.shape[0],
    (X_test["enrolled_cleaned"] != y_pred).sum(),
    100*(1-(X_test["enrolled_cleaned"] != y_pred).sum()/X_test.shape[0])
))

