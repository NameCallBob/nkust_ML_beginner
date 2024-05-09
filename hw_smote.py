from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score


import numpy as np
from data import Data

data = Data().data_encoder()

# 提取特徵和目標變數
X = data.drop(columns=['salary'])  # 特徵
y = data['salary']  # 目標變數

# 初始化SMOTE
smote = SMOTE()

# 使用SMOTE進行過取樣
X_resampled, y_resampled = smote.fit_resample(X, y)
print(y_resampled.value_counts())

model_1 = LogisticRegression(
    # {
    #     'C': 100, 
    #     'penalty': 'l1'
    # }
    )  # 羅吉斯

model_2 = SVC(

)  # 支援向量機




from HW_past.hw_model_score import model_training

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42)

if __name__ == "__main__":
    # 功課輸出
    print("Logistic Regression的模型績效如下")
    print("-"*50)
    model_training(X_train, X_test, y_train, y_test, model_1)
    print("-"*50)
    print("Support Vector Machine的模型績效如下")
    print("-"*50)
    model_training(X_train, X_test, y_train, y_test, model_2)
    print("-"*50)
