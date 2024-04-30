# 模型績效

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score

from ..data import Data

df = Data().data_encoder()

X = df.drop("salary", axis=1)
y = df['salary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


def model_training(X_train, X_test, y_train, y_test, model):
    """簡單的模型訓練"""
    model = model

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    precision = precision_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Confusion Matrix:")
    print(cm)
    print("Precision:", precision)
    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("績效說明:")
    """
    羅吉斯 ->
    Confusion Matrix:
    [[4660  282]
    [1107  464]]
    Precision: 0.6219839142091153
    Accuracy: 0.7867342238599724
    Recall: 0.29535327816677276
    F1 Score: 0.40051791109192925
    筆記:
    CM的True Negative (4660) 和 True Positive (464) 都相對較高
    但發現False Negative (1107) 的數量較高，這表示模型在捕捉實際正例方面可能存在一些問題。
    Recall (召回率)：0.295，表示模型僅捕捉到了實際正例的29.5%。
    F1 Score：0.401，是Precision和Recall的平衡值，較低的F1 Score可能表明模型在Precision和Recall之間沒有很好的平衡。

    SVM ->
    Confusion Matrix:
    [[4938    4]
    [1326  245]]
    Precision: 0.9839357429718876
    Accuracy: 0.7957930293259634
    Recall: 0.15595162316995545
    F1 Score: 0.2692307692307692
    筆記:
    True Negative (4938) 和 True Positive (245) 都相對較高
    但False Negative (1326) 的數量較高，這表示模型在捕捉實際正例方面也存在一些問題
    Recall (召回率)：0.156，表示模型僅捕捉到了實際正例的15.6%，這是一個較低的值。
    F1 Score：0.269，是Precision和Recall的平衡值，較低的F1 Score可能表明模型在Precision和Recall之間沒有很好的平衡。
    """
    print("")


model_1 = LogisticRegression()  # 羅吉斯
model_2 = SVC()  # 支援向量機

# output
print("Logistic Regression的模型績效如下")
print("-"*50)
model_training(X_train, X_test, y_train, y_test, model_1)
print("-"*50)
print("Support Vector Machine的模型績效如下")
print("-"*50)
model_training(X_train, X_test, y_train, y_test, model_2)
print("-"*50)
