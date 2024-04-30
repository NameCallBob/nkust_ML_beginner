from ..data import Data
import pandas as pd

# 資料預處理
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split


data = Data().data_clean()
train_data = data
print(data)

labelencoder = LabelEncoder()
# 類別資料轉數值

for i in ["age","sex","workclass","education","occupation","salary"]:
    data[i] = labelencoder.fit_transform(data[i])

Ｘ = data[["age","workclass","education","occupation"]]
y = data[['salary']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()

# 訓練模型
model.fit(X_train, y_train)

# 預測測試集
y_pred = model.predict(X_test)

# 計算準確率
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("準確率：", accuracy)



