# RF  

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from ..data import Data  # 假設您有一個名為Data的模組來獲取資料
data = Data().data_encoder()

# 繼上題，了解所有的特徵後選擇1,3,11,13
X = data[['age','fnlwgt',"capital-gain","hours-per-week"]]
y = data[['salary']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)


knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

rf_y_pred = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print("隨機森林分類器的準確率:", rf_accuracy)

knn_y_pred = knn_classifier.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_y_pred)
print("K 最近鄰分類器的準確率:", knn_accuracy)


# 繪製分類圖
plt.figure(figsize=(12, 6))

# 隨機森林的預測結果
plt.subplot(1, 2, 1)
plt.scatter(X_test['age'], X_test['hours-per-week'], c=rf_y_pred, cmap='coolwarm', alpha=0.6)
plt.title('Random Forest Predictions')
plt.xlabel('Age')
plt.ylabel('Hours per Week')

# K 最近鄰的預測結果
plt.subplot(1, 2, 2)
plt.scatter(X_test['age'], X_test['hours-per-week'], c=knn_y_pred, cmap='coolwarm', alpha=0.6)
plt.title('KNN Predictions')
plt.xlabel('Age')
plt.ylabel('Hours per Week')

plt.tight_layout()
plt.show()