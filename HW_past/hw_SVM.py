"""SVM作業"""
from ..data import Data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
print("SVM作業!")


data = Data().data_encoder()

X = data[['education-num', 'occupation']].values  # 將特徵轉換為數值陣列
y = data['salary'].values  # 將目標變數轉換為數值陣列

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

svm_classifier = SVC()

svm_classifier.fit(X_train, y_train)

y_pred = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("模型準確度：", accuracy)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=30)

# 繪製決策邊界
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                     np.linspace(ylim[0], ylim[1], 50))
Z = svm_classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
            linestyles=['--', '-', '--'])

# 標註支持向量
plt.scatter(svm_classifier.support_vectors_[:, 0], svm_classifier.support_vectors_[:, 1],
            s=100, facecolors='none', edgecolors='k')

plt.title('SVM RESULT')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.show()

print("助教 這張圖代表什麼?")
# 參考 https://ithelp.ithome.com.tw/articles/10333921
