# 升級模型

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from data import Data
class UP:
    def way1_vote(self):
        X_train, X_test, y_train, y_test = Data().SMOTE_fitted_trainingdata()

        # 定義基模型
        clf1 = LogisticRegression()
        clf1.set_params(**{
            'C': 0.01, 'penalty': 'l2', 'solver': 'newton-cg'
            })
        clf2 = DecisionTreeClassifier()
        clf2.set_params(
            **{'criterion': 'entropy',
                 'max_depth': 20,
                 'max_features': None,
                 'max_leaf_nodes': None,
                 'min_samples_leaf': 2,
                 'min_samples_split': 2,
                 'splitter': 'random'})
        clf3 = SVC(
            probability=True
        )

        # 定義投票分類器（軟投票）
        eclf = VotingClassifier(estimators=[('lr', clf1), ('dt', clf2), ('svc', clf3)], voting='soft')

        # 訓練投票分類器
        eclf.fit(X_train, y_train)
        y_pred = eclf.predict(X_train)
        self.__output_score("votting",y_train,y_pred,0)
        # 評估模型性能
        y_pred = eclf.predict(X_test)
        self.__output_score("votting",y_test,y_pred,1)
    
    def __output_score(self,title,model_data,pred,status):
        """
        輸出模型績效指數
        @ title -> 模型標題
        @ model_data -> 模型資料
        @ pred -> 模型實際預測
        @ status -> 輸出模式(0:訓練集分數;1:測試集分數)
        """
        from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score , roc_auc_score

        cm = confusion_matrix(model_data, pred)
        precision = precision_score(model_data, pred)
        accuracy = accuracy_score(model_data, pred)
        recall = recall_score(model_data, pred)
        f1 = f1_score(model_data, pred)
        auc = roc_auc_score(model_data, pred)
        # 輸出
        print(f"-----------{title}-----------")
        print("----測試資料之模型績效----" if status  else "----訓練資料之模型績效----")
        print("Confusion Matrix:")
        print(cm)
        print()
        print("Precision:", precision)
        print("Accuracy:", accuracy)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("AUC:", auc)
        print("-----------END-----------")

if __name__ == "__main__":
    UP().way1_vote()