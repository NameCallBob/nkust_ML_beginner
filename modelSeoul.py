# -----------------------------------------------------------------------------
# Copyright (c) 2024 拼拼
#
# 此程式碼依照 GNU 通用公共授權條款第3版（或您選擇的任何更新版本）發佈。
# 您可以自由地重新發佈和修改此程式碼，只要您遵守授權條款。
#
# 此程式碼是基於希望它能有用的前提下發佈，但不附帶任何擔保，
# 甚至不包含針對特定目的的隱含擔保。詳情請參閱 GNU 通用公共授權。
#
# 您應當已經收到一份 GNU 通用公共授權的副本。如果沒有，請參閱
# <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------


from model import Model_c
from data import Data_seoul

class Model_Seoul_C():
    """首爾資料"""
    def __init__(self) -> None:
        super().__init__()
        self.TrainingData = Data_seoul().Source_trainingdata_forClassifer()
        self.model_score = {
            "name" : [],
            # 訓練集分數
            "score":{
                "cm":[],
                "precision":[],
                "accuracy":[],
                "recall":[],
                "f1":[],
                "auc":[]
            },
            # 實際預測分數
            "score_1":{
                "cm":[],
                "precision":[],
                "accuracy":[],
                "recall":[],
                "f1":[],
                "auc":[]
            }
        }
        self.count = 0
        self.roc_data = []
        self.roc_data_train = []

    def main_SEOUL(self,save=False):
        """主執行_首爾"""
        if not self.TrainingData:
            print("未取得資料，終止執行");

        X_train, X_test, y_train, y_test = self.TrainingData

        self.logistic(X_train, X_test, y_train, y_test)
        self.DecisionTree(X_train, X_test, y_train, y_test)
        self.RandomForest(X_train, X_test, y_train, y_test)
        self.XGBoost_classifer(X_train, X_test, y_train, y_test)
        self.Adaboost_classifer(X_train, X_test, y_train, y_test)
        self.SVM(X_train, X_test, y_train, y_test)
        self.KNN(X_train, X_test, y_train, y_test)

        if save:
            self.save_toExcel(
                name="ClassiferSeoulResult.xlsx"
            )
    def logistic(self,X_train, X_test, y_train, y_test):
            """羅吉斯回歸"""
            from sklearn.linear_model import LogisticRegression
            param = {'C': 0.01, 'penalty': 'l2'}
            model = LogisticRegression(multi_class='ovr', solver='liblinear')
            model.set_params(**param)
            # 訓練模型
            model.fit(X_train, y_train)

            # 預測測試集
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            self.__output_score("Logistic羅吉斯",y_train,y_pred_train,0)
            self.__output_score("Logistic羅吉斯",y_test,y_pred_test,1)

            self.draw_ROC_pic(model,X_train, X_test, y_train, y_test,"Logistic")

    def DecisionTree(self,X_train, X_test, y_train, y_test):
        """決策樹"""
        from sklearn.tree import DecisionTreeClassifier
        param = {'max_depth': 10, 'min_samples_split': 2}
        model = DecisionTreeClassifier()
        model.set_params(**param)
        # 訓練模型
        model.fit(X_train, y_train)

        # 預測測試集
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        self.__output_score("Decision決策樹",y_train,y_pred_train,0)
        self.__output_score("Decision決策樹",y_test,y_pred_test,1)

        self.draw_ROC_pic(model,X_train, X_test, y_train, y_test,"DecisionTree")

    def RandomForest(self,X_train, X_test, y_train, y_test):
        """隨機森林"""
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100,
                           max_depth=10,
                           min_samples_split=10,
                           min_samples_leaf=5,
                           random_state=42)
        model.fit(X_train, y_train)
        # 預測測試集
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        self.__output_score("RandomForest",y_train,y_pred_train,0)
        self.__output_score("RandomForest",y_test,y_pred_test,1)
        self.draw_ROC_pic(model,X_train, X_test, y_train, y_test,"RandomForest")

    def Adaboost_classifer(self,X_train, X_test, y_train, y_test):
        """"""
        from sklearn.ensemble import AdaBoostClassifier
        param = {'learning_rate': 0.5, 'n_estimators': 100}
        ada = AdaBoostClassifier(random_state=42)
        ada.set_params(**param)
        ada.fit(X_train, y_train)

        # 預測
        # 預測測試集
        y_pred_train = ada.predict(X_train)
        y_pred_test = ada.predict(X_test)

        self.__output_score("Adaboost",y_train,y_pred_train,0)
        self.__output_score("Adaboost",y_test,y_pred_test,1)
        self.draw_ROC_pic(ada,X_train, X_test, y_train, y_test,"adaboost")

    def XGBoost_classifer(self,X_train, X_test, y_train, y_test):
        """XGBboost 分類"""
        from xgboost import XGBClassifier

        # 建立 XGBClassifier 模型
        xgboostModel = XGBClassifier(n_estimators=100, learning_rate= 0.1,max_depth=10)
        # 使用訓練資料訓練模型
        xgboostModel.fit(X_train, y_train)
        # 使用訓練資料預測分類
        y_pred_train = xgboostModel.predict(X_train)
        y_pred_test = xgboostModel.predict(X_test)

        self.__output_score("XGBoostModel",y_train,y_pred_train,0)
        self.__output_score("XGBoostModel",y_test,y_pred_test,1)
        self.draw_ROC_pic(xgboostModel,X_train, X_test, y_train, y_test,"XGBboost classifer")


    def SVM(self,X_train, X_test, y_train, y_test):
        """向量機模型"""
        from sklearn.svm import SVC
        svm_classifier = SVC(
            probability=True,
            C=0.1,
            kernel="linear"
        )

        svm_classifier.fit(X_train, y_train)

        y_pred_train = svm_classifier.predict(X_train)
        y_pred_test = svm_classifier.predict(X_test)

        self.__output_score("SVM 向量機學習",y_train,y_pred_train,0)
        self.__output_score("SVM 向量機學習",y_test,y_pred_test,1)

        self.draw_ROC_pic(svm_classifier,X_train, X_test, y_train, y_test,"SVM model")

    def KNN(self,X_train, X_test, y_train, y_test):
        """KNN模型"""
        from sklearn.neighbors import KNeighborsClassifier

        param = {'algorithm': 'auto', 'leaf_size': 50, 'n_neighbors':11, 'p': 1, 'weights': 'uniform'}
        knn_classifier = KNeighborsClassifier()
        knn_classifier.set_params(**param)
        knn_classifier.fit(X_train, y_train)

        y_pred_train = knn_classifier.predict(X_train)
        y_pred_test = knn_classifier.predict(X_test)

        self.__output_score("KNN模型",y_train,y_pred_train,0)
        self.__output_score("KNN模型",y_test,y_pred_test,1)

        self.draw_ROC_pic(knn_classifier,X_train, X_test, y_train, y_test,"KNN model")

    def draw_ROC_pic(self,model,X_train, X_test, y_train, y_test,model_name):
            import numpy as np
            import matplotlib.pyplot as plt
            from sklearn.metrics import roc_curve, auc
            return ;
            y_score = model.predict_proba(X_test)[:, 1]
            # 計算ROC曲線
            fpr, tpr, _ = roc_curve(y_test, y_score)
            # 計算AUC
            roc_auc = auc(fpr, tpr)
            # 儲存ROC數據
            self.roc_data.append((fpr, tpr, roc_auc, model_name))
            self.count += 1

            # 如果count達到14，繪製所有ROC曲線
            if self.count == 7:
                self.plot_all_roc_curves()
            plt.show()

    def plot_all_roc_curves(self):
        import matplotlib.pyplot as plt
        plt.figure()
        for fpr, tpr, roc_auc, model_name in self.roc_data:
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig('pic/output_AllModelROC.png')
        plt.show()

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
        precision = precision_score(model_data, pred, average='macro')
        accuracy = accuracy_score(model_data, pred)
        recall = recall_score(model_data, pred, average='macro')
        f1 = f1_score(model_data, pred, average='macro')
        # 將多類別改為二元
        from sklearn.preprocessing import label_binarize
        bin_pred = label_binarize(pred,classes=[0,1,2,3])
        auc = roc_auc_score(model_data, bin_pred , multi_class="ovr")
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
        # 測試集
        if status :
            self.model_score["score_1"]["cm"].append(cm)
            self.model_score["score_1"]["precision"].append(precision)
            self.model_score["score_1"]["accuracy"].append(accuracy)
            self.model_score["score_1"]["recall"].append(recall)
            self.model_score["score_1"]["f1"].append(f1)
            self.model_score["score_1"]["auc"].append(auc)
            return;
        # 訓練集
        self.model_score["name"].append(title)
        self.model_score["score"]["cm"].append(cm)
        self.model_score["score"]["precision"].append(precision)
        self.model_score["score"]["accuracy"].append(accuracy)
        self.model_score["score"]["recall"].append(recall)
        self.model_score["score"]["f1"].append(f1)
        self.model_score["score"]["auc"].append(auc)

    def save_toExcel(self,name="ClassiferResult.xlsx"):
            import pandas as pd
            pd.DataFrame(
                {
                "name":self.model_score["name"],
                "cm_0":self.model_score["score"]["cm"],
                "cm_1":self.model_score["score_1"]["cm"],
                "pre_0":self.model_score["score"]["precision"],
                "pre_1":self.model_score["score_1"]["precision"],
                "acc_0":self.model_score["score"]["accuracy"],
                "acc_1":self.model_score["score_1"]["accuracy"],
                "recall_0":self.model_score["score"]["recall"],
                "recall_1":self.model_score["score_1"]["recall"],
                "f1_0":self.model_score["score"]["f1"],
                "f1_1":self.model_score["score_1"]["f1"],
                "auc_0":self.model_score["score"]["auc"],
                "auc_1":self.model_score["score_1"]["auc"]
                }
            ).to_excel(f"./result/{name}")

            print("輸出完畢")
    def best_params(self):
        from sklearn.model_selection import GridSearchCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
        from xgboost import XGBClassifier
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        X_train, X_test, y_train, y_test = self.TrainingData
        # 找到最佳Logistic Regression參數
        param_grid = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}
        logistic_clf = LogisticRegression(multi_class='ovr', solver='liblinear')
        logistic_grid = GridSearchCV(logistic_clf, param_grid, cv=5)
        logistic_grid.fit(X_train, y_train)
        print('最佳Logistic Regression參數:', logistic_grid.best_params_)

        # 找到最佳Decision Tree參數
        param_grid = {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]}
        dt_clf = DecisionTreeClassifier()
        dt_grid = GridSearchCV(dt_clf, param_grid, cv=5)
        dt_grid.fit(X_train, y_train)
        print('最佳Decision Tree參數:', dt_grid.best_params_)

        # 找到最佳Random Forest參數
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10]}
        rf_clf = RandomForestClassifier()
        rf_grid = GridSearchCV(rf_clf, param_grid, cv=5)
        rf_grid.fit(X_train, y_train)
        print('最佳Random Forest參數:', rf_grid.best_params_)

        # 找到最佳XGBoost參數
        param_grid = {'learning_rate': [0.01, 0.1, 0.3], 'max_depth': [3, 5, 10]}
        xgb_clf = XGBClassifier()
        xgb_grid = GridSearchCV(xgb_clf, param_grid, cv=5)
        xgb_grid.fit(X_train, y_train)
        print('最佳XGBoost參數:', xgb_grid.best_params_)

        # 找到最佳Adaboost參數
        param_grid = {'n_estimators': [50, 100, 150], 'learning_rate': [0.1, 0.5, 1.0]}
        ada_clf = AdaBoostClassifier()
        ada_grid = GridSearchCV(ada_clf, param_grid, cv=5)
        ada_grid.fit(X_train, y_train)
        print('最佳Adaboost參數:', ada_grid.best_params_)

        # 找到最佳SVM參數
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        svm_clf = SVC()
        svm_grid = GridSearchCV(svm_clf, param_grid, cv=5)
        svm_grid.fit(X_train, y_train)
        print('最佳SVM參數:', svm_grid.best_params_)

        # 找到最佳KNN參數
        param_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
        knn_clf = KNeighborsClassifier()
        knn_grid = GridSearchCV(knn_clf, param_grid, cv=5)
        knn_grid.fit(X_train, y_train)
        print('最佳KNN參數:', knn_grid.best_params_)

if __name__ == "__main__" :
    Model_Seoul_C().main_SEOUL(save=True)
    # Model_Seoul_C().best_params()
