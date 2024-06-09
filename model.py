

from data import Data

class Model_c():
    """集結所有機器訓練之模型"""

    def __init__(self) -> None:
        data = Data()
        self.smote_data = data.SMOTE_fitted_trainingdata()
        self.resource_data = data.Source_trainingdata()
        self.pca_smote_data = data.SMOTE_PCA_trainingData()
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

    def logistic(self,X_train, X_test, y_train, y_test):
        """羅吉斯回歸"""
        from sklearn.linear_model import LogisticRegression
        param = {'C': 0.01, 'penalty': 'l2', 'solver': 'newton-cg'}
        model = LogisticRegression()
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
        param = {'criterion': 'entropy',
                 'max_depth': 20,
                 'max_features': None,
                 'max_leaf_nodes': None,
                 'min_samples_leaf': 2,
                 'min_samples_split': 2,
                 'splitter': 'random'}
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
                           max_depth=5,
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
        ada = AdaBoostClassifier(random_state=42)
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
        xgboostModel = XGBClassifier(n_estimators=100, learning_rate= 0.3)
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
            probability=True
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

    def KNN_test(self,X_train, X_test, y_train, y_test):
        # 嘗試不同的k值並使用指定的參數
        from sklearn.model_selection import cross_val_score
        from sklearn.neighbors import KNeighborsClassifier
        import numpy as np

        param = {'algorithm': 'ball_tree', 'leaf_size': 10, 'p': 1, 'weights': 'distance'}
        k_range = range(1, 31)
        k_scores = []

        for k in k_range:
            param['n_neighbors'] = k
            knn = KNeighborsClassifier(**param)
            scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
            k_scores.append(scores.mean())

        # 找出最佳k值
        best_k = k_range[np.argmax(k_scores)]
        print(f'最佳k值: {best_k}, 平均準確率: {max(k_scores):.4f}')

    def draw_ROC_pic(self,model,X_train, X_test, y_train, y_test,model_name):
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
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

    def main(self,use_smote=True,pca=False,save=False):
        """主執行"""
        if not self.smote_data or not self.resource_data:
            print("未取得資料，終止執行");
        if use_smote:
            if pca:
                X_train, X_test, y_train, y_test = self.pca_smote_data
            else:
                X_train, X_test, y_train, y_test = self.smote_data
        else:
            X_train, X_test, y_train, y_test = self.resource_data

        self.logistic(X_train, X_test, y_train, y_test)
        self.DecisionTree(X_train, X_test, y_train, y_test)
        self.RandomForest(X_train, X_test, y_train, y_test)
        self.XGBoost_classifer(X_train, X_test, y_train, y_test)
        self.Adaboost_classifer(X_train, X_test, y_train, y_test)
        self.SVM(X_train, X_test, y_train, y_test)
        self.KNN(X_train, X_test, y_train, y_test)

        if save:
            self.save_toExcel()

    def main_find(self,use_smote=True):
        """主要 尋找"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from xgboost import XGBClassifier
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.ensemble import RandomForestClassifier


        if not self.smote_data or not self.resource_data:
            print("未取得資料，終止執行");
        if use_smote:
            X_train, X_test, y_train, y_test = self.smote_data
        else:
            X_train, X_test, y_train, y_test = self.resource_data

        model = [
            LogisticRegression(),
            DecisionTreeClassifier(),
            KNeighborsClassifier(),
            SVC(),
            XGBClassifier(),
            AdaBoostClassifier(),
            RandomForestClassifier()
        ]
        param = [
        # Logistic
        {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        },
        # Decision
        {
        'criterion': ['gini', 'entropy'],  # 衡量分裂品質的功能
        'splitter': ['best', 'random'],  # 選擇分裂點的策略
        'max_depth': [None, 10, 20, 30, 40, 50],  # 樹的最大深度
        'min_samples_split': [2, 5, 10],  # 分裂一個內部節點所需的最小樣本數
        'min_samples_leaf': [1, 2, 4],  # 在葉子節點中需要的最小樣本數
        'max_features': [None, 'sqrt', 'log2'],  # 在每次分裂時考慮的最大特徵數
        'max_leaf_nodes': [None, 10, 20, 30]  # 最大葉子節點數
        },
        # KNN
        {
            'n_neighbors': [3, 5, 7, 9, 11],  # 最近鄰的數量
            'weights': ['uniform', 'distance'],  # 鄰居權重
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # 用於計算最近鄰的算法
            'leaf_size': [10, 20, 30, 40, 50],  # 構建樹時的葉子大小
            'p': [1, 2]  # 距離度量，1=曼哈頓距離，2=歐氏距離
        },
        # SVM
        {
            'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
        },
        # xgboost
        {
            'n_estimators': [50, 100, 150],  # 樹的數量
            'max_depth': [3, 5, 7],  # 樹的最大深度
            'learning_rate': [0.01, 0.1, 0.2],  # 學習率
            'subsample': [0.6, 0.8, 1.0],  # 每棵樹訓練的數據比例
            'colsample_bytree': [0.6, 0.8, 1.0],  # 每棵樹使用的特徵比例
            'gamma': [0, 0.1, 0.2],  # 節點分裂所需的最小損失減少量
            'reg_alpha': [0, 0.01, 0.1],  # L1 正則化
            'reg_lambda': [0.1, 1, 10]  # L2 正則化
        },
        # ada
        {
            'n_estimators': [50, 100, 150, 200],  # 弱學習器的數量
            'learning_rate': [0.01, 0.1, 0.5, 1.0],  # 學習率
            'base_estimator__max_depth': [1, 2, 3, 4, 5]  # 基礎分類器的最大深度
        },
        # RF
        {
            'n_estimators': [50, 100, 150, 200],  # 樹的數量
            'max_depth': [None, 10, 20, 30],  # 樹的最大深度
            'min_samples_split': [2, 5, 10],  # 分裂一個節點所需的最小樣本數
            'min_samples_leaf': [1, 2, 4],  # 在葉子節點中需要的最小樣本數
            'max_features': ['auto', 'sqrt', 'log2'],  # 每次分裂時考慮的最大特徵數
            'bootstrap': [True, False]  # 是否採用自助法
        }
        ]

        for i in range(len(model)):
            self.find_best_param(model[i],
                                 param[i],
                                 X_train,
                                 y_train,
                                 X_test)

    def find_best_param(self,model,model_param,X_train,y_train,X_test):
        """找尋模型最佳參數"""
        from sklearn.model_selection import GridSearchCV
        # 初始化支持向量機模型

        # 使用GridSearchCV進行網格搜尋
        grid_search = GridSearchCV(model, model_param, refit=True, verbose=0, cv=5)
        grid_search.fit(X_train, y_train)

        # 輸出最佳參數和最佳分數
        print("Best Parameters:", grid_search.best_params_)
        print("Best Score:", grid_search.best_score_)

        # 使用最佳參數進行預測
        best_model = grid_search.best_estimator_
        predictions = best_model.predict(X_test)

        # 輸出預測結果
        print("Predictions:", predictions)

    def save_toExcel(self):
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
        ).to_excel("./result/ClassiferResult.xlsx")
        print("輸出完畢")




if __name__ == "__main__":
    import warnings

    # 忽略所有警告  
    warnings.filterwarnings('ignore')
    Model_c().main(save=True)
    # Model_c().main_find()