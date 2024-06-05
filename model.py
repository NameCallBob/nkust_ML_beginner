

from data import Data

class Model_c():
    """集結所有機器訓練之模型"""

    def __init__(self) -> None:
        data = Data()
        self.smote_data = data.SMOTE_fitted_trainingdata()
        self.resource_data = data.Source_trainingdata()

    def logistic(self,X_train, X_test, y_train, y_test):
        """羅吉斯回歸"""
        from sklearn.linear_model import LogisticRegression
        param = {'C': 10,
             'penalty': 'l1'
             ,'solver': 'liblinear'
             }
        model = LogisticRegression()
        model.set_params(**param)
        # 訓練模型
        model.fit(X_train, y_train)

        # 預測測試集
        y_pred = model.predict(X_test)

        self.__output_score("Logistic羅吉斯",y_test,y_pred)
        self.draw_ROC_pic(model,X_train, X_test, y_train, y_test,"Logistic")

    def DecisionTree(self,X_train, X_test, y_train, y_test):
        """決策樹"""
        from sklearn.tree import DecisionTreeClassifier
        param = {'criterion': 'gini',
             'max_depth': 10,
             'max_features': None,
             'max_leaf_nodes': None,
             'min_samples_leaf': 1,
             'min_samples_split': 2,
             'splitter': 'best'}
        model = DecisionTreeClassifier(
                                       )
        model.set_params(**param)
        # 訓練模型
        model.fit(X_train, y_train)

        # 預測測試集
        y_pred = model.predict(X_test)
        self.__output_score("Decision決策樹",y_test,y_pred)
        self.draw_ROC_pic(model,X_train, X_test, y_train, y_test,"DecisionTree")

    def RandomForest(self,X_train, X_test, y_train, y_test):
        """隨機森林"""
        from sklearn.ensemble import RandomForestClassifier
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train, y_train)
        y_pred = rf_classifier.predict(X_test)
        self.__output_score("ＲＦ隨機森林",y_test,y_pred)
        self.draw_ROC_pic(rf_classifier,X_train, X_test, y_train, y_test,"RandomForest")

    def Adaboost_classifer(self,X_train, X_test, y_train, y_test):
        """"""
        from sklearn.ensemble import AdaBoostClassifier
        ada = AdaBoostClassifier(random_state=42)
        ada.fit(X_train, y_train)

        # 預測
        y_pred = ada.predict(X_test)
        # y_pred = np.argmax(y_pred,axis=1)
        self.__output_score("Adaboost",y_test,y_pred)
        self.draw_ROC_pic(ada,X_train, X_test, y_train, y_test,"adaboost")

    def XGBoost_classifer(self,X_train, X_test, y_train, y_test):
        """XGBboost 分類"""
        from xgboost import XGBClassifier

        # 建立 XGBClassifier 模型
        xgboostModel = XGBClassifier(n_estimators=100, learning_rate= 0.3)
        # 使用訓練資料訓練模型
        xgboostModel.fit(X_train, y_train)
        # 使用訓練資料預測分類
        y_pred = xgboostModel.predict(X_test)

        self.__output_score("XGBoostModel",y_test,y_pred)
        self.draw_ROC_pic(xgboostModel,X_train, X_test, y_train, y_test,"XGBboost classifer")

    def XGBoost_linear(self,X_train, X_test, y_train, y_test):
        """XGBboost 線性"""
        from xgboost import  XGBRegressor

        # 建立 XGBRegressor 模型
        xgboostModel = XGBRegressor(n_estimators=100, learning_rate= 0.3)
        # 使用訓練資料訓練模型
        xgboostModel.fit(X_train, y_train)
        # 使用訓練資料預測分類
        y_pred = xgboostModel.predict(X_test)

        self.__output_score("XGBRegressor",y_test,y_pred)

    def SVM(self,X_train, X_test, y_train, y_test):
        """向量機模型"""
        from sklearn.svm import SVC
        svm_classifier = SVC(
            probability=True
        )

        svm_classifier.fit(X_train, y_train)

        y_pred = svm_classifier.predict(X_test)

        self.__output_score("SVM 向量機學習",y_test,y_pred)
        self.draw_ROC_pic(svm_classifier,X_train, X_test, y_train, y_test,"SVM model")

    def KNN(self,X_train, X_test, y_train, y_test):
        """KNN模型"""
        from sklearn.neighbors import KNeighborsClassifier
        param = {'algorithm': 'ball_tree', 'leaf_size': 20, 'n_neighbors': 3, 'p': 1, 'weights': 'distance'}
        knn_classifier = KNeighborsClassifier()
        knn_classifier.set_params(**param)
        knn_classifier.fit(X_train, y_train)
        y_pred = knn_classifier.predict(X_test)

        self.__output_score("KNN模型",y_test,y_pred)
        self.draw_ROC_pic(knn_classifier,X_train, X_test, y_train, y_test,"KNN model")


    def all_score(self,X_train,y_train,X_test,y_test):
        """列出之模型績效"""
        pass

    def draw_ROC_pic(self,model,X_train, X_test, y_train, y_test,model_name):
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
        y_score = model.predict_proba(X_test)[:, 1]
        # 計算ROC曲線
        xgb_fpr, xgb_tpr, _ = roc_curve(y_test, y_score)

        # 計算AUC
        xgb_auc = auc(xgb_fpr, xgb_tpr)

        # 繪製ROC曲線
        plt.figure()
        plt.plot(xgb_fpr, xgb_tpr, color='blue', lw=2, label=f'{model_name} ROC curve (area = {xgb_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def __output_score(self,title,y_test,y_pred):
        """
        輸出模型績效指數
        @ title -> 模型標題
        @ y_test -> 模型訓練集
        @ y_pred -> 模型實際預測
        """
        from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score , roc_auc_score

        cm = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        # 輸出
        print(f"-----------{title}-----------")
        print("Confusion Matrix:")
        print(cm)
        print()
        print("Precision:", precision)
        print("Accuracy:", accuracy)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("AUC:", auc)
        print("-----------END-----------")


    def main(self,use_smote=True):
        """主執行"""
        if not self.smote_data or not self.resource_data:
            print("未取得資料，終止執行");
        if use_smote:
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




if __name__ == "__main__":
    Model_c().main()
    # AllModel().main_find()