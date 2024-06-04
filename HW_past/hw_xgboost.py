from data import Data
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier , XGBRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor

import xgboost as xgb
from sklearn.model_selection import train_test_split,GridSearchCV
import matplotlib.pyplot as plt


class training_model_salary:
    def __init__(self) -> None:
        self.df = Data().data_encoder()
    def getTrainingData(self):
        return self.__SMOTE_training_data_split()
    def __training_data_split(self):
        """訓練資料分割"""
        X = self.df.drop(columns=['salary'])
        y = self.df["salary"]
        # 訓練資料分割
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test
    
    def __SMOTE_training_data_split(self):
        """使用SMOTE增加資料後進行訓練"""
        from imblearn.over_sampling import SMOTE
        # 分割特徵
        X = self.df.drop(columns=['salary'])
        y = self.df["salary"]
        # SMOTE
        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X, y)
        # 訓練資料分割
        X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test
    
    def XGB_classifer(self,X_train, X_test, y_train, y_test):
        """利用XGBoost的分類器進行訓練"""

        # 建立 XGBClassifier 模型
        xgboostModel = XGBClassifier(n_estimators=100, learning_rate= 0.3)
        # 使用訓練資料訓練模型
        xgboostModel.fit(X_train, y_train)
        # 使用訓練資料預測分類
        y_pred = xgboostModel.predict(X_test)
        
        self.__output_score("XGBoostModel",y_test,y_pred)
        self.draw_ROC_pic(xgboostModel,X_train, X_test, y_train, y_test)

    def adaboost_classifer(self,X_train, X_test, y_train, y_test):
        """利用adaboost進行訓練"""
        import numpy as np 
        base_estimator = DecisionTreeRegressor(max_depth=4)
        ada = AdaBoostClassifier(random_state=42)
        ada.fit(X_train, y_train)

        # 預測
        y_pred = ada.predict(X_test)
        # y_pred = np.argmax(y_pred,axis=1)
        self.__output_score("adaboost",y_test,y_pred)
        self.draw_ROC_pic(ada,X_train, X_test, y_train, y_test)


    def XGB_linear(self,X_train, X_test, y_train, y_test):
        """利用XGBRegressor的回歸進行訓練"""

        # 建立 XGBRegressor 模型
        xgboostModel = XGBRegressor(n_estimators=100, learning_rate= 0.3)
        # 使用訓練資料訓練模型
        xgboostModel.fit(X_train, y_train)
        # 使用訓練資料預測分類
        y_pred = xgboostModel.predict(X_test)

        self.__output_score("XGBRegressor",y_test,y_pred)


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

    def RF_importance(self,X_train,y_train):
        """找出欄位重要性並用圖表呈現"""
        # 計算特徵重要性
        dtrain = xgb.DMatrix(X_train, label=y_train)

        # 設定模型參數
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 4,
            'eta': 0.1
        }

        # 訓練模型
        bst = xgb.train(params, dtrain, num_boost_round=100)

        importance = bst.get_score(importance_type='weight')

        # 將特徵重要性排序
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        # 計算前50%的特徵數量
        top_features_count = int(len(sorted_importance) * 0.5)

        # 獲取前50%的特徵
        top_features = sorted_importance[:top_features_count]

        # 分別提取特徵名稱和其對應的重要性分數
        features, scores = zip(*top_features)

        # 繪製柱狀圖
        plt.figure(figsize=(10, 6))
        plt.barh(features, scores, color='skyblue')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.title('Top 50% Important Features')
        plt.gca().invert_yaxis()  # 反轉Y軸，使得最重要的特徵排在最上面
        plt.show()

    def findBestQuery(self,X_train,y_train):
        """找出最佳XGBoost參數"""

        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2]
        }

        xgb_clf = xgb.XGBClassifier(objective='multi:softprob', use_label_encoder=False)

        # 使用 GridSearchCV 進行參數搜索
        grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid,
                                scoring='accuracy', cv=5, verbose=1)

        # 執行參數搜索
        grid_search.fit(X_train, y_train)

        # 輸出最佳參數和最佳分數
        print(f"Best parameters found: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_}")


    def draw_ROC_pic(self,model,X_train, X_test, y_train, y_test,model_name):
        """繪製ROC"""
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

    def main(self):
        """主執行"""
        print("執行開始")
        X_train, X_test, y_train, y_test = self.__SMOTE_training_data_split()
        self.XGB_classifer( X_train, X_test, y_train, y_test)
        self.RF_importance(X_train,y_train)
        self.adaboost_classifer(X_train, X_test, y_train, y_test)
        
        

# class training_model_bike():
#     def __init__(self) -> None:
#         self.df = Data_seoul().data_clean()
    
#     def __training_data_split(self):
#         """訓練資料分割"""
#         X = self.df.drop(columns="Date")
#         y = self.df["Rented Bike Count"]
#         # 訓練資料分割
#         X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42)

#         return X_train, X_test, y_train, y_test
    
#     def __SMOTE_training_data_split(self):
#         """使用SMOTE增加資料後進行訓練"""
#         from imblearn.over_sampling import SMOTE
#         # 分割特徵
#         X = self.df.drop(columns="Date")
#         y = self.df["Rented Bike Count"]        
#         # SMOTE
#         smote = SMOTE()
#         X_resampled, y_resampled = smote.fit_resample(X, y)
#         # 訓練資料分割
#         X_train, X_test, y_train, y_test = train_test_split(
#         X_resampled, y_resampled, test_size=0.2, random_state=42)

#         return X_train, X_test, y_train, y_test
    
#     def XGB_classifer(self,X_train, X_test, y_train, y_test):
#         """利用XGBoost的分類器進行訓練"""

#         # 建立 XGBClassifier 模型
#         xgboostModel = XGBClassifier(n_estimators=100, learning_rate= 0.3)
#         # 使用訓練資料訓練模型
#         xgboostModel.fit(X_train, y_train)
#         # 使用訓練資料預測分類
#         y_pred = xgboostModel.predict(X_test)

#         self.__output_score("XGBoostModel",y_test,y_pred)


#     def XGB_linear(self,X_train, X_test, y_train, y_test):
#         """利用XGBRegressor的回歸進行訓練"""

#         # 建立 XGBRegressor 模型
#         xgboostModel = XGBRegressor(n_estimators=100, learning_rate= 0.3)
#         # 使用訓練資料訓練模型
#         xgboostModel.fit(X_train, y_train)
#         # 使用訓練資料預測分類
#         y_pred = xgboostModel.predict(X_test)

#         self.__output_score("XGBRegressor",y_test,y_pred)

#     def main(self):
#         """主執行"""
#         print("執行開始")
#         X_train, X_test, y_train, y_test = self.__training_data_split()
#         self.XGB_linear( X_train, X_test, y_train, y_test )

class RFmodel():

    def RF_importance(self,X_train, y_train):
        from sklearn.ensemble import RandomForestRegressor
        import pandas as pd 
        
        # 訓練隨機森林模型
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        # 獲取特徵重要性
        importance = rf.feature_importances_

        # 創建DataFrame
        feature_importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': importance
        })

        # 按重要性排序
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        # 計算前50%的特徵數量
        top_features_count = int(len(feature_importance_df) * 0.5)

        # 獲取前50%的特徵
        top_features = feature_importance_df.head(top_features_count)

        # 繪製柱狀圖
        plt.figure(figsize=(10, 6))
        plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.title('Top 50% Important Features')
        plt.gca().invert_yaxis()  # 反轉Y軸，使得最重要的特徵排在最上面
        plt.show()
        


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = training_model_salary().getTrainingData()
    print("RF")
    RFmodel().RF_importance(X_train, y_train)
    print("Xgboost and Adaboost")
    training_model_salary().main()
    # training_model_bike().main()