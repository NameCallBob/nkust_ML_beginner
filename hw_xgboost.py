from data import Data
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split,GridSearchCV

class training_model:
    def __init__(self) -> None:
        self.df2 = Data()
        self.df = Data().data_encoder()

    def __training_data_split(self):
        """訓練資料分割"""
        X = self.df[["age","workclass","education","occupation"]]
        y = self.df["salary"]
        # 訓練資料分割
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test
    
    def __SMOTE_training_data_split(self):
        """使用SMOTE增加資料後進行訓練"""
        from imblearn.over_sampling import SMOTE
        # 分割特徵
        X = self.df[["age","workclass","education","occupation"]]
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


    def XGB_linear(self,X_train, X_test, y_train, y_test):
        """利用XGBoost的回歸進行訓練"""


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

    def draw_ROC_pic(self,X,y):
        """繪製ROC"""
        import numpy as np
        import matplotlib.pyplot as plt
        X=np.array(X)        
        y=np.array(y)    

        # 繪圖。   
        plt.title('Receiver Operating Characteristic')
        plt.plot(X, y, color = 'orange')
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()  

    def main(self):
        """主執行"""
        print("執行開始")
        X_train, X_test, y_train, y_test = self.__training_data_split()
        self.XGB_classifer( X_train, X_test, y_train, y_test )
    

if __name__ == "__main__":
    training_model().main()