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
# 升級模型

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from data import Data , Data_seoul
class UP:
    def way1_vote(self):
        """升級方法-> 採集成訓練:VOTE"""
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

    def way2_ploy(self):
        """分段線性回歸"""
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.metrics import mean_squared_error, r2_score
        import matplotlib.pyplot as plt
        X,y = Data_seoul().Source_trainingdata_getXY()
        X_train, X_test, y_train, y_test = Data_seoul().Source_trainingdata()

        # 建立多項式特徵
        poly = PolynomialFeatures(degree=3)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        # 建立線性回歸模型
        model = LinearRegression()
        model.fit(X_train_poly, y_train)

        # 預測
        y_train_pred = model.predict(X_train_poly)
        y_test_pred = model.predict(X_test_poly)

        # 評估模型
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        print(f"訓練集 MSE: {train_mse}")
        print(f"測試集 MSE: {test_mse}")
        
        print(f"訓練集 R^2: {train_r2}")
        print(f"測試集 R^2: {test_r2}")

        # 視覺化結果
        plt.scatter(X['Humidity(%)'], y, color='black', label='數據點')
        plt.plot(X_train, y_train_pred, color='blue', linewidth=2, label='訓練集擬合')
        plt.plot(X_test, y_test_pred, color='red', linewidth=2, linestyle='dashed', label='測試集擬合')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.show()

        self.__draw("ploy",y_train, y_train_pred, y_test, y_test_pred)

    def way3_(self):
        """"""

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

    def __draw(self,title,y_train, y_train_pred, y_test, y_test_pred):
            """繪圖"""
            import matplotlib.pyplot as plt
            import os

            # 建立pic資料夾如果它不存在
            if not os.path.exists('pic'):
                os.makedirs('pic')

            # 計算殘差
            train_residuals = y_train - y_train_pred
            test_residuals = y_test - y_test_pred

            # 繪製殘差圖
            plt.figure(figsize=(10, 6))

            # 繪製訓練資料的殘差
            plt.scatter(y_train_pred, train_residuals, color='blue', alpha=0.5, label='Train Data')

            # 繪製測試資料的殘差
            plt.scatter(y_test_pred, test_residuals, color='green', alpha=0.5, label='Test Data')

            # 劃一條 y=0 的紅色水平線
            plt.axhline(y=0, color='red', linestyle='-')

            plt.title(f'Residual Plot_{title}')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.legend()
            plt.savefig(f'pic/output_{title}.png')
            plt.show()

if __name__ == "__main__":
    # UP().way1_vote()
    UP().way2_ploy()