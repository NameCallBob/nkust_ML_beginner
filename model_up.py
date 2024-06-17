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
from data import Data, Data_seoul
import pandas as pd


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
        eclf = VotingClassifier(
            estimators=[('lr', clf1), ('dt', clf2), ('svc', clf3)], voting='soft')

        # 訓練投票分類器
        eclf.fit(X_train, y_train)
        y_pred = eclf.predict(X_train)
        self.__output_score("votting", y_train, y_pred, 0)
        # 評估模型性能
        y_pred = eclf.predict(X_test)
        self.__output_score("votting", y_test, y_pred, 1)

        self.__drawOneROC(eclf,X_train, X_test, y_train, y_test,"Voting")


    def way2_ploy(self):
        """多項線性回歸"""
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.metrics import mean_squared_error, r2_score
        import matplotlib.pyplot as plt

        def draw_OriginScatterPic(title, X_origin, y, X_train, y_train_pred, X_test, y_test_pred):
            # 視覺化結果
            # 準備網格數據用於繪圖
            import numpy as np
            X_pred = np.linspace(0, 1, 100).reshape(-1, 1)
            X_pred_poly = poly.transform(X_pred)
            Y_pred = model.predict(X_pred_poly)
            # 繪製數據點和回歸曲線
            plt.scatter(X_train, y_train, color='blue', label='Training Data')
            plt.scatter(X_test, y_test, color='green', label='Test Data')
            plt.plot(X_pred, Y_pred, color='red',
                     label='Polynomial Regression Curve')
            plt.title(title)
            plt.legend()
            plt.xlabel('X_date')
            plt.ylabel('y_Rented Bike')
            plt.show()

        X, y, X_origin = Data_seoul().Source_trainingdata_getXY()
        X_train, X_test, y_train, y_test = Data_seoul().Source_trainingdata()

        # 建立多項式特徵
        poly = PolynomialFeatures(degree=2)
        # poly = PolynomialFeatures(degree=3)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        # Linear
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        # 預測
        y_train_pred = model.predict(X_train_poly)
        y_test_pred = model.predict(X_test_poly)

        self.__output_model_score("Ploy_Linear_train", y_train, y_train_pred)
        self.__output_model_score("Ploy_Linear_test", y_test, y_test_pred)

        self.__draw("ploy_linear", y_train, y_train_pred, y_test, y_test_pred)
        from sklearn.linear_model import Lasso

        # Lasso_l1
        model = Lasso()
        model.set_params(**
                         {'alpha': 0.1, 'copy_X': True, 'fit_intercept': True,
                             'max_iter': 100, 'tol': 0.001}
                         )
        model.fit(X_train_poly, y_train)
        # 預測
        y_train_pred = model.predict(X_train_poly)
        y_test_pred = model.predict(X_test_poly)

        self.__output_model_score("Ploy_Lasso_train", y_train, y_train_pred)
        self.__output_model_score("Ploy_Lasso_test", y_test, y_test_pred)
        self.__draw("ploy_lasso", y_train, y_train_pred, y_test, y_test_pred)

        # Ridge_L2
        from sklearn.linear_model import Ridge
        model = Ridge()
        model.set_params(**
                         {'alpha': 10.0, 'copy_X': True, 'fit_intercept': True,
                             'max_iter': 100, 'tol': 0.001}
                         )
        model.fit(X_train_poly, y_train)
        # 預測
        y_train_pred = model.predict(X_train_poly)
        y_test_pred = model.predict(X_test_poly)

        self.__output_model_score("Ploy_Ridge_train", y_train, y_train_pred)
        self.__output_model_score("Ploy_Ridge_test", y_test, y_test_pred)

        self.__draw("ploy_ridge", y_train, y_train_pred, y_test, y_test_pred)

    def __output_score(self, title, model_data, pred, status):
        """
        輸出模型績效指數
        @ title -> 模型標題
        @ model_data -> 模型資料
        @ pred -> 模型實際預測
        @ status -> 輸出模式(0:訓練集分數;1:測試集分數)
        """
        from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, roc_auc_score

        cm = confusion_matrix(model_data, pred)
        precision = precision_score(model_data, pred)
        accuracy = accuracy_score(model_data, pred)
        recall = recall_score(model_data, pred)
        f1 = f1_score(model_data, pred)
        auc = roc_auc_score(model_data, pred)
        # 輸出
        print(f"-----------{title}-----------")
        print("----測試資料之模型績效----" if status else "----訓練資料之模型績效----")
        print("Confusion Matrix:")
        print(cm)
        print()
        print("Precision:", precision)
        print("Accuracy:", accuracy)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("AUC:", auc)
        print("-----------END-----------")

    def __output_model_score(self, title, y_test, y_pred):
        """
        輸出模型績效
        """
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        print(f"-----------{title}-----------")
        print(f"MSE: {mse:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"MAE: {mae:.3f}")
        print(f"R^2: {r2:.3f}")
        print("-----------END-----------")

    def __draw(self, title, y_train, y_train_pred, y_test, y_test_pred):
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
        plt.scatter(y_train_pred, train_residuals,
                    color='blue', alpha=0.5, label='Train Data')

        # 繪製測試資料的殘差
        plt.scatter(y_test_pred, test_residuals, color='green',
                    alpha=0.5, label='Test Data')

        # 劃一條 y=0 的紅色水平線
        plt.axhline(y=0, color='red', linestyle='-')

        plt.title(f'Residual Plot_{title}')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.legend()
        plt.savefig(f'pic/output_{title}.png')
        plt.show()

    def __drawOneROC(self,model,X_train, X_test, y_train, y_test,model_name):
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
        y_score = model.predict_proba(X_test)[:, 1]
        # 計算ROC曲線
        fpr, tpr, _ = roc_curve(y_test, y_score)
        # 計算AUC
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(f'pic/output_{model_name}.png')
        plt.show()

if __name__ == "__main__":
    UP().way1_vote()
    # UP().way2_ploy()
