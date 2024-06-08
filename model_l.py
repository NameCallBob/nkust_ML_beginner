from data import Data_seoul

class model_l():
    def __init__(self) -> None:
            data = Data_seoul()
            self.resource_data = data.Source_trainingdata()
            self.modelScore = {
                "name":[],
                "score":{
                    "MAE":[],
                    "MSE":[],
                    "R^2":[],
                    "RMSE":[]
                }
            }
    def Linear(self,X_train, X_test, y_train, y_test):
        """線性回歸"""
        from sklearn.linear_model import LinearRegression
        params = {'copy_X': True, 'fit_intercept': True, 'n_jobs': -1}
        model = LinearRegression()
        model.set_params(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        self.__output_model_score("LinearRegression",y_test,y_pred)
        self.__draw("LinearRegression",y_test,y_pred)

    def Elastic_net(self,X_train, X_test, y_train, y_test):
        from sklearn.linear_model import ElasticNet
        params ={'alpha': 0.1, 'copy_X': True, 'fit_intercept': True, 'l1_ratio': 0.9}
        model = ElasticNet()
        model.set_params(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        self.__output_model_score("ElasticNet",y_test,y_pred)
        self.__draw("ElasticNet",y_test,y_pred)
    def RidgeRegressor(self,X_train, X_test, y_train, y_test):
        from sklearn.linear_model import Ridge
        param =  {'alpha': 10.0, 'copy_X': True, 'fit_intercept': True, 'max_iter': 100, 'tol': 0.001}
        # 建立Ridge模型並訓練
        ridge_model = Ridge()
        ridge_model.set_params(**param)
        ridge_model.fit(X_train, y_train)

        # 進行預測
        y_pred = ridge_model.predict(X_test)
        self.__output_model_score("Ridge",y_test,y_pred)
        self.__draw("Ridge",y_test,y_pred)

    def LassoRegressor(self,X_train, X_test, y_train, y_test):
        from sklearn.linear_model import Lasso
        param = {'alpha': 0.1, 'copy_X': True, 'fit_intercept': True, 'max_iter': 100, 'tol': 0.001}
        # 建立LASSO模型並訓練
        lasso_model = Lasso()
        lasso_model.set_params(**param)
        lasso_model.fit(X_train, y_train)

        # 進行預測
        y_pred = lasso_model.predict(X_test)
        self.__output_model_score("Lasso",y_test,y_pred)
        self.__draw("Lasso",y_test,y_pred)

    def Xgboost_Regressor(self,X_train, X_test, y_train, y_test):
        """
        Xgboost
        """
        from xgboost import XGBRegressor
        param = {'colsample_bytree': 1.0, 'gamma': 0, 'learning_rate': 0.05, 'max_depth': 7, 'min_child_weight': 3, 'n_estimators': 200, 'subsample': 0.6}
        model = XGBRegressor(
                    colsample_bytree = 1.0,
                    gamma = 0,
                    learning_rate = 0.05 ,
                    max_depth = 7,
                    min_child_weight = 3 ,
                    n_estimators = 200,
                    subsample = 0.6,
                    random_state=42
                )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        self.__output_model_score("XGBoostRegressor",y_test,y_pred)
        self.__draw("XGboostRegessor",y_test,y_pred)

    def Adaboost_Regressor(self,X_train, X_test, y_train, y_test):
        from sklearn.ensemble import AdaBoostRegressor

        # 創建 AdaBoostRegressor 模型
        model = AdaBoostRegressor(
            learning_rate=0.1,
            loss = 'linear',
            n_estimators=100,
            random_state=42
        )
        # 訓練模型
        model.fit(X_train, y_train)
        # 評估模型
        y_pred = model.predict(X_test)

        self.__output_model_score("AdaboostRegressor",y_test,y_pred)
        self.__draw("AdaBoostRegressor",y_test,y_pred)

    def train(self):
        pass

    def get_Training_columns():
        """
        獲得重點模型特徵
        """
        from sklearn.model_selection import cross_val_score ,train_test_split
        import matplotlib.pyplot as plt

        df = Data_seoul().data_encoder()
        X = df.drop(columns=["Rented Bike Count"])
        y = df["Rented Bike Count"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor()
        model.fit(X_train,y_train)
        print(X.columns)
        # 獲取特徵重要性
        feature_importances = model.feature_importances_
        feature_names = X.columns[:-1] # 排除目標變量

        # 將特徵與其重要性打包成list
        feature_importance_pairs = list(zip(feature_names, feature_importances))

        # 按重要性排序
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)

        # 繪製直方圖
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(feature_importance_pairs)), [pair[1] for pair in feature_importance_pairs])
        plt.xticks(range(len(feature_importance_pairs)), [pair[0] for pair in feature_importance_pairs], rotation=90)
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.title('Feature Importance')
        plt.show()

    def __output_model_score(self,title,y_test,y_pred):
        """
        輸出模型績效
        """
        from sklearn.metrics import mean_squared_error, r2_score , mean_absolute_error

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test,y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        print(f"-----------{title}-----------")
        print(f"MSE: {mse:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"MAE: {mae:.3f}")
        print(f"R^2: {r2:.3f}")
        print("-----------END-----------")
        # 儲存到暫存
        self.modelScore["name"].append(title)
        self.modelScore["score"]["MAE"].append(mae)
        self.modelScore["score"]["MSE"].append(mse)
        self.modelScore["score"]["RMSE"].append(rmse)
        self.modelScore["score"]["R^2"].append(r2)

    def __draw(self,title,y_test,y_pred):
        """繪圖"""
        import matplotlib.pyplot as plt
        # 繪製原始數據點
        # 繪製原始數據點和預測值

        # 計算殘差
        residuals = y_test - y_pred

        # 繪製殘差圖
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, color='blue')
        plt.axhline(y=0, color='red', linestyle='-')
        plt.title(f'Residual Plot_{title}')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.show()

    def ModelScoreToExcel(self):
        """將模型績效進行儲存"""
        import pandas as pd
        if len(self.modelScore["name"]) == 0 :
            print("無儲存任意模型績效")
            return
        df = pd.DataFrame(
            {
                "name":self.modelScore["name"],
                "MAE":self.modelScore["score"]["MAE"],
                "MSE":self.modelScore["score"]["MSE"],
                "RMSE":self.modelScore["score"]["RMSE"],
                "R^2":self.modelScore["score"]["R^2"]
            }
        )
        df.to_excel("./result/RegressorResult.xlsx")
        print("儲存成功")


    def main(self,use_smote=False):
        if not self.resource_data:
            print("未取得資料，終止執行");
            return;

        X_train, X_test, y_train, y_test = self.resource_data

        self.Linear(X_train, X_test, y_train, y_test)
        self.Elastic_net(X_train, X_test, y_train, y_test)
        self.RidgeRegressor(X_train, X_test, y_train, y_test)
        self.LassoRegressor(X_train, X_test, y_train, y_test)
        self.Adaboost_Regressor(X_train, X_test, y_train, y_test)
        self.Xgboost_Regressor(X_train, X_test, y_train, y_test)

        self.ModelScoreToExcel()


    def get_best_params(self):
        import json
        from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.model_selection import GridSearchCV
        from xgboost import XGBRegressor

        X_train, X_test, y_train, y_test = self.resource_data
        with open('./model_train_params.json', 'r', encoding='utf-8') as file:
            data = json.load(file)['regressor']
            ml_model = [
                LinearRegression(),Lasso(),Ridge(),
                ElasticNet(),AdaBoostRegressor(),XGBRegressor()
            ]
            ml_model_p = [
                data['linear'],data['lasso'],data['ridge'],
                data['elastic'],data['ada'],data['xgb']
            ]

            for i in range(len(ml_model)):
                grid_search = GridSearchCV(ml_model[i], ml_model_p[i], cv=5)
                grid_search.fit(X_train, y_train)
                print("Best Parameters:", grid_search.best_params_)





if __name__ == "__main__":
    import warnings

    # 忽略所有警告
    warnings.filterwarnings('ignore')
    model_l().main()
    # model_l().get_best_params()