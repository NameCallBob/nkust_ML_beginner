from sklearn.linear_model import LinearRegression
from data import Data_seoul



class model_l():

    def Linear():
        """線性回歸"""
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


    def main():
        pass





if __name__ == "__main__":
    model_l.get_Training_columns()