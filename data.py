"""千芬＿機器學習"""

"""
簡介
1.缺失值
1.5 欄位特徵設計
2.label 轉 代碼
3.分 訓練資料、學習資料
4.特徵縮放
"""

#套件
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder , MinMaxScaler

from sklearn.model_selection import train_test_split


class Data:

    def data_clean(self,output_log=False):
        """資料清洗"""
        df = pd.read_csv("./salary.csv")

        # print(df.isnull().sum())
        # print("資料無任一缺失值")
        return df

    def data_intro(self):
        """資料特徵介紹"""
        df = self.data_clean()
        print("數值資料詳細")
        numric_detail = df.describe()
        print("類別資料詳細")
        for column in df.select_dtypes(include='object').columns:
            print(f"\n{column}:")
            print(df[column].value_counts())

    def data_encoder(self):
        """資料預處理，含分割資料、資料轉換"""
        df = self.data_clean()
        label_encoder = LabelEncoder()
        for column in df.select_dtypes(include='object').columns:
            df[column] = label_encoder.fit_transform(df[column])
            # print(df[column])
        return df

    def data_split(self):
        origin_df = self.data_clean()
        encoded_df = self.data_encoder()
        print("挑選特徵欄位將資料分開") ; print("以下是分割後的訓練資料")
        X_train, X_val, y_train, y_val = train_test_split(encoded_df[['sex','race','education']], encoded_df['salary'], test_size=0.2, random_state=42)
        # print(X_train);print(y_train)
        return X_train, X_val, y_train, y_val

    def data_scaler(self):
        encoded_df = self.data_encoder()
        scaler = MinMaxScaler()
        scaler.fit(encoded_df)
        scaled_training_data = scaler.transform(encoded_df)
        print("縮放後的資料：")
        print(scaled_training_data)
        return scaled_training_data
    def main(self):
        """執行"""
        print("1.資料清洗")
        self.data_clean()
        print("2.資料特徵介紹")
        self.data_intro()
        print("3.LabelToNumeric")
        self.data_encoder()
        print("4.分資料")
        self.data_split()
        print("5.資料縮放")
        self.data_scaler()
