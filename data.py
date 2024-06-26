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
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder , MinMaxScaler

from sklearn.model_selection import train_test_split


class Data:

    def data_clean(self,output_log=False):
        """資料清洗"""
        df = pd.read_csv("./SourceData/salary.csv")
        # df2 = pd.read_csv("")

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
        """
        salary 正與負
        <=50K    24720
        >50K      7841
        """

    def data_encoder(self):
        """資料預處理，含分割資料、資料轉換"""
        import numpy as np
        df = self.data_clean()
        label_encoder = LabelEncoder()
        scaler = StandardScaler()
        for column in df.select_dtypes(include=['float64', 'int64']).columns:
            tmp = np.array(df[column])
            df[column] = scaler.fit_transform(tmp.reshape(-1,1))

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

    def Source_trainingdata(self,selected=False):
        """原資料得訓練資料"""
        self.df = self.data_encoder()
        if selected:
            X = self.df[['"age","workclass","education","occupation"']]
        else:
            X = self.df.drop(columns=['salary'])

        y = self.df["salary"]

        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

    def SMOTE_PCA_trainingData(self,selected=False):
        from sklearn.decomposition import PCA
        from imblearn.over_sampling import SMOTE

        self.df = self.data_encoder()
        # 分割特徵
        if selected:
            X = self.df[["age","workclass","education","occupation"]]
        else:
            X = self.df.drop(columns=['salary'])
        y = self.df["salary"]

        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X, y)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_resampled)

        # 分割數據集
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y_resampled, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test

    def SMOTE_fitted_trainingdata(self,selected=False):
        """
        利用SMOTE取得相同比例的資料
        @params selected -> 是否使用5折特徵
        """
        from imblearn.over_sampling import SMOTE
        self.df = self.data_encoder()
        # 分割特徵
        if selected:
            X = self.df[["age","workclass","education","occupation"]]
        else:
            X = self.df.drop(columns=['salary'])
        y = self.df["salary"]
        # SMOTE
        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X, y)
        # 訓練資料分割
        X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

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



class Data_seoul():
    """資料-首爾單車"""

    def data_clean(self,output_log=False):
        """資料清洗"""
        df = pd.read_csv("./SourceData/SeoulBikeData.csv",encoding='unicode_escape')
        # 先篩選單車工作日
        df = df[df['Functioning Day'] == "Yes"]
        # 空值填補平均值
        for column in df.columns:
            if df[column].dtype == 'float64' or df[column].dtype == 'int64':
                if df[column].isnull().values.any():
                    mean_value = df[column].mean()
                    df[column].fillna(mean_value, inplace=True)
        return df

    def data_intro(self):
        """資料特徵介紹"""
        df = self.data_clean()

        print("資料欄位")
        print(df.columns)

        print("數值資料詳細")
        numric_detail = df.describe()
        print(numric_detail)
        print("類別資料詳細")
        for column in df.select_dtypes(include='object').columns:
            print(f"\n{column}:")
            print(df[column].value_counts())

    def data_encoder(self):
        """資料預處理，含分割資料、資料轉換"""
        import numpy as np
        df = self.data_clean()
        label_encoder = LabelEncoder()
        scaler = StandardScaler()
        # for column in df.select_dtypes(include=['float64', 'int64']).columns:
        #     tmp = np.array(df[column])
        #     df[column] = scaler.fit_transform(tmp.reshape(-1,1))

        for column in df.select_dtypes(include='object').columns:
            df[column] = label_encoder.fit_transform(df[column])
            # 字典內容
            # le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

        """
        字典內容：
        {'01/01/2018': 0, '01/02/2018': 1, '01/03/2018': 2, '01/04/2018': 3, '01/05/2018': 4, '01/06/2018': 5, '01/07/2018': 6, '01/08/2018': 7, '01/09/2018': 8, '01/10/2018': 9, '01/11/2018': 10, '01/12/2017': 11, '02/01/2018': 12, '02/02/2018': 13, '02/03/2018': 14, '02/04/2018': 15, '02/05/2018': 16, '02/06/2018': 17, '02/07/2018': 18, '02/08/2018': 19, '02/09/2018': 20, '02/10/2018': 21, '02/11/2018': 22, '02/12/2017': 23, '03/01/2018': 24, '03/02/2018': 25, '03/03/2018': 26, '03/04/2018': 27, '03/05/2018': 28, '03/06/2018': 29, '03/07/2018': 30, '03/08/2018': 31, '03/09/2018': 32, '03/10/2018': 33, '03/11/2018': 34, '03/12/2017': 35, '04/01/2018': 36, '04/02/2018': 37, '04/03/2018': 38, '04/04/2018': 39, '04/05/2018': 40, '04/06/2018': 41, '04/07/2018': 42, '04/08/2018': 43, '04/09/2018': 44, '04/10/2018': 45, '04/11/2018': 46, '04/12/2017': 47, '05/01/2018': 48, '05/02/2018': 49, '05/03/2018': 50, '05/04/2018': 51, '05/05/2018': 52, '05/06/2018': 53, '05/07/2018': 54, '05/08/2018': 55, '05/09/2018': 56, '05/10/2018': 57, '05/11/2018': 58, '05/12/2017': 59, '06/01/2018': 60, '06/02/2018': 61, '06/03/2018': 62, '06/04/2018': 63, '06/05/2018': 64, '06/06/2018': 65, '06/07/2018': 66, '06/08/2018': 67, '06/09/2018': 68, '06/10/2018': 69, '06/11/2018': 70, '06/12/2017': 71, '07/01/2018': 72, '07/02/2018': 73, '07/03/2018': 74, '07/04/2018': 75, '07/05/2018': 76, '07/06/2018': 77, '07/07/2018': 78, '07/08/2018': 79, '07/09/2018': 80, '07/10/2018': 81, '07/11/2018': 82, '07/12/2017': 83, '08/01/2018': 84, '08/02/2018': 85, '08/03/2018': 86, '08/04/2018': 87, '08/05/2018': 88, '08/06/2018': 89, '08/07/2018': 90, '08/08/2018': 91, '08/09/2018': 92, '08/10/2018': 93, '08/11/2018': 94, '08/12/2017': 95, '09/01/2018': 96, '09/02/2018': 97, '09/03/2018': 98, '09/04/2018': 99, '09/05/2018': 100, '09/06/2018': 101, '09/07/2018': 102, '09/08/2018': 103, '09/09/2018': 104, '09/10/2018': 105, '09/11/2018': 106, '09/12/2017': 107, '10/01/2018': 108, '10/02/2018': 109, '10/03/2018': 110, '10/04/2018': 111, '10/05/2018': 112, '10/06/2018': 113, '10/07/2018': 114, '10/08/2018': 115, '10/09/2018': 116, '10/10/2018': 117, '10/11/2018': 118, '10/12/2017': 119, '11/01/2018': 120, '11/02/2018': 121, '11/03/2018': 122, '11/04/2018': 123, '11/05/2018': 124, '11/06/2018': 125, '11/07/2018': 126, '11/08/2018': 127, '11/09/2018': 128, '11/10/2018': 129, '11/11/2018': 130, '11/12/2017': 131, '12/01/2018': 132, '12/02/2018': 133, '12/03/2018': 134, '12/04/2018': 135, '12/05/2018': 136, '12/06/2018': 137, '12/07/2018': 138, '12/08/2018': 139, '12/09/2018': 140, '12/10/2018': 141, '12/11/2018': 142, '12/12/2017': 143, '13/01/2018': 144, '13/02/2018': 145, '13/03/2018': 146, '13/04/2018': 147, '13/05/2018': 148, '13/06/2018': 149, '13/07/2018': 150, '13/08/2018': 151, '13/09/2018': 152, '13/10/2018': 153, '13/11/2018': 154, '13/12/2017': 155, '14/01/2018': 156, '14/02/2018': 157, '14/03/2018': 158, '14/04/2018': 159, '14/05/2018': 160, '14/06/2018': 161, '14/07/2018': 162, '14/08/2018': 163, '14/09/2018': 164, '14/10/2018': 165, '14/11/2018': 166, '14/12/2017': 167, '15/01/2018': 168, '15/02/2018': 169, '15/03/2018': 170, '15/04/2018': 171, '15/05/2018': 172, '15/06/2018': 173, '15/07/2018': 174, '15/08/2018': 175, '15/09/2018': 176, '15/10/2018': 177, '15/11/2018': 178, '15/12/2017': 179, '16/01/2018': 180, '16/02/2018': 181, '16/03/2018': 182, '16/04/2018': 183, '16/05/2018': 184, '16/06/2018': 185, '16/07/2018': 186, '16/08/2018': 187, '16/09/2018': 188, '16/10/2018': 189, '16/11/2018': 190, '16/12/2017': 191, '17/01/2018': 192, '17/02/2018': 193, '17/03/2018': 194, '17/04/2018': 195, '17/05/2018': 196, '17/06/2018': 197, '17/07/2018': 198, '17/08/2018': 199, '17/09/2018': 200, '17/10/2018': 201, '17/11/2018': 202, '17/12/2017': 203, '18/01/2018': 204, '18/02/2018': 205, '18/03/2018': 206, '18/04/2018': 207, '18/05/2018': 208, '18/06/2018': 209, '18/07/2018': 210, '18/08/2018': 211, '18/09/2018': 212, '18/10/2018': 213, '18/11/2018': 214, '18/12/2017': 215, '19/01/2018': 216, '19/02/2018': 217, '19/03/2018': 218, '19/04/2018': 219, '19/05/2018': 220, '19/06/2018': 221, '19/07/2018': 222, '19/08/2018': 223, '19/09/2018': 224, '19/10/2018': 225, '19/11/2018': 226, '19/12/2017': 227, '20/01/2018': 228, '20/02/2018': 229, '20/03/2018': 230, '20/04/2018': 231, '20/05/2018': 232, '20/06/2018': 233, '20/07/2018': 234, '20/08/2018': 235, '20/09/2018': 236, '20/10/2018': 237, '20/11/2018': 238, '20/12/2017': 239, '21/01/2018': 240, '21/02/2018': 241, '21/03/2018': 242, '21/04/2018': 243, '21/05/2018': 244, '21/06/2018': 245, '21/07/2018': 246, '21/08/2018': 247, '21/09/2018': 248, '21/10/2018': 249, '21/11/2018': 250, '21/12/2017': 251, '22/01/2018': 252, '22/02/2018': 253, '22/03/2018': 254, '22/04/2018': 255, '22/05/2018': 256, '22/06/2018': 257, '22/07/2018': 258, '22/08/2018': 259, '22/09/2018': 260, '22/10/2018': 261, '22/11/2018': 262, '22/12/2017': 263, '23/01/2018': 264, '23/02/2018': 265, '23/03/2018': 266, '23/04/2018': 267, '23/05/2018': 268, '23/06/2018': 269, '23/07/2018': 270, '23/08/2018': 271, '23/09/2018': 272, '23/10/2018': 273, '23/11/2018': 274, '23/12/2017': 275, '24/01/2018': 276, '24/02/2018': 277, '24/03/2018': 278, '24/04/2018': 279, '24/05/2018': 280, '24/06/2018': 281, '24/07/2018': 282, '24/08/2018': 283, '24/09/2018': 284, '24/10/2018': 285, '24/11/2018': 286, '24/12/2017': 287, '25/01/2018': 288, '25/02/2018': 289, '25/03/2018': 290, '25/04/2018': 291, '25/05/2018': 292, '25/06/2018': 293, '25/07/2018': 294, '25/08/2018': 295, '25/09/2018': 296, '25/10/2018': 297, '25/11/2018': 298, '25/12/2017': 299, '26/01/2018': 300, '26/02/2018': 301, '26/03/2018': 302, '26/04/2018': 303, '26/05/2018': 304, '26/06/2018': 305, '26/07/2018': 306, '26/08/2018': 307, '26/09/2018': 308, '26/10/2018': 309, '26/11/2018': 310, '26/12/2017': 311, '27/01/2018': 312, '27/02/2018': 313, '27/03/2018': 314, '27/04/2018': 315, '27/05/2018': 316, '27/06/2018': 317, '27/07/2018': 318, '27/08/2018': 319, '27/09/2018': 320, '27/10/2018': 321, '27/11/2018': 322, '27/12/2017': 323, '28/01/2018': 324, '28/02/2018': 325, '28/03/2018': 326, '28/04/2018': 327, '28/05/2018': 328, '28/06/2018': 329, '28/07/2018': 330, '28/08/2018': 331, '28/09/2018': 332, '28/10/2018': 333, '28/11/2018': 334, '28/12/2017': 335, '29/01/2018': 336, '29/03/2018': 337, '29/04/2018': 338, '29/05/2018': 339, '29/06/2018': 340, '29/07/2018': 341, '29/08/2018': 342, '29/09/2018': 343, '29/10/2018': 344, '29/11/2018': 345, '29/12/2017': 346, '30/01/2018': 347, '30/03/2018': 348, '30/04/2018': 349, '30/05/2018': 350, '30/06/2018': 351, '30/07/2018': 352, '30/08/2018': 353, '30/09/2018': 354, '30/10/2018': 355, '30/11/2018': 356, '30/12/2017': 357, '31/01/2018': 358, '31/03/2018': 359, '31/05/2018': 360, '31/07/2018': 361, '31/08/2018': 362, '31/10/2018': 363, '31/12/2017': 364}
        {'Autumn': 0, 'Spring': 1, 'Summer': 2, 'Winter': 3}
        {'Holiday': 0, 'No Holiday': 1}
        {'No': 0, 'Yes': 1}
        """

        return df

    def Source_trainingdata_forClassifer(self,selected=False):
        """將Rented Bike變為類別"""
        self.df = self.data_encoder()
        # 計算四分位數
        q1 = self.df['Rented Bike Count'].quantile(0.25)
        q2 = self.df['Rented Bike Count'].quantile(0.5)
        q3 = self.df['Rented Bike Count'].quantile(0.75)
        # 定義分類函數
        def classify(value):
            if value <= q1:
                # return 'Very Low Rented'
                return 0
            elif value <= q2:
                # return 'Low Rented'
                return 1
            elif value <= q3:
                # return 'High Rented'
                return 2
            else:
                # return 'Very High Rented'
                return 3

        # 將分類應用到DataFrame中
        self.df['Rented Bike Count Classifer'] = self.df['Rented Bike Count'].apply(classify)
        X = self.df.drop(columns=[
            "Rented Bike Count","Date","Functioning Day","Rented Bike Count Classifer"
        ])
        y = self.df["Rented Bike Count Classifer"]


        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

    def Source_trainingdata(self,selected=False):
        """原資料得訓練資料"""
        self.df = self.data_encoder()

        X = self.df.drop(columns=["Rented Bike Count","Date","Functioning Day"])
        y = self.df["Rented Bike Count"]

        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
        print("資料OK")
        return X_train, X_test, y_train, y_test

    def Source_trainingdata_getXY(self,selected=False):
        """拿取XY"""
        self.df = self.data_encoder()
        tmp_df = self.data_clean()
        X = self.df.drop(columns=["Rented Bike Count","Date","Functioning Day"])
        X_origin = tmp_df.drop(columns=["Rented Bike Count","Functioning Day"])
        y = self.df["Rented Bike Count"]
        return X,y,X_origin



if __name__ == "__main__":
    pass
    # Data().data_intro()
    df = Data_seoul().data_encoder()
    # Data_seoul().data_intro()
