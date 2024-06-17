# -----------------------------------------------------------------------------
# Copyright (c) 2024 楊采穎
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

from matplotlib import font_manager
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 嘗試不同的編碼格式來讀取資料
try:
    df_salary = pd.read_csv('./SourceData/salary.csv', encoding='utf-8')
except UnicodeDecodeError:
    df_salary = pd.read_csv('./SourceData/salary.csv', encoding='latin1')

try:
    df_bike = pd.read_csv('./SourceData/SeoulBikeData.csv', encoding='utf-8')
except UnicodeDecodeError:
    df_bike = pd.read_csv('./SourceData/SeoulBikeData.csv', encoding='latin1')

# 計算樣本數、平均和標準差 - 薪資
numeric_features_salary = df_salary.select_dtypes(include=[np.number]).columns
statistics_salary = df_salary[numeric_features_salary].describe().loc[[
    'count', 'mean', 'std']]

print("薪資的樣本數、平均和標準差:")
print(statistics_salary)

# 計算樣本數、平均和標準差 - 首爾腳踏車
numeric_features_bike = df_bike.select_dtypes(include=[np.number]).columns
statistics_bike = df_bike[numeric_features_bike].describe().loc[[
    'count', 'mean', 'std']]

print("首爾腳踏車的樣本數、平均和標準差:")
print(statistics_bike)

# 創建輸出資料夾
output_dir = 'pic'
os.makedirs(output_dir, exist_ok=True)

# 設置中文字體，這裡假設 SimHei.ttf 放置在腳本相同目錄下
font_path = 'SimHei.ttf'
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 生成各類別樣本數的散點圖矩陣 - 薪資
pairplot_salary = sns.pairplot(df_salary)
plt.suptitle("各類別樣本數散點圖矩陣 - 薪資", y=1.02)
pairplot_salary_path = os.path.join(output_dir, 'scatter_matrix_salary.png')
pairplot_salary.savefig(pairplot_salary_path)
plt.show()

# 生成各類別樣本數的散點圖矩陣 - 首爾腳踏車
pairplot_bike = sns.pairplot(df_bike)
plt.suptitle("各類別樣本數散點圖矩陣 - 首爾腳踏車", y=1.02)
pairplot_bike_path = os.path.join(output_dir, 'scatter_matrix_bike.png')
pairplot_bike.savefig(pairplot_bike_path)
plt.show()

# 相關係數矩陣 - 薪資
corr_matrix_salary = df_salary.corr()

# 生成相關係數矩陣的熱圖 - 薪資
plt.figure(figsize=(10, 8))
heatmap_salary = sns.heatmap(
    corr_matrix_salary, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('相關係數矩陣熱圖 - 薪資')
heatmap_salary_path = os.path.join(
    output_dir, 'correlation_heatmap_salary.png')
plt.savefig(heatmap_salary_path)
plt.show()

# 相關係數矩陣 - 首爾腳踏車
corr_matrix_bike = df_bike.corr()

# 生成相關係數矩陣的熱圖 - 首爾腳踏車
plt.figure(figsize=(10, 8))
heatmap_bike = sns.heatmap(
    corr_matrix_bike, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('相關係數矩陣熱圖 - 首爾腳踏車')
heatmap_bike_path = os.path.join(output_dir, 'correlation_heatmap_bike.png')
plt.savefig(heatmap_bike_path)
plt.show()

print(f"薪資的散點圖矩陣已保存到 {pairplot_salary_path}")
print(f"薪資的相關係數矩陣熱圖已保存到 {heatmap_salary_path}")
print(f"首爾腳踏車的散點圖矩陣已保存到 {pairplot_bike_path}")
print(f"首爾腳踏車的相關係數矩陣熱圖已保存到 {heatmap_bike_path}")
