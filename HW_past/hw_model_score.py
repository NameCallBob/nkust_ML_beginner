# 模型績效

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score , roc_auc_score

from data import Data



def model_training(X_train, X_test, y_train, y_test, model):
    """簡單的模型訓練"""
    model = model

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    precision = precision_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Confusion Matrix:")
    print(cm)
    print("Precision:", precision)
    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("F1 Score:", f1)
    """
    羅吉斯 ->
    Confusion Matrix:
    [[4660  282]
    [1107  464]]
    Precision: 0.6219839142091153
    Accuracy: 0.7867342238599724
    Recall: 0.29535327816677276
    F1 Score: 0.40051791109192925

    筆記:
    CM的True Negative (4660) 和 True Positive (464) 都相對較高
    但發現False Negative (1107) 的數量較高，這表示模型在捕捉實際正例方面可能存在一些問題。
    Recall (召回率)：0.295，表示模型僅捕捉到了實際正例的29.5%。
    F1 Score：0.401，是Precision和Recall的平衡值，較低的F1 Score可能表明模型在Precision和Recall之間沒有很好的平衡。

    SVM ->
    Confusion Matrix:
    [[4938    4]
    [1326  245]]
    Precision: 0.9839357429718876
    Accuracy: 0.7957930293259634
    Recall: 0.15595162316995545
    F1 Score: 0.2692307692307692

    筆記:
    True Negative (4938) 和 True Positive (245) 都相對較高
    但False Negative (1326) 的數量較高，這表示模型在捕捉實際正例方面也存在一些問題
    Recall (召回率)：0.156，表示模型僅捕捉到了實際正例的15.6%，這是一個較低的值。
    F1 Score：0.269，是Precision和Recall的平衡值，較低的F1 Score可能表明模型在Precision和Recall之間沒有很好的平衡。

    """

def model_best_params(model,param_grid,X_train, X_test, y_train, y_test):
    """找到最佳參數"""
    # 使用網格搜索交叉驗證來尋找最佳參數
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # 打印最佳參數和對應的分數
    print("Best parameters found: ", grid_search.best_params_)
    print("Best score found: ", grid_search.best_score_)

    # 使用最佳參數構建最終模型
    best_svm_model = grid_search.best_estimator_

    # 在測試集上評估最終模型
    test_score = best_svm_model.score(X_test, y_test)
    print("Test set score with best parameters: ", test_score)

def feature(df,target_column):
    """找到最重要的特徵指數"""
    from sklearn.ensemble import RandomForestRegressor
    
    X = df.drop(columns=[target_column])  # 特徵
    y = df[target_column]  # 目標

    rf_model = RandomForestRegressor()
    rf_model.fit(X, y)

    feature_importances = rf_model.feature_importances_

    important_features = X.columns[feature_importances.argsort()[::-1]]

    cumulative_importance = 0
    selected_features = []
    for index, feature in enumerate(important_features):
        cumulative_importance += feature_importances[index]
        selected_features.append(feature)
        if cumulative_importance >= 0.5:
                break
        
    print("要使用的特徵！:", selected_features)
    return selected_features

def model_score(X,y,model):
    """做5折交叉驗證, 附上績效指標"""
    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve  ,auc  
    import matplotlib.pyplot as plt
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=kf, method='predict_proba')

    fpr, tpr, thresholds = roc_curve(y, y_pred[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    y_pred_binary = (y_pred[:, 1] > 0.5).astype(int)
    conf_matrix = confusion_matrix(y, y_pred_binary)
    accuracy = accuracy_score(y, y_pred_binary)
    precision = precision_score(y, y_pred_binary)
    recall = recall_score(y, y_pred_binary)
    f1 = f1_score(y, y_pred_binary)

    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nAccuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("AUC:", roc_auc)

# 羅吉斯
model_1 = LogisticRegression(
     max_iter=1000, solver='liblinear', random_state=42
    )  

LR_params = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l2']
}

# 支援向量機
model_2 = SVC(
    probability=True, random_state=42
)  
SVC_params= {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['linear', 'rbf']
}

def smote(X,y):
    """將資料比例調整平衡後進行訓練"""
    from imblearn.over_sampling import SMOTE
    # 初始化SMOTE
    smote = SMOTE()
    # 使用SMOTE進行過取樣
    X_resampled, y_resampled = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    model_1 = LogisticRegression(
     max_iter=1000, solver='liblinear', random_state=42
    )  

    model_2 = SVC(
        probability=True, random_state=42
    )  
    print("做五折訓練");print("-"*50);print("logistic")
    model_training(X_train, X_test, y_train, y_test ,model_1)
    print("-"*50);
    model_training(X_train, X_test, y_train, y_test ,model_2)
    print("-"*50);
    
    model_score(X,y,model_1)

    model_score(X,y,model_2)
    
    

     

if __name__ == "__main__":
    df = Data().data_encoder()

    # X = df[['age','capital-gain','race','occupation','hours-per-week','native-country','education-num']]
    X = df.drop("salary",axis=1)
    y = df['salary']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    print("1.全部特徵放入, 建立模型") ; print("以下為結果");print("-"*50)

    print("logistic")
    model_training(X_train, X_test, y_train, y_test ,model_1)
    print("-"*50);print("SVM")
    model_training(X_train, X_test, y_train, y_test ,model_2)
    print("-"*50);print("END")

    

    # 第二題
    print("");print("2.利用RF之Factor important, 取重要性加總50%之重要性特徵");print("以下為結果")

    feature_column = feature(df,"salary")
    X = df[feature_column]
    
    # 最後題
    print("logistic")
    model_score(X,y,model_1)
    print("-"*50);print("SVM")
    model_score(X,y,model_2)
    print("-"*50);print("END")
    
    smote(X,y)

