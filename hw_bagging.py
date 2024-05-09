# Begging 模型績效


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


def Model_training_individual(X_train, X_test, y_train, y_test):

    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    
    print("1:LR")
    model = LogisticRegression()    
    model_training(model,X_train, X_test, y_train, y_test)
    print("2:DT")
    model = DecisionTreeClassifier()
    model_training(model,X_train, X_test, y_train, y_test)
    print("3:SVM")
    model = SVC()
    model_training(model,X_train, X_test, y_train, y_test)
    print("4:KNN")
    model = KNeighborsClassifier()
    model_training(model,X_train, X_test, y_train, y_test)

def Bagging(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    # 初始化基本分類器
    base_classifier = DecisionTreeClassifier()

    # 初始化Bagging分類器
    bagging_classifier = BaggingClassifier(
        base_estimator=base_classifier,
        n_estimators=10,
        random_state=42
        )

    # 訓練Bagging分類器
    bagging_classifier.fit(X_train, y_train)

    # 預測測試集
    y_pred = bagging_classifier.predict(X_test)

    conf_matrix = confusion_matrix(y_test, y_pred)

    print("混淆矩陣：")
    print(conf_matrix)

    accuracy = accuracy_score(y_test, y_pred)
    print("準確性：", accuracy)

    precision = precision_score(y_test, y_pred, average='weighted')
    print("精確率：", precision)

    recall = recall_score(y_test, y_pred, average='weighted')
    print("召回率：", recall)

    f1 = f1_score(y_test, y_pred, average='weighted')
    print("F1分數：", f1)
        


if __name__ == "__main__":
    df = Data().data_encoder()

    X = df.drop("salary",axis=1)
    y = df['salary']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    Model_training_individual(X_train, X_test, y_train, y_test)

    Bagging(X_train, X_test, y_train, y_test)
