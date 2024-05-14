# Begging 模型績效


from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score , roc_auc_score

from data import Data

def output_score(y_test,y_pred):
    """輸出模型績效指數"""
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

    auc = roc_auc_score(y_test, y_pred)
    print("AUC:", auc)

def model_training(X_train, X_test, y_train, y_test, model):
    """簡單的模型訓練"""
    model = model

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    output_score(y_test,y_pred)

def Model_training_individual(X_train, X_test, y_train, y_test):

    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier

    # 輸出
    print("1:LR");print("-" * 50)

    model = LogisticRegression()    
    model_training(X_train, X_test, y_train, y_test,model)

    print("-" * 50); print("2:DT");print("-" * 50)

    model = DecisionTreeClassifier()
    model_training(X_train, X_test, y_train, y_test,model)

    print("-" * 50);  print("3:SVM");print("-" * 50)

    model = SVC()
    model_training(X_train, X_test, y_train, y_test,model) 

    print("-" * 50); print("4:KNN");print("-" * 50)

    model = KNeighborsClassifier()
    model_training(X_train, X_test, y_train, y_test,model)
    
    print("-"*20,end="");print("第一題END",end="");print("-"*20)

def Bagging(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    base_classifier = DecisionTreeClassifier()

    bagging_classifier = BaggingClassifier(
        base_estimator=base_classifier,
        n_estimators=10,
        random_state=42
        )

    bagging_classifier.fit(X_train, y_train)

    y_pred = bagging_classifier.predict(X_test)

    print("begging");print("-" * 50)

    output_score(y_test,y_pred)


if __name__ == "__main__":
    data = Data().data_encoder()
    # 依照RF的重要性
    Ｘ = data[["age","workclass","education","occupation"]]
    y = data[['salary']]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    Model_training_individual(X_train, X_test, y_train, y_test)

    Bagging(X_train, X_test, y_train, y_test)
