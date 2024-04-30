"""決策樹作業"""
from sklearn.tree import DecisionTreeClassifier, export_text
from ..data import Data
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
print("決策樹作業!")

data = Data().data_encoder()
X = data[['education-num', 'occupation']]
y = data[['salary']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

dt_classifier = DecisionTreeClassifier()

dt_classifier.fit(X_train, y_train)


tree_rules = export_text(dt_classifier, feature_names=list(X.columns))
print("決策樹規則：\n", tree_rules)


y_pred = dt_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("模型準確度：", accuracy)
