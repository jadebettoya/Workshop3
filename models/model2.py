import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# We split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM model
model_svm = SVC(kernel='linear', random_state=42)
model_svm.fit(X_train, y_train)

# Evaluate the model (on the test set)
accuracy_svm = accuracy_score(y_test, model_svm.predict(X_test))

print("Accuracy SVM:", accuracy_svm)

