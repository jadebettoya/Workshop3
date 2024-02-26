# app.py
from flask import Flask, request, render_template
from models.model1 import model1
from models.model2 import model2
from models.model3 import model3
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load Iris dataset
iris = datasets.load_iris()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y = pd.DataFrame(iris.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get predictions from each model
    y_pred1 = model1.predict(X_test)
    y_pred2 = model2.predict(X_test)
    y_pred3 = model3.predict(X_test)

    # Calculate accuracy for each model
    accuracy1 = accuracy_score(y_test, y_pred1)
    accuracy2 = accuracy_score(y_test, y_pred2)
    accuracy3 = accuracy_score(y_test, y_pred3)

    # Calculate consensus (you can implement your own logic here)
    consensus = (accuracy1 + accuracy2 + accuracy3) / 3

    return render_template('result.html', accuracy1=accuracy1, accuracy2=accuracy2, accuracy3=accuracy3, consensus=consensus)

if __name__ == '__main__':
    app.run(debug=True)
