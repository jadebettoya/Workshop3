from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd

# Charger le jeu de données Iris
iris = datasets.load_iris()
x = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y = pd.DataFrame(iris.target)

# Diviser les données en ensembles d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialiser et entraîner le modèle RandomForest
model1 = RandomForestClassifier(random_state=42)
model1.fit(x_train, y_train.values.ravel())
