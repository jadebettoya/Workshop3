from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd
iris = datasets.load_iris()
x = pd.DataFrame(data=iris.data,columns=iris.feature_names)
y = pd.DataFrame(iris.target)

print (x, y)

from sklearn.metrics import accuracy_score

# Ici je divise les données en ensembles d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Je crée un modèle RandomForest
model = RandomForestClassifier(random_state=42)

# ensuite on entraine ce modèle sur l'ensemble d'entraînement
model.fit(x_train, y_train.values.ravel())


predictions = model.predict(x_test)

# Évaluez la précision du modèle
accuracy = accuracy_score(y_test, predictions)
print(f"Précision du modèle : {accuracy * 100:.2f}%")


from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

# Matrice de confusion
conf_matrix = confusion_matrix(y_test, predictions)
print("Matrice de confusion :\n", conf_matrix)

# Rapport de classification
class_report = classification_report(y_test, predictions)
print("Rapport de classification :\n", class_report)

# Validation croisée
cross_val_scores = cross_val_score(model, x, y.values.ravel(), cv=5)
print("Précision moyenne par validation croisée :", cross_val_scores.mean())
