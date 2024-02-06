# A MODIFIER AVEC NGROK
import requests
import numpy as np
from sklearn.metrics import accuracy_score

svm_url = "http://123.ngrok.io/predict"  

def get_predictions(input_data):
    svm_response = requests.get(svm_url, json={"data": input_data.tolist()}).json()
    return svm_response['prediction']

# Générer des données de test (à remplacer par vos propres données)
test_data = np.array([[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [6.9, 3.1, 5.4, 2.1]])

predictions = []
for data_point in test_data:
    model_prediction = get_predictions(data_point)
    predictions.append(model_prediction)

consensus_prediction = np.mean(predictions, axis=0)

print("Consensus Prediction:", consensus_prediction)

true_labels = np.array([0, 0, 1])  

accuracy_consensus = accuracy_score(true_labels, consensus_prediction)
print("Accuracy Consensus Prediction:", accuracy_consensus)
