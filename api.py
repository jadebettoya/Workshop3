from flask import Blueprint, request, jsonify
from sklearn.metrics import accuracy_score

api = Blueprint('api', __name__)

@api.route('/predict', methods=['GET'])
def predict():
    # Récupére les arguments de la requête GET
    args = request.args

    required_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    if not all(feature in args for feature in required_features):
        return jsonify({"error": f"Provide all features: {', '.join(required_features)}"})

    # Prépare les données pour la prédiction
    features = [float(args[feature]) for feature in required_features]
    input_data = [features]

    # Effectue la prédiction avec le modèle
    prediction = model1.predict(input_data)

    # Retourne la prédiction au format JSON
    return jsonify({"prediction": int(prediction[0])})

@api.route('/evaluate', methods=['GET'])
def evaluate():
    # Effectue l'évaluation du modèle
    predictions = model1.predict(x_test)
    accuracy = accuracy_score(y_test, predictions) * 100

    # Retourne la précision au format JSON
    return jsonify({"accuracy": accuracy})
