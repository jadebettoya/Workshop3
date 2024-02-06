from flask import Flask, request, jsonify

app = Flask(__name__)

# Route pour effectuer les prédictions
@app.route('/predict', methods=['GET'])
def predict():
    # Récupérer les arguments de la requête GET
    args = request.args

    required_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    if not all(feature in args for feature in required_features):
        return jsonify({"error": f"Provide all features: {', '.join(required_features)}"})

    # on prepare les données pour la prédiction
    features = [float(args[feature]) for feature in required_features]
    input_data = [features]

    # Model de prediction appliqué
    prediction = model.predict(input_data)

    # on retourne mtn la prédiction au format JSON
    return jsonify({"prediction": int(prediction[0])})

# Route pour évaluer la précision du modèle
@app.route('/evaluate', methods=['GET'])
def evaluate():
    # Effectuer l'évaluation du modèle
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions) * 100

    # Retourner la précision au format JSON
    return jsonify({"accuracy": accuracy})

# Point d'entrée de l'application Flask
if __name__ == '__main__':
    app.run(debug=True)
