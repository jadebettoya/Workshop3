from flask import Blueprint, request, jsonify
from models.model1 import model1
from models.model2 import train_model
#etc...
import pandas as pd

api = Blueprint('api', __name__)

@api.route('/predict', methods=['GET'])
def predict():

    parameters = request.args.to_dict()

    prediction1 = model1.predict(pd.DataFrame(parameters, index=[0]))
    prediction2 = train_model.predict(pd.DataFrame(parameters, index=[0]))
    # Prediction for Model 3
    # ...


    consensus_prediction = (prediction1 + prediction2) / 2


    response = {
        'consensus_prediction': consensus_prediction.tolist(),
        'individual_predictions': {
            'model1': prediction1.tolist(),
            'model2': prediction2.tolist(),
            # ...
        }
    }

    return jsonify(response)
