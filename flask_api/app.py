from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load('models/flight_price_model.pkl')


@app.route('/')
def home():
    return jsonify({
        "message": "Flight Price Prediction API Running"
    })


@app.route('/predict', methods=['POST'])
def predict():

    try:

        data = request.get_json()

        input_df = pd.DataFrame([data])

        prediction = model.predict(input_df)[0]

        return jsonify({
            "predicted_price": round(float(prediction), 2)
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5001)