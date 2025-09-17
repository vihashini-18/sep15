from flask import Flask, request, jsonify
import mlflow
import numpy as np
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


RUN_ID = "5f81dbb8a33243e98bc43c174637a114"
ARTIFACT_NAME = "model"  
MODEL_URI = f"runs:/{RUN_ID}/{ARTIFACT_NAME}"

print(f"Loading model from: {MODEL_URI} ...")
model = mlflow.sklearn.load_model(MODEL_URI)
print("Model loaded successfully!")


app = Flask(__name__)

FEATURE_COLUMNS = [
    'Leading_Current_Reactive_Power_kVarh',
    'Lagging_Current_Power_Factor',
    'Leading_Current_Power_Factor',
    'NSM',
    'WeekStatus_Weekend',
    'Day_of_week_Monday',
    'Day_of_week_Saturday',
    'Day_of_week_Sunday',
    'Day_of_week_Thursday',
    'Day_of_week_Tuesday',
    'Day_of_week_Wednesday',
    'Load_Type_Maximum_Load',
    'Load_Type_Medium_Load'
]

@app.route("/")
def home():
    return "MLflow Energy Usage Regression API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_json = request.get_json()
        features_dict = input_json.get("features", {})

        df = pd.DataFrame([features_dict], columns=FEATURE_COLUMNS)

      
        prediction = model.predict(df)[0]

        return jsonify({
            "prediction_kWh": float(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/metrics", methods=["POST"])
def metrics():
    """
    Send a JSON of actual vs predicted to get evaluation metrics:
    {
        "data": [
            {"features": {...}, "actual": 123},
            {"features": {...}, "actual": 150}
        ]
    }
    """
    try:
        input_json = request.get_json()
        data_list = input_json.get("data", [])

        if not data_list:
            return jsonify({"error": "No data provided"}), 400

        df_features = pd.DataFrame([d['features'] for d in data_list], columns=FEATURE_COLUMNS)
        y_true = pd.Series([d['actual'] for d in data_list])

        y_pred = model.predict(df_features)

        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return jsonify({
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
