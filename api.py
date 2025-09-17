from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd

RUN_ID = "5f81dbb8a33243e98bc43c174637a114"
MODEL_URI = f"runs:/{RUN_ID}/model"

# Load model at startup
model = mlflow.sklearn.load_model(MODEL_URI)

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
    return "Energy Usage Regression API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json().get("features", {})
        df = pd.DataFrame([data], columns=FEATURE_COLUMNS)
        prediction = model.predict(df)[0]
        return jsonify({"prediction_kWh": float(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
