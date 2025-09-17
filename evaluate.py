import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RUN_ID = "5f81dbb8a33243e98bc43c174637a114"
MODEL_URI = f"runs:/{RUN_ID}/model"

model = mlflow.sklearn.load_model(MODEL_URI)


data = pd.read_csv("data/eval_data.csv")  
FEATURE_COLUMNS = [col for col in data.columns if col != "actual"]

X = data[FEATURE_COLUMNS]
y_true = data["actual"]

y_pred = model.predict(X)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"RMSE: {rmse}, MAE: {mae}, R2: {r2}")
