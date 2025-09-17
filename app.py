import gradio as gr
import mlflow
import mlflow.sklearn
import numpy as np

model = None

def load_latest_model(experiment_name="Energy_Usage_Regression", artifact_name="model"):
    global model
    if model is not None:
        return model

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise Exception(f"Experiment '{experiment_name}' not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.end_time DESC"],
        filter_string="attributes.status='FINISHED'"
    )
    if len(runs) == 0:
        raise Exception("No finished runs found")

    latest_run = runs[0]
    run_id = latest_run.info.run_id
    print(f"Loading best model from run: {run_id}")

    model_uri = f"runs:/{run_id}/{artifact_name}"
    model = mlflow.sklearn.load_model(model_uri)
    return model


week_options = ["Weekend", "Weekday"]
day_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
load_type_options = ["Maximum Load", "Medium Load", "Light Load"]


def predict_usage(
    Leading_Current_Reactive_Power_kVarh,
    Lagging_Current_Power_Factor,
    Leading_Current_Power_Factor,
    NSM,
    WeekStatus,
    Day_of_week,
    Load_Type
):
   
    features = []

    
    features.append(float(Leading_Current_Reactive_Power_kVarh))
    features.append(float(Lagging_Current_Power_Factor))
    features.append(float(Leading_Current_Power_Factor))
    features.append(float(NSM))

    features.append(1 if WeekStatus == "Weekend" else 0)  

  
    day_list = ["Monday", "Saturday", "Sunday", "Thursday", "Tuesday", "Wednesday"]
    for day in day_list:
        features.append(1 if Day_of_week == day else 0)

    load_list = ["Maximum Load", "Medium Load"]
    for lt in load_list:
        features.append(1 if Load_Type == lt else 0)

    
    features = np.array([features])

    prediction = load_latest_model().predict(features)[0]
    return round(prediction, 2)

iface = gr.Interface(
    fn=predict_usage,
    inputs=[
        gr.Number(label="Leading Current Reactive Power (kVarh)", value=50, info="Example range: 10-100"),
        gr.Number(label="Lagging Current Power Factor", value=0.95, info="Typical range: 0.85-1.0"),
        gr.Number(label="Leading Current Power Factor", value=0.95, info="Typical range: 0.85-1.0"),
        gr.Number(label="NSM (Seconds from midnight)", value=0, info="Range: 0-86400"),
        gr.Dropdown(week_options, label="Week Status"),
        gr.Dropdown(day_options, label="Day of Week"),
        gr.Dropdown(load_type_options, label="Load Type")
    ],
    outputs=gr.Textbox(label="Predicted Energy Usage (kWh)"),
    title="Energy Usage Prediction",
    description="Predict energy consumption using your Gradient Boosting MLflow model."
)

if __name__ == "__main__":
    iface.launch()
