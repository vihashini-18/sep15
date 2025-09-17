python
import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "features": {
        "Leading_Current_Reactive_Power_kVarh": 50,
        "Lagging_Current_Power_Factor": 0.95,
        "Leading_Current_Power_Factor": 0.95,
        "NSM": 3600,
        "WeekStatus_Weekend": 1,
        "Day_of_week_Monday": 0,
        "Day_of_week_Saturday": 0,
        "Day_of_week_Sunday": 0,
        "Day_of_week_Thursday": 0,
        "Day_of_week_Tuesday": 0,
        "Day_of_week_Wednesday": 1,
        "Load_Type_Maximum_Load": 0,
        "Load_Type_Medium_Load": 1
    }
}

response = requests.post(url, json=data)
print(response.json())
