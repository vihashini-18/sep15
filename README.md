# 🏭 Steel Industry Prediction Project

This project is a **Machine Learning pipeline** for predicting steel industry data and includes a **Gradio web app**, APIs, evaluation scripts, and Docker support.

---

## Project Structure

- `Steel_industry_data.csv` – Dataset for training and evaluation  
- `code.ipynb` – Jupyter Notebook for data exploration, preprocessing, and model training  
- `app.py` – Gradio-based interactive web application for predictions  
- `api.py` – Flask/FastAPI (or custom) API for serving the model programmatically  
- `evaluate.py` – Script to evaluate model performance (MAE, MSE, R2, RMSE)  
- `flow.py` – Data pipeline / workflow script (optional)  
- `requirements.txt` – Project dependencies  
- `docker/` – Docker configuration files to containerize the app  
- `mlruns/` – MLflow experiment tracking folder  
- `.gradio/flagged` – Internal Gradio folder (user interface management)  
- `test.py` – Test script to validate model or API functionality  

---

## Features

- **Machine Learning model** trained on steel industry dataset  
- **Interactive Gradio app** for easy predictions  
- **API endpoint** for programmatic predictions  
- **Evaluation scripts** to measure model performance  
- **Docker support** for containerized deployment  
- ML experiment tracking with **MLflow**  

---

## Requirements

- Python 3.8+  
- Gradio  
- TensorFlow / PyTorch (depending on model)  
- scikit-learn  
- pandas, numpy  
- MLflow  

Install dependencies via:

```bash
pip install -r requirements.txt
