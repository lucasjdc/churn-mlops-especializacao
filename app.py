from fastapi import FastAPI
import mlflow
import mlflow.sklearn
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Carregar modelo treinado com MLflow
model = mlflow.sklearn.load_model("models:/log_reg_model/1")

class Data(BaseModel):
    features: list

@app.post("/predict/")
async def predict(data: Data):
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}
