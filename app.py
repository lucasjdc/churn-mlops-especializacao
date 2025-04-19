from fastapi import FastAPI
import mlflow
import numpy as np
from pydantic import BaseModel
import logging

# Configuração básica de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

app = FastAPI()

# Log ao iniciar o app
logging.info("Iniciando a aplicação e carregando o modelo...")

# Caminho do modelo
model_path = "/home/lucas/Especializacao/MLOps/atividade_2/mlartifacts/766149959934061426/4b298027518b41e083ad5ddfba630928/artifacts/log_reg_model"

try:
    model = mlflow.pyfunc.load_model(model_path)
    logging.info("Modelo carregado com sucesso.")
except Exception as e:
    logging.error(f"Erro ao carregar o modelo: {e}")
    raise

class Data(BaseModel):
    features: list

@app.post("/predict/")
async def predict(data: Data):
    logging.info(f"Recebida requisição de predição com dados: {data.features}")
    try:
        features = np.array(data.features).reshape(1, -1)
        prediction = model.predict(features)
        result = int(prediction[0])
        logging.info(f"Predição gerada: {result}")
        return {"prediction": result}
    except Exception as e:
        logging.error(f"Erro durante predição: {e}")
        return {"error": str(e)}
