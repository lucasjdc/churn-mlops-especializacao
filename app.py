from fastapi import FastAPI
import mlflow
import numpy as np
from pydantic import BaseModel
import logging
from mlflow.tracking import MlflowClient

# Configuração de log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

app = FastAPI()

# Log de inicialização
logging.info("Iniciando a aplicação e carregando o modelo...")

# Obter o modelo mais recente automaticamente
try:
    client = MlflowClient()
    latest_model = client.get_latest_versions("log_reg_model", stages=["None", "Staging", "Production"])[-1]
    model_uri = f"runs:/{latest_model.run_id}/log_reg_model"
    model = mlflow.pyfunc.load_model(model_uri)
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

