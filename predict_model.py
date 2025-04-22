import mlflow
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Configurar MLflow
mlflow.set_tracking_uri("http://localhost:5000")

# Carregar o run_id salvo
with open("latest_run_id.txt", "r") as f:
    run_id = f.read().strip()

# Definir URI do modelo
model_uri = f"runs:/{run_id}/model"

# Carregar modelo
model = mlflow.pyfunc.load_model(model_uri)

# Carregar dados para predição
data = pd.read_csv("data/processed/telecom_churn_X.csv")
categorical_cols = data.select_dtypes(include=["object"]).columns

le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Fazer predição
preds = model.predict(data)

# Aplicar limiar
data['predicted'] = preds > 0.6

# Salvar resultados
data.to_csv("data/predictions/churn.csv", index=False)
print("Predições salvas em 'data/predictions/churn.csv'")

