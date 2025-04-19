import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("churn_prediction")

# Dados
df = pd.read_csv("data/raw/telecom_churn.csv")
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Função para treinar e logar
def treinar_modelo(nome_modelo, modelo, log_model_func):
    with mlflow.start_run(run_name=nome_modelo) as run:
        logger.info(f"Treinando modelo {nome_modelo}")
        modelo.fit(X_train, y_train)
        preds = modelo.predict(X_test)
        acc = accuracy_score(y_test, preds)
        logger.info(f"Acurácia ({nome_modelo}): {acc:.4f}")

        mlflow.log_param("modelo", nome_modelo)
        mlflow.log_metric("accuracy", acc)
        log_model_func(modelo, artifact_path="model")

        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusão - {nome_modelo}')
        plt.xlabel('Previsto')
        plt.ylabel('Real')
        plt.savefig(f"confusion_matrix_{nome_modelo}.png")
        mlflow.log_artifact(f"confusion_matrix_{nome_modelo}.png")
        os.remove(f"confusion_matrix_{nome_modelo}.png")

        logger.info(f"Run ID ({nome_modelo}): {run.info.run_id}")
        with open(f"latest_run_id_{nome_modelo}.txt", "w") as f:
            f.write(run.info.run_id)

# Árvore de Decisão
treinar_modelo("DecisionTree", DecisionTreeClassifier(random_state=42), mlflow.sklearn.log_model)

# XGBoost
treinar_modelo("XGBoost", XGBClassifier(eval_metric="logloss", random_state=42), mlflow.xgboost.log_model)

print("✅ Execuções finalizadas com sucesso.")

"""
Comando para iniciar o mlflow

mlflow server \
  --backend-store-uri sqlite:///mlflow_db/mlflow.db \
  --default-artifact-root file:///home/lucas/Especializacao/MLOps/atividade_2/mlruns \
  --host 0.0.0.0 \
  --port 5000    
"""