import logging
import pandas as pd
import numpy as np
import mlflow
import os
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Configurar o logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configurar MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("churn_prediction")

# Função para logar matriz de confusão
def log_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusão - {model_name}')
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    
    # Salvar a imagem temporariamente
    file_name = f"{model_name}_confusion_matrix.png"
    plt.savefig(file_name)
    plt.close()

    # Fazer o log do arquivo como artefato no MLflow
    mlflow.log_artifact(file_name)

    # (Opcional) Remover o arquivo local após o log
    os.remove(file_name)

# Carregar os dados
logger.info("Carregando dados do arquivo CSV.")
df = pd.read_csv("data/raw/telecom_churn.csv")

# Converter colunas categóricas para numéricas
logger.info("Convertendo colunas categóricas para numéricas.")
df_encoded = df.copy()
categorical_cols = df_encoded.select_dtypes(include=["object"]).columns
logger.debug(f"Colunas categóricas detectadas: {list(categorical_cols)}")

le = LabelEncoder()
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])

# Separar features (X) e rótulo (y)
X = df_encoded.drop(columns=["Churn"])
y = df_encoded["Churn"]
logger.info("Dados preparados para o modelo.")

# Inicializar o k-fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Armazenar resultados e melhores modelos
best_xgb_model = None
best_log_reg_model = None
best_xgb_acc = 0
best_log_reg_acc = 0

xgb_accuracies = []  # Inicializando a lista para armazenar as acurácias
log_reg_accuracies = []  # Inicializando a lista para armazenar as acurácias da regressão logística

# Loop de validação cruzada
for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
    logger.info(f"Iniciando fold {fold+1}")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Normalizar os dados para a regressão logística
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with mlflow.start_run(run_name=f"fold_{fold+1}"):
        # Modelo XGBoost
        logger.debug("Treinando modelo XGBoost.")
        xgb = XGBClassifier(eval_metric="logloss", random_state=42)
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        acc_xgb = accuracy_score(y_test, y_pred_xgb)
        
        # Verificar se o modelo XGBoost é o melhor
        if acc_xgb > best_xgb_acc:
            best_xgb_acc = acc_xgb
            best_xgb_model = xgb
        
        logger.debug(f"Acurácia XGBoost (fold {fold+1}): {acc_xgb:.4f}")
        mlflow.log_metric("Acurácia: ", acc_xgb)

        report_xgb = classification_report(y_test, y_pred_xgb, output_dict=True)
        mlflow.log_metric("xgb_accuracy", acc_xgb)
        mlflow.log_dict(report_xgb, "xgb_classification_report.json")
        mlflow.xgboost.log_model(xgb, "xgb_model")
        log_confusion_matrix(y_test, y_pred_xgb, "XGBoost")

        xgb_accuracies.append(acc_xgb)  # Corrigido para usar a variável correta

        # Modelo Regressão Logística
        logger.debug("Treinando modelo Regressão Logística.")
        log_reg = LogisticRegression(max_iter=2000, class_weight="balanced", solver="saga")
        log_reg.fit(X_train_scaled, y_train)
        y_pred_log_reg = log_reg.predict(X_test_scaled)
        acc_log_reg = accuracy_score(y_test, y_pred_log_reg)
        
        # Verificar se o modelo Regressão Logística é o melhor
        if acc_log_reg > best_log_reg_acc:
            best_log_reg_acc = acc_log_reg
            best_log_reg_model = log_reg
        
        logger.debug(f"Acurácia Regressão Logística (fold {fold+1}): {acc_log_reg:.4f}")

        report_log_reg = classification_report(y_test, y_pred_log_reg, output_dict=True)
        mlflow.log_metric("log_reg_accuracy", acc_log_reg)
        mlflow.log_dict(report_log_reg, "log_reg_classification_report.json")
        mlflow.sklearn.log_model(log_reg, "log_reg_model")
        log_confusion_matrix(y_test, y_pred_log_reg, "Logistic Regression")

        log_reg_accuracies.append(acc_log_reg)

        print(f"XGBoost Acc: {acc_xgb:.4f}")
        print(f"Regressão Logística Acc: {acc_log_reg:.4f}")
        print("\nMatriz de Confusão - XGBoost:")
        print(confusion_matrix(y_test, y_pred_xgb))
        print("\nMatriz de Confusão - Regressão Logística:")
        print(confusion_matrix(y_test, y_pred_log_reg))

# Registrar o melhor modelo XGBoost no MLflow
if best_xgb_model is not None:
    logger.info(f"Registrando o melhor modelo XGBoost com acurácia {best_xgb_acc:.4f}.")
    with mlflow.start_run(run_name="best_xgb_model"):
        mlflow.xgboost.log_model(best_xgb_model, "best_xgb_model")
        logger.info("Modelo XGBoost registrado no MLflow.")

# Registrar o melhor modelo Regressão Logística no MLflow
if best_log_reg_model is not None:
    logger.info(f"Registrando o melhor modelo Regressão Logística com acurácia {best_log_reg_acc:.4f}.")
    with mlflow.start_run(run_name="best_log_reg_model"):
        mlflow.sklearn.log_model(best_log_reg_model, "best_log_reg_model")
        logger.info("Modelo Regressão Logística registrado no MLflow.")

# Resultados finais
logger.info("Validação cruzada finalizada.")
logger.info(f"Média de acurácia XGBoost: {np.mean(xgb_accuracies):.4f}")
logger.info(f"Média de acurácia Regressão Logística: {np.mean(log_reg_accuracies):.4f}")

print(f"\nXGBoost Média de Acurácia: {np.mean(xgb_accuracies):.4f}")
print(f"Regressão Logística Média de Acurácia: {np.mean(log_reg_accuracies):.4f}")

logger.info(f"Média de acurácia XGBoost: {np.mean(xgb_accuracies):.4f}")