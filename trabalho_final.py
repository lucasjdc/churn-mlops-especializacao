import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configurar MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("churn_prediction")

# Carregar os dados
df = pd.read_csv("data/raw/telecom_churn.csv")

# Converter colunas categóricas para numéricas
df_encoded = df.copy()
categorical_cols = df_encoded.select_dtypes(include=["object"]).columns

le = LabelEncoder()
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])

# Separar features (X) e rótulo (y)
X = df_encoded.drop(columns=["Churn"])  # Ajuste conforme o nome da coluna-alvo
y = df_encoded["Churn"]

# Inicializar o k-fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Armazenar resultados
xgb_accuracies = []
log_reg_accuracies = []

# Loop de validação cruzada
for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Normalizar os dados para a regressão logística
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with mlflow.start_run(run_name=f"fold_{fold+1}"):
        # Modelo XGBoost
        xgb = XGBClassifier(eval_metric="logloss", random_state=42)
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        acc_xgb = accuracy_score(y_test, y_pred_xgb)
        xgb_accuracies.append(acc_xgb)

        # Relatório de classificação e matriz de confusão
        report_xgb = classification_report(y_test, y_pred_xgb, output_dict=True)
        conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)

        mlflow.log_metric("xgb_accuracy", acc_xgb)
        mlflow.log_dict(report_xgb, "xgb_classification_report.json")
        mlflow.xgboost.log_model(xgb, "xgb_model")

        # Modelo Regressão Logística
        log_reg = LogisticRegression(max_iter=2000, class_weight="balanced", solver="saga")
        log_reg.fit(X_train_scaled, y_train)
        y_pred_log_reg = log_reg.predict(X_test_scaled)
        acc_log_reg = accuracy_score(y_test, y_pred_log_reg)
        log_reg_accuracies.append(acc_log_reg)

        # Relatório de classificação e matriz de confusão
        report_log_reg = classification_report(y_test, y_pred_log_reg, output_dict=True)
        conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)

        mlflow.log_metric("log_reg_accuracy", acc_log_reg)
        mlflow.log_dict(report_log_reg, "log_reg_classification_report.json")
        mlflow.sklearn.log_model(log_reg, "log_reg_model")

        print(f"\n🔹 Fold {fold+1} concluído!")
        print(f"📌 XGBoost Acc: {acc_xgb:.4f}")
        print(f"📌 Regressão Logística Acc: {acc_log_reg:.4f}")
        print("\nMatriz de Confusão - XGBoost:")
        print(conf_matrix_xgb)
        print("\nMatriz de Confusão - Regressão Logística:")
        print(conf_matrix_log_reg)

# Resultados finais
print(f"\n🏆 XGBoost Média de Acurácia: {np.mean(xgb_accuracies):.4f}")
print(f"🏆 Regressão Logística Média de Acurácia: {np.mean(log_reg_accuracies):.4f}")
