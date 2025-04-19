#!/bin/bash

echo "Iniciando pipeline..."

# 1. Treinar modelo
python model_training.py

# 2. Verificar data drift
python data_drift_monitoring.py

# 3. Rodar predições com modelo treinado
python predict_model.py

# 4. Subir a API (se desejar automático com uvicorn)
# uvicorn app:app --reload

echo "Pipeline finalizado."
