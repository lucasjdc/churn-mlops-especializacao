import mlflow
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logged_model = '/home/lucas/Especializacao/MLOps/atividade_2/mlartifacts/766149959934061426/4b298027518b41e083ad5ddfba630928/artifacts/log_reg_model'

# Carregar o modelo registrado como um PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Carregar os dados para previsão.
data = pd.read_csv('data/processed/telecom_churn_X.csv')

categorical_columns = data.select_dtypes(include=['object']).columns

# Instanciar o LabelEncoder
label_encoder = LabelEncoder()

# Transformar as variáveis categóricas para valores numéricos
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Realizar a previsão no conjunto de dados.
predicted = loaded_model.predict(data)

# Ajuste de limiar
threshold = 0.6  
data['predicted'] = predicted > threshold  

# Salvar as previsões em um novo arquivo CSV.
data.to_csv('data/predictions/churn.csv', index=False)

print("Predições salvas em 'data/predictions/churn.csv'")