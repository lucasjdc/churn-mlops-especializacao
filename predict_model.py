import mlflow

logged_model = 'runs:/mlflow-artifacts:/766149959934061426/4b298027518b41e083ad5ddfba630928/artifacts/log_reg_model'
#logged_model = 'file:////home/lucas/Especializacao/MLOps/atividade_2/mlartifacts/766149959934061426/5946b730911245adb6e7f8e812c3d169/artifacts/log_reg_model'
#logged_model = 'runs:/4b298027518b41e083ad5ddfba630928/log_reg_model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
data = pd.read_csv('data/processed/telecom_churn_X.csv')
predicted = loaded_model.predict(data)

data['predicted'] = predicted

data.to_csv('data/predictions/churn.csv')

