import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logging.info("Lendo dados brutos…")
df = pd.read_csv('data/raw/telecom_churn.csv')

logging.info("Removendo coluna alvo 'Churn'")
df.drop(columns=['Churn']) \
  .to_csv('data/processed/telecom_churn_X.csv', index=False)

logging.info("Arquivo de features salvo em data/processed/telecom_churn_X.csv")
