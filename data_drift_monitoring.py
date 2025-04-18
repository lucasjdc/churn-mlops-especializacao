import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Carregar os dados
df = pd.read_csv("data/raw/telecom_churn.csv")

# Dividir os dados
split_index = int(0.8 * len(df))
df_ref = df[:split_index]
df_cur = df[split_index:]

# Criar e rodar o relatório
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=df_ref, current_data=df_cur)

# Salvar em HTML
report.save_html("drift_report.html")