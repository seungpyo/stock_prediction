from google.cloud import bigquery as bq
from google.oauth2 import service_account
import pandas as pd
import os
credentials = service_account.Credentials.from_service_account_file(
    filename=f"{os.getenv('HOME')}/.key/minerva-375407-cc4ad2c89cfe.json"
)
bqc = bq.Client(project='minerva-375407', credentials=credentials)

df = bqc.query('SELECT * FROM `minerva-375407.dw.test` LIMIT 10').result().to_dataframe()
print(df)