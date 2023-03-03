import os
import yfinance as yf
from google.cloud import bigquery as bq
from google.oauth2 import service_account
import FinanceDataReader as fdr
import logging
import multiprocessing as mp
from tqdm import tqdm
from typing import Optional
from functools import partial
import pandas as pd
from dask import dataframe as dd
from enum import Enum
class ColumnName(str, Enum):
    INFO_AT = "info_at"
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"
    DIVIDENDS = "dividends"
    STOCK_SPLITS = "stock_splits"
    TICKER = "ticker"
    
def make_default_schema() -> bq.SchemaField:
    return [
        bq.SchemaField(ColumnName.INFO_AT.value, bq.enums.StandardSqlTypeNames.DATETIME),
        bq.SchemaField(ColumnName.OPEN.value, bq.enums.StandardSqlTypeNames.FLOAT64),
        bq.SchemaField(ColumnName.HIGH.value, bq.enums.StandardSqlTypeNames.FLOAT64),
        bq.SchemaField(ColumnName.LOW.value, bq.enums.StandardSqlTypeNames.FLOAT64),
        bq.SchemaField(ColumnName.CLOSE.value, bq.enums.StandardSqlTypeNames.FLOAT64),
        bq.SchemaField(ColumnName.VOLUME.value, bq.enums.StandardSqlTypeNames.INT64),
        bq.SchemaField(ColumnName.DIVIDENDS.value, bq.enums.StandardSqlTypeNames.FLOAT64),
        bq.SchemaField(ColumnName.STOCK_SPLITS.value, bq.enums.StandardSqlTypeNames.FLOAT64),
        bq.SchemaField(ColumnName.TICKER.value, bq.enums.StandardSqlTypeNames.STRING),
    ]
      
def merge_parquet(parquet_dir: str) -> str:
    parquets = [os.path.join(parquet_dir, p) for p in os.listdir(parquet_dir) if p.endswith(".parquet")]
    dd.read_parquet(
        path=parquet_dir,
        engine="pyarrow"
    ).repartition(
        npartitions=1
    ).to_parquet(
        parquet_dir, 
        name_function=lambda i: "all.parquet"
    )
    for p in parquets:
        os.remove(p)
    return os.path.join(parquet_dir, "all.parquet")

def upload_parquet(paquet_path: str, project_id: str) -> None:
    cfg = bq.LoadJobConfig(
        source_format=bq.SourceFormat.PARQUET,
        schema=make_default_schema(),
        write_disposition=bq.WriteDisposition.WRITE_APPEND,
    )
    credentials = service_account.Credentials.from_service_account_file(
        filename=f"{os.getenv('HOME')}/.key/minerva-375407-cc4ad2c89cfe.json"
    )
    bqc = bq.Client(project=project_id, credentials=credentials)
    with open(paquet_path, 'rb') as f:
        job = bqc.load_table_from_file(
            f, 
            f'{project_id}.dw.raw_prices',
            job_config=cfg,
        )
        job.result()
    os.remove(paquet_path)
    
def load_from_ticker(tmp_dir: str, code: str) -> None:
    ticker = yf.Ticker(f"{code}.KS")
    try:
        df = ticker.history(period='10y', interval='1d', raise_errors=True)
    except Exception as e:
        return 
    df = df.reset_index().rename(columns={"Date": "info_at"}) 
    logger.debug("Before rename")
    logger.debug(df.columns)
    df.rename(columns={k: k.lower().replace(' ', '_') for k in df.columns}, inplace=True)
    for c in ColumnName.__members__.values():
        if c.value not in df.columns:
            df[c.value] = None
    logger.debug("After rename")
    logger.debug(df.columns)
    df[ColumnName.TICKER.value] = code
    df.to_parquet(f"{tmp_dir}/{code}.parquet")
   
def main(project_id: str, market: str, limit_rows: Optional[int], num_proc: int, tmp_dir: str) -> None:
    os.makedirs(tmp_dir, exist_ok=True)
    logger.debug(f"Saving temporary parquets at {tmp_dir}")
    all_stocks = fdr.StockListing(market)
    codes = all_stocks['Code'].to_list()
    if limit_rows is not None:
        assert limit_rows < len(codes), f"limit_rows({limit_rows}) must be less than len(codes)({len(codes)})"
        codes = codes[:limit_rows]
    logger.debug(f"Found {len(codes)} codes from {market}")
    logger.debug(f"Using {num_proc} processes")
    with mp.Pool(processes=num_proc) as pool:
        with tqdm(total=len(codes)) as pbar:
            for _ in tqdm(pool.imap_unordered(partial(load_from_ticker, tmp_dir), codes)):
                pbar.update()
    logger.debug("Merging parquets")
    parquet_path = merge_parquet(tmp_dir)
    logger.debug(f"Uploading parquet to {project_id}.dw.raw_prices")
    upload_parquet(parquet_path, project_id)
    logger.info("DONE!")
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', type=str, default='minerva-375407')
    parser.add_argument('--market', type=str, default='KRX')
    parser.add_argument('--limit_rows', type=int, default=None)
    parser.add_argument('--num_proc', type=int, default=mp.cpu_count())
    parser.add_argument('--tmp_dir', type=str, default='/tmp/stock_parquet')
    args = parser.parse_args()
    logger = logging.Logger("populate_prices")
    logger.setLevel(logging.INFO)
    streamHandler = logging.StreamHandler()
    logger.addHandler(streamHandler)
    bq_logger = logging.getLogger('')
    bq_logger.setLevel(logging.ERROR)
    main(
        project_id=args.project_id, 
        market=args.market, 
        limit_rows=args.limit_rows,
        num_proc=args.num_proc,
        tmp_dir=args.tmp_dir,
    )
    
    