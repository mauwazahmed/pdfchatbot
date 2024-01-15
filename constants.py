import os 
import chromadb
from pydantic_settings import BaseSettings 
CHROMA_SETTINGS = BaseSettings(
        chroma_db_impl='duckdb+parquet',
        persist_directory='db',
        anonymized_telemetry=False
)
