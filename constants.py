import os 
import chromadb
from pydantic_settings import BaseSettings,SettingsConfigDict

class Settings(BaseSettings):
    CHROMA_SETTINGS = SettingsConfigDict(
        chroma_db_impl='duckdb+parquet',
        persist_directory='db',
        anonymized_telemetry=False
    )

Settings()
