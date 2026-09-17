import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
class Settings(BaseSettings):
    LLM_PROVIDER: str = Field(default="gemini")
    LLM_MODEL: str = Field(default="gemini-3.5-flash")
    
    GOOGLE_API_KEY: str = Field(default="")
    GEMINI_API_KEY: str = Field(default="")
    LLM_API_KEY: str = Field(default="")
    LOG_LEVEL: str = Field(default="INFO")
    EMBEDDING_PROVIDER: str = Field(default="gemini")
    EMBEDDING_MODEL: str = Field(default="text-embedding-004")
    EMBEDDING_API_KEY: str = Field(default="")

    JSON_STORAGE_PATH: Path = Field(default=PROJECT_ROOT / "data" / "user_profile.json")
    DIGEST_OUTPUT_DIR: Path = Field(default=PROJECT_ROOT / "digests")

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

active_key = settings.GOOGLE_API_KEY or settings.GEMINI_API_KEY or settings.LLM_API_KEY
if active_key:
    os.environ["GOOGLE_API_KEY"] = active_key
    os.environ["GEMINI_API_KEY"] = active_key
    os.environ["LLM_API_KEY"] = active_key
    os.environ["EMBEDDING_API_KEY"] = active_key

os.environ["LLM_PROVIDER"] = settings.LLM_PROVIDER
os.environ["LLM_MODEL"] = settings.LLM_MODEL
os.environ["EMBEDDING_PROVIDER"] = settings.EMBEDDING_PROVIDER
os.environ["EMBEDDING_MODEL"] = settings.EMBEDDING_MODEL
#print(settings.GOOGLE_API_KEY,'-')
