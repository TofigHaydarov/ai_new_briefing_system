from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    
    ENV: str = Field(default="development")
    LOG_LEVEL: str = Field(default="INFO")

    LLM_PROVIDER: str = Field(default="openai")
    LLM_MODEL: str = Field(default="gpt-4o-mini")
    OPENAI_API_KEY: str = Field(default="")
    ANTHROPIC_API_KEY: str = Field(default="")
    GEMINI_API_KEY: str = Field(default="")

    #DATABASE_URL: str = Field()
    STORAGE_TYPE: str = Field(default="json", description="Storage backend: 'json' or 'postgres'")
    JSON_STORAGE_PATH: Path = Field(default=Path("data/user_profile.json"))
    DIGEST_OUTPUT_DIR: Path = Field(default=Path("digests"))

    DEDUP_SIMILARITY_THRESHOLD :int = Field(default=0.85)
    MAX_CONCURRENT_REQUESTS: int = Field(default=5)
    REQUEST_TIMEOUT_SECONDS: int = Field(default=15)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()