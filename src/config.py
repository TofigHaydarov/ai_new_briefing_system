import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # LLM Settings
    LLM_PROVIDER: str = Field(default="openai")
    LLM_MODEL: str = Field(default="gpt-4o-mini")
    
    # API Keys for different providers
    OPENAI_API_KEY: str = Field(default="")
    ANTHROPIC_API_KEY: str = Field(default="")
    GOOGLE_API_KEY: str = Field(default="")
    
    # Embedding Settings
    EMBEDDING_PROVIDER: str = Field(default="openai")
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-small")
    
    # System & Database Settings
    LOG_LEVEL: str = Field(default="INFO")
    DATABASE_URL: str = Field(default="postgresql+asyncpg://postgres:dev@localhost:5432/newsbrief")
    DIGESTS_DIR: Path = Field(default=PROJECT_ROOT / "digests")
    DEDUP_NEAR_DUPLICATE_THRESHOLD: float = Field(default=0.70)
    MAX_PARALLEL_FETCHES: int = Field(default=5, ge=1)
    FETCH_TIMEOUT_SECONDS: int = Field(default=10, ge=1)

    # Backward compatibility paths if needed
    JSON_STORAGE_PATH: Path = Field(default=PROJECT_ROOT / "data" / "user_profile.json")
    DIGEST_OUTPUT_DIR: Path = Field(default=PROJECT_ROOT / "digests")

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# --- Sağlayıcıya göre doğru API anahtarını otomatik eşleme ---
# Kullanıcı .env dosyasına hangi sağlayıcıyı yazdıysa, ilgili API anahtarını 
# kütüphanelerin doğrudan okuyabileceği standart os.environ değişkenlerine aktarıyoruz.

provider = settings.LLM_PROVIDER.lower().strip()

if provider == "openai" and settings.OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
elif provider == "anthropic" and settings.ANTHROPIC_API_KEY:
    os.environ["ANTHROPIC_API_KEY"] = settings.ANTHROPIC_API_KEY
elif provider in ("gemini", "google") and settings.GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY

# Embedding için de benzer esneklik (OpenAI veya Google destekliyor)
embed_provider = settings.EMBEDDING_PROVIDER.lower().strip()
if embed_provider == "openai" and settings.OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
elif embed_provider in ("gemini", "google") and settings.GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY

# Genel ortam değişkeni atamaları
os.environ["LLM_PROVIDER"] = settings.LLM_PROVIDER
os.environ["LLM_MODEL"] = settings.LLM_MODEL
os.environ["EMBEDDING_PROVIDER"] = settings.EMBEDDING_PROVIDER
os.environ["EMBEDDING_MODEL"] = settings.EMBEDDING_MODEL
os.environ["LOG_LEVEL"] = settings.LOG_LEVEL