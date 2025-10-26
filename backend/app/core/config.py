"""Application configuration settings."""
from typing import List
from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    """Application settings."""
    
    model_config = ConfigDict(env_file=".env", case_sensitive=True)
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Music AI App"
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000"
    ]
    
    # Database
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "password"
    POSTGRES_DB: str = "musicai"
    DATABASE_URL: str = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_SERVER}/{POSTGRES_DB}"
    
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB: str = "musicai_metadata"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Cloud Storage
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_BUCKET_NAME: str = "musicai-storage"
    AWS_REGION: str = "us-east-1"
    
    # AI Model Settings
    AI_MODEL_PATH: str = "./models"
    MAX_TRACK_LENGTH: int = 180  # 3 minutes in seconds
    MIN_TRACK_LENGTH: int = 30   # 30 seconds


settings = Settings()
