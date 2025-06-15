"""Application configuration management."""

from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Set

class Settings(BaseSettings):
    # App settings
    APP_NAME: str = "BodyVision"
    VERSION: str = "2.0.0"
    DEBUG: bool = True
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Model paths
    DEPTH_MODEL_PATH: str = "app/models/weights/best_depth_Ours_Bilinear_inc_3_net_G.pth"
    DETECTION_MODEL_PATH: str = "app/models/weights/csv_retinanet_25.pt"
    CLASSES_PATH: str = "config/classes.csv"
    
    # Security (for future JWT implementation)
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # File upload limits
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png"}
    
    # CORS settings
    ALLOWED_ORIGINS: list = ["*"]  # Configure properly for production
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
