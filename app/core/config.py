"""Application configuration management."""

from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Set, List
import os

class Settings(BaseSettings):
    # App settings
    APP_NAME: str = "BodyVision"
    VERSION: str = "2.0.0"
    DEBUG: bool = False
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Model paths - FIXED: Proper defaults
    DEPTH_MODEL_PATH: str = "app/models/weights/best_depth_Ours_Bilinear_inc_3_net_G.pth"
    DETECTION_MODEL_PATH: str = "app/models/weights/csv_retinanet_25.pt"  # Legacy support
    CLASSES_PATH: str = "config/classes.csv"
    
    # MediaPipe Settings - NEW: Proper MediaPipe configuration
    MEDIAPIPE_MODEL_COMPLEXITY: int = 1  # 0=lite, 1=full, 2=heavy
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE: float = 0.7
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE: float = 0.5
    MEDIAPIPE_STATIC_IMAGE_MODE: bool = True
    
    # File upload limits - FIXED: Added missing settings
    MAX_FILE_SIZE: int = 15 * 1024 * 1024  # 15MB for 3 photos
    ALLOWED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp"}
    MAX_IMAGE_DIMENSION: int = 2048  # Max width/height
    MIN_IMAGE_DIMENSION: int = 400   # Min width/height
    JPEG_QUALITY: int = 90
    
    # Analysis Settings - NEW: Analysis configuration
    ENABLE_3_PHOTO_MODE: bool = True
    REQUIRE_ALL_3_PHOTOS: bool = True
    CONFIDENCE_THRESHOLD: float = 0.7
    ENABLE_ENHANCED_METRICS: bool = True
    
    # Health Metrics Configuration - NEW
    ENABLE_POSTURE_ANALYSIS: bool = True
    ENABLE_SYMMETRY_ANALYSIS: bool = True
    ENABLE_RISK_ASSESSMENT: bool = True
    
    # Security (for future JWT implementation)
    SECRET_KEY: str = "bodyvision-change-in-production-2024"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS settings - FIXED: Better defaults
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",    # React dev
        "http://localhost:7860",    # Gradio
        "http://localhost:19006",   # React Native
        "http://127.0.0.1:3000",
        "http://127.0.0.1:7860"
    ]
    ALLOW_CREDENTIALS: bool = False
    ALLOWED_METHODS: List[str] = ["GET", "POST", "OPTIONS"]
    ALLOWED_HEADERS: List[str] = ["*"]
    
    # Performance Settings - NEW
    MAX_CONCURRENT_ANALYSES: int = 5
    ANALYSIS_TIMEOUT_SECONDS: int = 45
    ENABLE_RESPONSE_CACHING: bool = False  # Disable for development
    
    # Logging Configuration - NEW  
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    ENABLE_FILE_LOGGING: bool = True
    LOG_FILE_PATH: str = "logs/app.log"
    LOG_MAX_SIZE: str = "10MB"
    LOG_BACKUP_COUNT: int = 5
    
    # Development Settings - NEW
    ENABLE_DEBUG_VISUALIZATIONS: bool = False
    SAVE_DEBUG_IMAGES: bool = False
    DEBUG_OUTPUT_DIR: str = "assets/outputs/debug"
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True

@lru_cache()
def get_settings():
    """Get cached settings instance."""
    return Settings()
