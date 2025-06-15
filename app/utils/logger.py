"""Logging configuration for the application."""

import logging
import logging.config
import yaml
import os
from pathlib import Path


def setup_logging(config_path: str = "config/logging_config.yaml"):
    """Setup logging configuration."""
    
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    else:
        # Fallback to basic configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )


def get_logger(name: str) -> logging.Logger:
    """Get logger instance."""
    setup_logging()
    return logging.getLogger(name)
