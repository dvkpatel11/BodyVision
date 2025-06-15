"""Main entry point for BodyVision Gradio application."""

import os
import sys
from pathlib import Path

# Add app directory to Python path
app_dir = Path(__file__).parent / 'app'
sys.path.insert(0, str(app_dir))

from app.api.gradio_interface import create_app
from app.utils.logger import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    logger.info("🚀 Starting BodyVision Gradio application")
    
    try:
        app = create_app()
        logger.info("✅ Gradio app created successfully")
        
        # Launch the app
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True,
            show_error=True
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start Gradio app: {e}")
        raise
