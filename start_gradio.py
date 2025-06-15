"""Start BodyVision Gradio interface."""

import sys
from pathlib import Path

# Add app directory to Python path
app_dir = Path(__file__).parent / 'app'
sys.path.insert(0, str(app_dir))

from app.api.gradio_interface import create_app
from app.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    """Start the Gradio interface."""
    print("🎨 Starting BodyVision Gradio Interface")
    print("=====================================")
    print("✅ Production-ready interface")
    print("✅ MediaPipe body detection")
    print("✅ Real-time health insights")
    print("🌐 Interface: http://localhost:7860")
    print("")
    
    try:
        app = create_app()
        logger.info("✅ Gradio interface ready")
        
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True,
            show_error=True,
            quiet=False
        )
        
    except Exception as e:
        logger.error(f"❌ Gradio startup failed: {e}")
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Ensure FastAPI services work: python3 -c 'from app.services import create_analysis_service; print(\"OK\")'")
        print("   2. Check port 7860 is available: lsof -i :7860")
        print("   3. Try different port: modify server_port in start_gradio.py")
        raise

if __name__ == "__main__":
    main()
