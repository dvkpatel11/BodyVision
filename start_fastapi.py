"""BodyVision FastAPI Server - Smart dispatcher for dev/prod modes."""

import uvicorn
import os
import sys
from app.core.config import get_settings

def main():
    settings = get_settings()
    
    # Check for command line arguments
    is_prod = '--prod' in sys.argv or '--production' in sys.argv
    is_dev = '--dev' in sys.argv or '--development' in sys.argv
    
    # Default to development if no flag specified
    if not is_prod and not is_dev:
        print("💡 No mode specified. Use:")
        print("   --dev or --development for development mode")
        print("   --prod or --production for production mode")
        print("")
        print("🔧 Defaulting to DEVELOPMENT mode...")
        is_dev = True
    
    if is_dev:
        print("🚀 Starting in DEVELOPMENT mode")
        print("==============================")
        print("✅ Hot reload enabled")
        print("✅ Debug logging")
        print("✅ Single worker")
        print("🌐 Docs: http://localhost:8000/docs")
        print("")
        
        # Development configuration
        os.environ['DEBUG'] = 'true'
        os.environ['BODYVISION_DEV_MODE'] = 'true'
        
        uvicorn.run(
            "app.main:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=True,
            workers=1,
            log_level="debug",
            reload_dirs=["app"],
            access_log=True
        )
        
    elif is_prod:
        print("🚀 Starting in PRODUCTION mode")
        print("==============================")
        print("⚡ Multi-worker mode")
        print("🛡️ Production optimizations")
        print("📊 Performance monitoring")
        print("")
        
        # Production configuration
        os.environ['DEBUG'] = 'false'
        os.environ['BODYVISION_DEV_MODE'] = 'false'
        
        # Calculate workers
        import multiprocessing
        workers = min(multiprocessing.cpu_count(), 4)
        print(f"👥 Using {workers} workers")
        print("")
        
        uvicorn.run(
            "app.main:app",
            host=settings.HOST,
            port=settings.PORT,
            workers=workers,
            reload=False,
            log_level="info",
            access_log=False
        )

if __name__ == "__main__":
    main()
