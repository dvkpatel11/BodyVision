"""BodyVision Development Server - Hot reload, debug mode, single worker."""

import uvicorn
import os

# Force development mode
os.environ['DEBUG'] = 'true'
os.environ['BODYVISION_DEV_MODE'] = 'true'

if __name__ == "__main__":
    print("🚀 Starting BodyVision Development Server")
    print("=========================================")
    print("✅ Hot reload enabled")
    print("✅ Debug mode enabled") 
    print("✅ Single worker for debugging")
    print("✅ Verbose logging")
    print("🌐 API Docs: http://localhost:8000/docs")
    print("📊 Health: http://localhost:8000/api/v1/health/")
    print("")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,           # 🔄 Hot reload for development
        workers=1,             # 🔧 Single worker for debugging
        log_level="debug",     # 📝 Verbose logging
        reload_dirs=["app"],   # 👀 Watch app directory for changes
        access_log=True        # 📋 Request logging
    )
