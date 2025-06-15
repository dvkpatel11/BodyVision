"""BodyVision Production Server - Optimized for production deployment."""

import uvicorn
import os
from app.core.config import get_settings

# Force production mode
os.environ['DEBUG'] = 'false'
os.environ['BODYVISION_DEV_MODE'] = 'false'

if __name__ == "__main__":
    print("🚀 Starting BodyVision Production Server")
    print("========================================")
    print("⚡ Multi-worker mode")
    print("🛡️ Production optimizations")
    print("📊 Performance logging")
    print("🔒 Security headers enabled")
    print("🌐 API Docs: http://localhost:8000/docs")
    print("")
    
    # Get CPU count for worker calculation
    import multiprocessing
    workers = min(multiprocessing.cpu_count(), 4)  # Max 4 workers
    
    print(f"👥 Starting {workers} workers")
    print("")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        workers=workers,       # 👥 Multiple workers for performance
        reload=False,          # 🚫 No reload in production
        log_level="info",      # 📊 Production logging level
        access_log=False,      # 🚫 Disable detailed access logs for performance
        loop="uvloop",         # ⚡ Fast event loop (if available)
        http="httptools"       # ⚡ Fast HTTP parser (if available)
    )
