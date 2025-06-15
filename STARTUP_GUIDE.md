# 🚀 BodyVision Startup Guide

## Development vs Production

### 🔧 Development Mode (Hot Reload, Debug)

**Recommended for development:**
```bash
# Option 1: Dedicated dev script
python3 start_dev.py

# Option 2: Smart dispatcher
python3 start_fastapi.py --dev

# Option 3: Development tools
python3 dev_tools.py --start
```

**Features:**
- ✅ Hot reload (auto-restart on code changes)
- ✅ Debug logging and verbose output
- ✅ Single worker for easier debugging
- ✅ Development mode enabled
- ✅ Request logging enabled

### ⚡ Production Mode (Optimized, Multi-worker)

**For production deployment:**
```bash
# Option 1: Dedicated prod script
python3 start_prod.py

# Option 2: Smart dispatcher
python3 start_fastapi.py --prod

# Option 3: Docker production
docker-compose -f docker-compose.prod.yml up
```

**Features:**
- ⚡ Multi-worker for performance
- 🛡️ Production optimizations
- 📊 Performance logging
- 🚫 No reload (stable)
- 🔒 Security optimizations

### 🐳 Docker Development

```bash
# Development with hot reload
docker-compose -f docker-compose.dev.yml up

# Production deployment
docker-compose -f docker-compose.prod.yml up
```

## Quick Start Commands

```bash
# 🔧 Development (what you want most of the time)
python3 start_dev.py

# 🧪 Run tests against dev server
python3 dev_tools.py --test

# 🔍 Check server health
python3 dev_tools.py --health

# ⚡ Production testing
python3 start_prod.py
```

## URLs

- **📚 API Docs**: http://localhost:8000/docs
- **🔍 Health Check**: http://localhost:8000/api/v1/health/
- **🤖 MediaPipe Status**: http://localhost:8000/api/v1/health/mediapipe-status
- **📋 Alternative Docs**: http://localhost:8000/redoc

## Development Workflow

1. **Start dev server**: `python3 start_dev.py`
2. **Make code changes** (auto-reloads)
3. **Test in browser**: http://localhost:8000/docs
4. **Run tests**: `python3 dev_tools.py --test`
5. **Check health**: `python3 dev_tools.py --health`
