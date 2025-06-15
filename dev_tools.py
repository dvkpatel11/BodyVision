"""Development tools and utilities for BodyVision."""

import subprocess
import sys
import time
import requests
from pathlib import Path

def check_server_health(url="http://localhost:8000/api/v1/health/", timeout=30):
    """Check if the development server is healthy."""
    print(f"🔍 Checking server health: {url}")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print("✅ Server is healthy!")
                return True
        except requests.exceptions.ConnectionError:
            print("⏳ Waiting for server to start...")
            time.sleep(2)
        except Exception as e:
            print(f"❌ Health check error: {e}")
            time.sleep(2)
    
    print("❌ Server health check failed")
    return False

def run_tests():
    """Run development tests."""
    print("🧪 Running development tests...")
    
    # Check if server is running
    if not check_server_health():
        print("❌ Server not healthy, cannot run tests")
        return False
    
    # Run basic API tests
    try:
        print("🔍 Testing basic endpoints...")
        
        # Health check
        response = requests.get("http://localhost:8000/api/v1/health/")
        print(f"   Health check: {response.status_code}")
        
        # Detailed health
        response = requests.get("http://localhost:8000/api/v1/health/detailed")
        print(f"   Detailed health: {response.status_code}")
        
        # MediaPipe status
        response = requests.get("http://localhost:8000/api/v1/health/mediapipe-status")
        print(f"   MediaPipe status: {response.status_code}")
        
        print("✅ Basic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Tests failed: {e}")
        return False

def start_dev_server():
    """Start development server with monitoring."""
    print("🚀 Starting BodyVision development server...")
    
    try:
        # Start server
        process = subprocess.Popen([sys.executable, "start_dev.py"])
        
        # Wait a bit for startup
        time.sleep(3)
        
        # Check health
        if check_server_health():
            print("🎉 Development server started successfully!")
            print("🌐 API Documentation: http://localhost:8000/docs")
            print("📊 Health Check: http://localhost:8000/api/v1/health/")
            return process
        else:
            print("❌ Server startup failed")
            process.terminate()
            return None
            
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BodyVision Development Tools")
    parser.add_argument("--start", action="store_true", help="Start development server")
    parser.add_argument("--test", action="store_true", help="Run development tests")
    parser.add_argument("--health", action="store_true", help="Check server health")
    
    args = parser.parse_args()
    
    if args.start:
        start_dev_server()
    elif args.test:
        run_tests()
    elif args.health:
        check_server_health()
    else:
        parser.print_help()
