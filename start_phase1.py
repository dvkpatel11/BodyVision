#!/usr/bin/env python3
"""Quick start script for BodyVision Phase 1 - 3-Photo Analysis."""

import subprocess
import sys
import time
import os
from pathlib import Path

def check_requirements():
    """Check if basic requirements are met."""
    print("🔍 Checking Phase 1 requirements...")
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ required")
        return False
    
    # Check if in BodyVision directory
    if not Path("app").exists():
        print("❌ Please run from BodyVision root directory")
        return False
    
    # Check if requirements are installed
    try:
        import fastapi
        import gradio
        import mediapipe
        print("✅ Core packages found")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def start_services():
    """Start Phase 1 services."""
    if not check_requirements():
        return
    
    print("\n🚀 Starting BodyVision Phase 1 - 3-Photo Analysis")
    print("=" * 50)
    print("📸 Features: 3-Photo comprehensive analysis")
    print("📊 Metrics: 9 health metrics including body fat, BMI, ratios")
    print("🎨 Interface: Gradio development UI")
    print("🌐 Backend: FastAPI with MediaPipe detection")
    print("=" * 50)
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    print("\n📡 Starting FastAPI backend...")
    print("   API: http://localhost:8000")
    print("   Docs: http://localhost:8000/docs")
    
    print("\n🎨 Starting Gradio interface...")
    print("   UI: http://localhost:7860")
    
    print("\n⏳ Services starting...")
    
    # Start the development server
    try:
        subprocess.run([sys.executable, "start_dev.py"])
    except KeyboardInterrupt:
        print("\n🛑 Phase 1 services stopped")

if __name__ == "__main__":
    start_services()
