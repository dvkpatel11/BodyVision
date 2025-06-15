"""Test script for FastAPI endpoints."""

import requests
import json
import os
from PIL import Image
import io

def test_health_endpoint(base_url="http://localhost:8000"):
    """Test health check endpoint."""
    print("🔍 Testing health endpoint...")
    
    try:
        response = requests.get(f"{base_url}/api/v1/health/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Test detailed health
        response = requests.get(f"{base_url}/api/v1/health/detailed")
        print(f"Detailed health: {response.json()}")
        
        return True
    except Exception as e:
        print(f"Health test failed: {e}")
        return False

def test_analysis_endpoint(base_url="http://localhost:8000"):
    """Test analysis endpoint."""
    print("🔍 Testing analysis endpoint...")
    
    try:
        # Create test image or use sample
        if os.path.exists("assets/samples/204.jpg"):
            image_path = "assets/samples/204.jpg"
        else:
            # Create dummy image
            img = Image.new('RGB', (400, 300), color='red')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            with open('test_image.jpg', 'wb') as f:
                f.write(img_bytes.getvalue())
            image_path = 'test_image.jpg'
        
        # Test analysis
        with open(image_path, 'rb') as f:
            response = requests.post(
                f"{base_url}/api/v1/analysis/analyze",
                files={"image": f},
                data={
                    "height": 175,
                    "weight": 70,
                    "age": 25,
                    "sex": "male"
                },
                timeout=30
            )
        
        print(f"Analysis status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("✅ Analysis successful!")
            print(f"Body fat: {result['body_composition']['body_fat_percentage']}%")
            print(f"Processing time: {result['processing_time_seconds']}s")
        else:
            print(f"Analysis failed: {response.text}")
        
        # Cleanup
        if image_path == 'test_image.jpg':
            os.unlink('test_image.jpg')
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"Analysis test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing FastAPI endpoints...")
    
    health_ok = test_health_endpoint()
    analysis_ok = test_analysis_endpoint()
    
    if health_ok and analysis_ok:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
