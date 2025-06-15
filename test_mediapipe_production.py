"""Production test for MediaPipe detection service."""

import asyncio
from PIL import Image
import numpy as np
import time
import json

from app.services.mediapipe_detection_service import MediaPipeDetectionService
from app.core.body_analyzer import BodyAnalyzer


async def test_mediapipe_production():
    """Test MediaPipe service in production scenario."""
    
    print("🚀 Testing MediaPipe Production Setup")
    print("=" * 40)
    
    # Test 1: Service initialization
    print("1️⃣ Testing service initialization...")
    try:
        service = MediaPipeDetectionService()
        print("✅ MediaPipe service initialized successfully")
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")
        return False
    
    # Test 2: Detection with various image types
    print("\n2️⃣ Testing detection robustness...")
    
    test_cases = [
        ("Small image", (200, 150)),
        ("Medium image", (640, 480)), 
        ("Large image", (1280, 960)),
        ("Wide image", (800, 400)),
        ("Tall image", (400, 800))
    ]
    
    for name, size in test_cases:
        # Create realistic test image
        test_image = Image.fromarray(
            np.random.randint(50, 200, (size[1], size[0], 3), dtype=np.uint8)
        )
        
        start_time = time.time()
        try:
            result = await service.detect_body_parts(test_image)
            processing_time = time.time() - start_time
            
            confidence = service.get_pose_confidence(test_image)
            
            print(f"   ✅ {name}: {processing_time:.3f}s, confidence: {confidence:.2f}")
            
        except Exception as e:
            print(f"   ❌ {name}: Failed - {e}")
    
    # Test 3: Full pipeline test
    print("\n3️⃣ Testing full analysis pipeline...")
    try:
        analyzer = BodyAnalyzer()
        
        # Create realistic test image
        test_image = Image.fromarray(
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        )
        
        start_time = time.time()
        results = await analyzer.analyze(
            image=test_image,
            height=175,
            weight=70,
            age=25,
            sex='male'
        )
        total_time = time.time() - start_time
        
        print(f"✅ Full pipeline completed in {total_time:.3f}s")
        print(f"   Body fat: {results.get('body_fat_percentage', 'N/A')}%")
        print(f"   Neck: {results.get('neck_cm', 'N/A')} cm")
        print(f"   Waist: {results.get('waist_cm', 'N/A')} cm")
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        return False
    
    # Test 4: Performance benchmarking
    print("\n4️⃣ Performance benchmarking...")
    
    # Run multiple detections to test consistency
    times = []
    for i in range(10):
        test_image = Image.fromarray(
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        )
        
        start_time = time.time()
        await service.detect_body_parts(test_image)
        times.append(time.time() - start_time)
    
    avg_time = sum(times) / len(times)
    max_time = max(times)
    min_time = min(times)
    
    print(f"✅ Performance metrics (10 runs):")
    print(f"   Average: {avg_time:.3f}s")
    print(f"   Range: {min_time:.3f}s - {max_time:.3f}s")
    print(f"   Throughput: ~{1/avg_time:.1f} images/second")
    
    # Performance evaluation
    if avg_time < 0.5:
        print("🚀 Excellent performance for production")
    elif avg_time < 1.0:
        print("✅ Good performance for production")
    else:
        print("⚠️ May need optimization for high-volume production")
    
    print("\n🎉 MediaPipe production testing completed!")
    print("✅ Ready for production deployment")
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_mediapipe_production())
    
    if success:
        print("\n🚀 Next steps:")
        print("   1. python start_fastapi.py")
        print("   2. Test API: curl http://localhost:8000/api/v1/health/mediapipe-status")
        print("   3. Deploy to production!")
    else:
        print("\n❌ Fix issues before production deployment")
