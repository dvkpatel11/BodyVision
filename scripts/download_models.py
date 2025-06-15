"""Script to download or help locate required model weights."""

import sys
from pathlib import Path


def check_models():
    """Check if required model files exist."""

    models_dir = Path("app/models/weights")
    models_dir.mkdir(parents=True, exist_ok=True)

    required_models = {
        "best_depth_Ours_Bilinear_inc_3_net_G.pth": "Depth estimation model",
        "csv_retinanet_25.pt": "Body part detection model",
    }

    missing_models = []

    print("🔍 Checking for required model files...")

    for model_file, description in required_models.items():
        model_path = models_dir / model_file
        if model_path.exists():
            print(f"✅ {description}: Found at {model_path}")
        else:
            print(f"❌ {description}: Missing at {model_path}")
            missing_models.append((model_file, description))

    if missing_models:
        print("\n⚠️  Missing model files:")
        for model_file, description in missing_models:
            print(f"   - {model_file}: {description}")

        print("\n📋 To resolve:")
        print("1. Check your backup directory for these model files")
        print("2. Copy them to app/models/weights/")
        print("3. Or retrain the models using your original training scripts")

        return False
    else:
        print("\n✅ All required models found!")
        return True


if __name__ == "__main__":
    success = check_models()
    sys.exit(0 if success else 1)
