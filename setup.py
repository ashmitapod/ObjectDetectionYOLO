#!/usr/bin/env python3
"""
Setup and Verification Script for YOLO Detection System
Checks dependencies, downloads models, and verifies installation
"""

import os
import sys
import subprocess
import urllib.request

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def check_python_version():
    """Verify Python version"""
    print_header("Checking Python Version")
    
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7+ required")
        return False
    
    print("✅ Python version OK")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    print_header("Checking Dependencies")
    
    required = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib'
    }
    
    missing = []
    
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✅ {package} installed")
        except ImportError:
            print(f"❌ {package} NOT installed")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    print("\n✅ All dependencies installed")
    return True

def create_directories():
    """Create necessary directories"""
    print_header("Creating Directories")
    
    dirs = [
        "weights",
        "cfg",
        "outputs",
        "outputs/clips",
        "outputs/logs",
        "outputs/screenshots"
    ]
    
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created: {directory}")
        else:
            print(f"✓  Exists: {directory}")
    
    return True

def check_model_files():
    """Check if YOLO model files exist"""
    print_header("Checking Model Files")
    
    files = {
        "weights/yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights",
        "cfg/yolov3.cfg": None,  # Should already exist
        "coco.names": None  # Should already exist
    }
    
    all_present = True
    
    for file_path, url in files.items():
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"✅ {file_path} ({size:.1f} MB)")
        else:
            print(f"❌ {file_path} NOT FOUND")
            all_present = False
            
            if url:
                print(f"   Download from: {url}")
    
    return all_present

def download_yolo_weights():
    """Download YOLOv3 weights"""
    print_header("Downloading YOLO Weights")
    
    weights_path = "weights/yolov3.weights"
    
    if os.path.exists(weights_path):
        print("✅ Weights already downloaded")
        return True
    
    url = "https://pjreddie.com/media/files/yolov3.weights"
    
    print(f"📥 Downloading YOLOv3 weights (237 MB)...")
    print("⏳ This may take several minutes...")
    
    try:
        def progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = (downloaded / total_size) * 100
            sys.stdout.write(f"\r   Progress: {percent:.1f}% ({downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB)")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, weights_path, reporthook=progress)
        print("\n✅ Download complete!")
        return True
    
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nManual download:")
        print(f"1. Visit: {url}")
        print(f"2. Save to: {weights_path}")
        return False

def verify_opencv():
    """Verify OpenCV installation"""
    print_header("Verifying OpenCV")
    
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
        
        # Test DNN module
        net = cv2.dnn.readNet
        print("✅ OpenCV DNN module available")
        
        # Check for GPU support
        try:
            backends = cv2.dnn.getAvailableBackends()
            if 'CUDA' in str(backends):
                print("🚀 CUDA support available (GPU acceleration possible)")
            else:
                print("ℹ️  CUDA not available (CPU mode only)")
        except:
            print("ℹ️  Running in CPU mode")
        
        return True
    
    except Exception as e:
        print(f"❌ OpenCV verification failed: {e}")
        return False

def test_webcam():
    """Test webcam access"""
    print_header("Testing Webcam")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot access webcam")
            print("   Check camera permissions or connection")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            h, w, _ = frame.shape
            print(f"✅ Webcam accessible ({w}x{h})")
            return True
        else:
            print("❌ Cannot read from webcam")
            return False
    
    except Exception as e:
        print(f"❌ Webcam test failed: {e}")
        return False

def create_test_config():
    """Create a test configuration"""
    print_header("Creating Test Configuration")
    
    config = """# Test Configuration
# Edit main.py EMAIL_CONFIG section for your email settings

EMAIL_SETTINGS:
  sender: your_email@gmail.com
  password: your_app_password
  receiver: alert_email@gmail.com

ROI_ZONES:
  - name: Test Zone
    x: 400
    y: 100
    width: 250
    height: 350

ALERT_OBJECTS:
  - person
  - car
  - truck
"""
    
    with open("test_config.txt", "w") as f:
        f.write(config)
    
    print("✅ Created test_config.txt")
    print("   Edit this file and update main.py accordingly")
    return True

def print_next_steps():
    """Print next steps for user"""
    print_header("Setup Complete! Next Steps")
    
    print("""
1. Update Email Configuration (if using alerts):
   - Edit main.py lines 40-46
   - Add your Gmail and App Password

2. Adjust ROI Zones (if using ROI):
   - Edit main.py lines 49-53
   - Customize zone positions and names

3. Run Basic Test:
   python main.py

4. Run with ROI:
   python main.py --roi

5. Full System Test:
   python main.py --roi --alert-objects person

6. View Analytics:
   python analytics_viewer.py

7. Manage Videos:
   python compile_alerts.py

📚 Read README.md for complete documentation!
    """)

def main():
    print("""
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║         YOLO Detection System - Setup Script              ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    all_checks_passed = True
    
    # Run all checks
    if not check_python_version():
        all_checks_passed = False
    
    if not check_dependencies():
        all_checks_passed = False
        print("\n⚠️  Install dependencies with:")
        print("pip install -r requirements.txt")
    
    create_directories()
    
    if not check_model_files():
        print("\n⚠️  Model files missing!")
        download = input("\nDownload YOLOv3 weights now? (yes/no): ").strip().lower()
        if download == 'yes':
            if not download_yolo_weights():
                all_checks_passed = False
        else:
            print("❌ Weights required for operation")
            all_checks_passed = False
    
    if not verify_opencv():
        all_checks_passed = False
    
    test_webcam()
    create_test_config()
    
    # Final summary
    print_header("Setup Summary")
    
    if all_checks_passed:
        print("✅ ALL CHECKS PASSED!")
        print("🚀 System ready to run!")
        print_next_steps()
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("Please resolve the issues above before running the system")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()