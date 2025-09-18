"""
Setup script for Work Activity Monitor
"""

from pathlib import Path
import subprocess
import sys


def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    print("Creating directories...")
    directories = ["models", "logs", "data"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created {directory}/ directory")


def test_installation():
    """Test the installation."""
    print("\nTesting installation...")
    try:
        # Test imports
        from work_monitor.config import ConfigManager
        from work_monitor.activity_monitor import ActivityMonitor
        from work_monitor.ml_model import ActivityClassifier
        
        print("✓ All modules imported successfully")
        
        # Test configuration loading
        config = ConfigManager("config.yaml")
        print("✓ Configuration loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Installation test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("Work Activity Monitor Setup")
    print("=" * 30)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("\nSetup failed. Please install requirements manually:")
        print("pip install -r requirements.txt")
        return
    
    # Test installation
    if test_installation():
        print("\n✓ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Edit config.yaml with your email settings")
        print("2. Run: python main.py test-email")
        print("3. Run: python main.py train")
        print("4. Run: python main.py start")
    else:
        print("\n✗ Setup completed with errors")
        print("Please check the error messages above")


if __name__ == "__main__":
    main()
