import sys
import platform

def check_python_version():
    print("Python Version Check")
    print("===================")
    
    # Get Python version
    python_version = sys.version
    print(f"Python version: {python_version}")
    
    # Check if version is 3.7 or higher
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 7):
        print("ERROR: Python 3.7 or higher is required!")
        print(f"Current version: {major}.{minor}")
        return False
    else:
        print(f"✓ Python version {major}.{minor} meets requirements")
    
    # Check for required modules
    required_modules = ['numpy', 'cv2', 'tensorflow', 'flask']
    missing_modules = []
    
    print("\nChecking required modules:")
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module} is installed")
        except ImportError:
            missing_modules.append(module)
            print(f"✗ {module} is NOT installed")
    
    if missing_modules:
        print("\nMissing modules:")
        print("  " + ", ".join(missing_modules))
        print("\nPlease install missing modules using:")
        print("pip install -r requirements.txt")
        return False
    
    print("\nAll checks passed! Your system is ready to run the Lane Detection System.")
    return True

if __name__ == "__main__":
    check_python_version() 