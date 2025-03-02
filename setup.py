import os
import sys
import subprocess
import platform

def is_windows():
    return platform.system().lower() == 'windows'

def run_command(command):
    """Run a command and return its output and success status"""
    try:
        result = subprocess.run(command, shell=True, check=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def setup_environment():
    """Set up the environment for the Lane Detection System"""
    print("Setting up Lane Detection System Environment")
    print("===========================================")
    
    # Create necessary directories
    print("\nCreating necessary directories...")
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('processed', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Check if virtual environment exists
    venv_dir = 'venv'
    if not os.path.exists(venv_dir):
        print("\nCreating virtual environment...")
        success, output = run_command(f"{sys.executable} -m venv {venv_dir}")
        if not success:
            print(f"Failed to create virtual environment: {output}")
            print("Trying with virtualenv...")
            success, output = run_command(f"{sys.executable} -m pip install virtualenv")
            if success:
                success, output = run_command(f"{sys.executable} -m virtualenv {venv_dir}")
                if not success:
                    print(f"Failed to create virtual environment with virtualenv: {output}")
                    return False
            else:
                print(f"Failed to install virtualenv: {output}")
                return False
    
    # Activate virtual environment and install requirements
    print("\nInstalling requirements...")
    
    # Determine the pip command to use within the virtual environment
    if is_windows():
        pip_cmd = f"{venv_dir}\\Scripts\\pip"
    else:
        pip_cmd = f"{venv_dir}/bin/pip"
    
    success, output = run_command(f"{pip_cmd} install -r requirements.txt")
    if not success:
        print(f"Failed to install requirements: {output}")
        return False
    
    # Create demo video if it doesn't exist
    if not os.path.exists('processed/demo.mp4'):
        print("\nCreating demo video...")
        if is_windows():
            python_cmd = f"{venv_dir}\\Scripts\\python"
        else:
            python_cmd = f"{venv_dir}/bin/python"
        
        success, output = run_command(f"{python_cmd} create_demo.py")
        if not success:
            print(f"Failed to create demo video: {output}")
            print("You can still run the application, but the demo video won't be available.")
    
    # Create fallback video if it doesn't exist
    if not os.path.exists('static/fallback.mp4'):
        print("\nCreating fallback video...")
        if is_windows():
            python_cmd = f"{venv_dir}\\Scripts\\python"
        else:
            python_cmd = f"{venv_dir}/bin/python"
        
        success, output = run_command(f"{python_cmd} create_fallback.py")
        if not success:
            print(f"Failed to create fallback video: {output}")
            print("You can still run the application, but the fallback video won't be available.")
    
    print("\nSetup complete!")
    print("\nTo run the application:")
    if is_windows():
        print("1. Run: venv\\Scripts\\python app.py")
    else:
        print("1. Run: venv/bin/python app.py")
    print("2. Open your web browser and navigate to http://localhost:5000")
    
    return True

if __name__ == "__main__":
    setup_environment() 