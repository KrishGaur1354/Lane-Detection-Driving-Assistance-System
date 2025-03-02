LANE DETECTION AND DRIVING ASSISTANCE SYSTEM
===========================================

INSTALLATION INSTRUCTIONS
------------------------

OPTION 1: USING THE BATCH FILE (WINDOWS ONLY)
---------------------------------------------
1. Double-click on "run.bat"
2. If Python is not found, the script will guide you to install it
3. Follow the on-screen instructions

OPTION 2: USING PYTHON DIRECTLY
------------------------------
1. Install Python 3.7 or higher from https://www.python.org/downloads/
   - During installation, make sure to check "Add Python to PATH"

2. Open a command prompt or terminal in this folder

3. Run the setup script:
   python setup.py

4. Start the application:
   - On Windows: venv\Scripts\python app.py
   - On macOS/Linux: venv/bin/python app.py

5. Open your web browser and go to: http://localhost:5000

TROUBLESHOOTING
--------------
If you encounter the "Python is not installed or not in PATH" error:

1. Make sure Python is installed
2. Add Python to your PATH environment variable:
   - Search for "Environment Variables" in Windows search
   - Click "Edit the system environment variables"
   - Click "Environment Variables"
   - Under "System variables", find "Path" and click "Edit"
   - Click "New" and add the path to your Python installation
     (e.g., C:\Python39 and C:\Python39\Scripts)
   - Click "OK" on all dialogs

3. Try running the batch file again

MANUAL SETUP
-----------
If the automated setup doesn't work:

1. Create a virtual environment:
   python -m venv venv

2. Activate the virtual environment:
   - Windows: venv\Scripts\activate
   - macOS/Linux: source venv/bin/activate

3. Install requirements:
   pip install -r requirements.txt

4. Create the demo video:
   python create_demo.py

5. Run the application:
   python app.py

6. Open your web browser and go to: http://localhost:5000

SYSTEM REQUIREMENTS
------------------
- Python 3.7 or higher
- 4GB RAM or more
- Modern web browser (Chrome, Firefox, Edge)
- Webcam (optional, for live video processing)

For more detailed information, see README.md 