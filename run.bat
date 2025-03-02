@echo off
echo Lane Detection and Driving Assistance System
echo ==========================================
echo.

REM Check if Python is installed and in PATH
where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    goto :python_found
)

REM Check common Python installation locations
set PYTHON_LOCATIONS=^
C:\Python37\python.exe^
C:\Python38\python.exe^
C:\Python39\python.exe^
C:\Python310\python.exe^
C:\Python311\python.exe^
C:\Python312\python.exe^
C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python37\python.exe^
C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python38\python.exe^
C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python39\python.exe^
C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310\python.exe^
C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe^
C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe^
C:\Program Files\Python37\python.exe^
C:\Program Files\Python38\python.exe^
C:\Program Files\Python39\python.exe^
C:\Program Files\Python310\python.exe^
C:\Program Files\Python311\python.exe^
C:\Program Files\Python312\python.exe^
C:\Program Files (x86)\Python37\python.exe^
C:\Program Files (x86)\Python38\python.exe^
C:\Program Files (x86)\Python39\python.exe^
C:\Program Files (x86)\Python310\python.exe^
C:\Program Files (x86)\Python311\python.exe^
C:\Program Files (x86)\Python312\python.exe

for %%p in (%PYTHON_LOCATIONS%) do (
    if exist %%p (
        echo Found Python at: %%p
        set PYTHON_PATH=%%p
        goto :python_found_at_path
    )
)

echo Python is not installed or not found in common locations.
echo Please install Python 3.7 or higher and make sure to check "Add Python to PATH" during installation.
echo You can download Python from: https://www.python.org/downloads/
goto :end

:python_found_at_path
echo Using Python found at: %PYTHON_PATH%
set PYTHON_CMD=%PYTHON_PATH%
goto :continue_setup

:python_found
echo Python found in PATH
set PYTHON_CMD=python

:continue_setup
REM Check Python version and required modules
echo Checking Python version and required modules...
%PYTHON_CMD% check_python.py
if %ERRORLEVEL% neq 0 (
    echo Python version check failed. Please ensure you have Python 3.7+ and all required modules.
    goto :end
)

REM Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    %PYTHON_CMD% -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo Failed to create virtual environment. Please install venv package.
        echo You can install it with: %PYTHON_CMD% -m pip install virtualenv
        goto :end
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements if needed
if not exist venv\Lib\site-packages\flask (
    echo Installing requirements...
    pip install -r requirements.txt
)

REM Create demo video if it doesn't exist
if not exist processed\demo.mp4 (
    echo Creating demo video...
    python create_demo.py
)

REM Create fallback video if it doesn't exist
if not exist static\fallback.mp4 (
    echo Creating fallback video...
    python create_fallback.py
)

REM Run the application
echo Starting the application...
echo.
echo Please open your web browser and navigate to http://localhost:5000
echo.
python app.py

:end
pause 