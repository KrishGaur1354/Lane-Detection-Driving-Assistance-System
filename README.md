# Lane Detection System

A computer vision system that detects lane markings and vehicles in dashboard camera footage.

## Features

- Lane detection using OpenCV
- Vehicle detection and distance estimation
- Simple web interface for video upload and processing
- Real-time video processing with visual feedback

## Requirements

- Python 3.7 or higher
- OpenCV
- TensorFlow
- Flask
- NumPy

## Quick Start

### Windows Users

1. Double-click `run.bat`
2. Open your browser and go to `http://localhost:5000`

### Other Users (or Manual Setup)

1. Install Python 3.7+ from [python.org](https://www.python.org/downloads/)
2. Open a terminal in the project directory
3. Run:
   ```bash
   python setup.py
   python app.py
   ```
4. Open your browser and go to `http://localhost:5000`

## Usage

1. Open the web interface at `http://localhost:5000`
2. Either:
   - Watch the demo video
   - Upload your own dashboard camera footage (MP4 format, max 100MB)
3. The system will process the video and display the results with lane detection overlay

## Project Structure

```
lane-detection/
├── app.py              # Flask web server
├── test.py             # Lane detection algorithm
├── create_demo.py      # Demo video generator
├── create_fallback.py  # Fallback video generator
├── setup.py           # Setup script
├── run.bat            # Windows setup script
├── requirements.txt   # Python dependencies
├── static/           # Static web assets
├── uploads/          # Uploaded videos
└── processed/        # Processed videos
```

## How It Works

1. **Video Input**: The system accepts dashboard camera footage through the web interface
2. **Lane Detection**: 
   - Converts frames to grayscale
   - Applies Gaussian blur and Canny edge detection
   - Uses Hough transform to detect lane lines
3. **Vehicle Detection**:
   - Uses computer vision to detect vehicles
   - Estimates distances based on vehicle size
4. **Output**: Displays processed video with lane markings and vehicle detection

## Troubleshooting

### Common Issues

1. **Python Not Found**
   - Install Python 3.7 or higher
   - Make sure to check "Add Python to PATH" during installation

2. **Missing Dependencies**
   - Run: `pip install -r requirements.txt`

3. **Video Processing Fails**
   - Check that the video is in MP4 format
   - Ensure the video is under 100MB
   - Verify that the video contains clear lane markings

### Error Messages

- "Python is not installed or not in PATH"
  - Solution: Reinstall Python with "Add Python to PATH" checked

- "Could not open video"
  - Solution: Check video file format and codec

## License

This project is licensed under the MIT License. See the LICENSE file for details. 