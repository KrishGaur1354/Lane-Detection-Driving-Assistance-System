# Lane Detection and Driving Assistance System

This project implements a lane detection and driving assistance system using computer vision and deep learning. It includes a web interface for uploading and processing dashboard camera footage.

## Features

- Lane detection using computer vision techniques
- Vehicle detection and distance estimation
- Lane change safety analysis
- Web interface for uploading and viewing processed videos

## Requirements

- Python 3.7+
- OpenCV
- TensorFlow
- Flask
- Other dependencies listed in requirements.txt

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:

```bash
python app.py
```

2. Open a web browser and navigate to `http://localhost:5000`

3. Upload a dashboard camera video through the web interface

4. The system will process the video and display the results with lane detection, vehicle detection, and safety analysis

## Project Structure

- `test.py`: Contains the core lane detection and vehicle detection algorithms
- `app.py`: Flask application that serves the web interface and processes videos
- `index.html`: Web interface for the application
- `uploads/`: Directory for storing uploaded videos
- `processed/`: Directory for storing processed videos

## Demo

To use the demo feature, place a pre-processed video named `demo.mp4` in the `processed/` directory.

## Notes

- The lane detection system uses computer vision techniques like Canny edge detection and Hough transform
- The vehicle detection is currently implemented using a simplified approach, but could be enhanced with a trained object detection model
- The system estimates distances to vehicles based on their apparent size in the frame

## License

This project is licensed under the MIT License - see the LICENSE file for details. 