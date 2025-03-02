from flask import Flask, render_template, request, jsonify, send_file, Response
import os
import cv2
import numpy as np
import tempfile
import threading
import time
from test import LaneDetectionSystem
import uuid

app = Flask(__name__, static_folder='static', template_folder='.')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['STATIC_FOLDER'] = 'static'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

# Initialize the lane detection system
lane_system = LaneDetectionSystem()

# Store processing status
processing_jobs = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
    
    # Generate a unique ID for this job
    job_id = str(uuid.uuid4())
    
    # Save the uploaded file
    filename = f"{job_id}_{video_file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(filepath)
    
    # Set up the output path
    output_filename = f"{job_id}_processed.mp4"
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
    
    # Start processing in a separate thread
    processing_jobs[job_id] = {
        'status': 'processing',
        'progress': 0,
        'input_path': filepath,
        'output_path': output_path,
        'data': {
            'current_lane': 'unknown',
            'lane_change_safe': False,
            'nearest_vehicle': 'unknown'
        }
    }
    
    thread = threading.Thread(target=process_video, args=(job_id, filepath, output_path))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'job_id': job_id,
        'status': 'processing'
    })

def process_video(job_id, input_path, output_path):
    try:
        # Process the video using the lane detection system
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            processing_jobs[job_id]['status'] = 'error'
            processing_jobs[job_id]['error'] = 'Could not open video'
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process the frame
            result_frame = lane_system.process_frame(frame)
            
            # Update job data with current lane information
            processing_jobs[job_id]['data']['current_lane'] = lane_system.current_lane
            processing_jobs[job_id]['data']['lane_change_safe'] = lane_system.lane_change_safe
            
            # Write to output video
            writer.write(result_frame)
            
            # Update progress
            frame_count += 1
            processing_jobs[job_id]['progress'] = min(99, int((frame_count / total_frames) * 100))
        
        # Release resources
        cap.release()
        writer.release()
        
        # Mark job as complete
        processing_jobs[job_id]['status'] = 'complete'
        processing_jobs[job_id]['progress'] = 100
        
    except Exception as e:
        processing_jobs[job_id]['status'] = 'error'
        processing_jobs[job_id]['error'] = str(e)

@app.route('/job/<job_id>', methods=['GET'])
def get_job_status(job_id):
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = processing_jobs[job_id]
    return jsonify({
        'job_id': job_id,
        'status': job['status'],
        'progress': job['progress'],
        'data': job['data']
    })

@app.route('/video/<job_id>', methods=['GET'])
def get_processed_video(job_id):
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = processing_jobs[job_id]
    if job['status'] != 'complete':
        return jsonify({'error': 'Video processing not complete'}), 400
    
    return send_file(job['output_path'], mimetype='video/mp4')

@app.route('/demo-video', methods=['GET'])
def get_demo_video():
    # First try to return the processed demo video
    demo_path = os.path.join(app.config['PROCESSED_FOLDER'], 'demo.mp4')
    if os.path.exists(demo_path):
        return send_file(demo_path, mimetype='video/mp4')
    
    # If that doesn't exist, try to use the fallback video
    fallback_path = os.path.join(app.config['STATIC_FOLDER'], 'fallback.mp4')
    if not os.path.exists(fallback_path):
        # Create the fallback video if it doesn't exist
        try:
            from create_fallback import create_fallback_video
            create_fallback_video(fallback_path)
        except Exception as e:
            return jsonify({'error': f'Could not create fallback video: {str(e)}'}), 500
    
    if os.path.exists(fallback_path):
        return send_file(fallback_path, mimetype='video/mp4')
    else:
        return jsonify({'error': 'No video available'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 