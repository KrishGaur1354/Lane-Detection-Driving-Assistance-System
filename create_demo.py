import os
import cv2
import numpy as np
from test import LaneDetectionSystem

def create_demo_video(output_path='processed/demo.mp4', duration=10, fps=30, width=1280, height=720):
    """
    Create a demo video with simulated lane markings for testing the system.
    
    Args:
        output_path: Path to save the demo video
        duration: Duration of the video in seconds
        fps: Frames per second
        width: Video width
        height: Video height
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize lane detection system
    lane_system = LaneDetectionSystem()
    
    # Generate frames
    total_frames = duration * fps
    
    for i in range(total_frames):
        # Create a blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add a gray road
        cv2.rectangle(frame, (0, height//2), (width, height), (80, 80, 80), -1)
        
        # Add lane markings
        # Left lane marking
        left_x = width // 3 + int(20 * np.sin(i / 30))  # Add some movement
        cv2.line(frame, (left_x, height), (width // 2 - 100, height // 2), (255, 255, 255), 10)
        
        # Right lane marking
        right_x = 2 * width // 3 + int(20 * np.sin(i / 30))  # Add some movement
        cv2.line(frame, (right_x, height), (width // 2 + 100, height // 2), (255, 255, 255), 10)
        
        # Add a vehicle in front (rectangle)
        if i % 90 < 60:  # Make the vehicle appear and disappear
            vehicle_width = 100
            vehicle_height = 80
            vehicle_x = width // 2 - vehicle_width // 2
            vehicle_y = height // 2 - vehicle_height - 50 + int(10 * np.sin(i / 15))  # Add some movement
            cv2.rectangle(frame, (vehicle_x, vehicle_y), 
                         (vehicle_x + vehicle_width, vehicle_y + vehicle_height), 
                         (0, 0, 255), -1)
        
        # Process the frame with the lane detection system
        result_frame = lane_system.process_frame(frame)
        
        # Write the frame
        writer.write(result_frame)
        
        # Display progress
        if i % fps == 0:
            print(f"Processing frame {i}/{total_frames} ({i/total_frames*100:.1f}%)")
    
    # Release resources
    writer.release()
    print(f"Demo video created at {output_path}")

if __name__ == "__main__":
    create_demo_video() 