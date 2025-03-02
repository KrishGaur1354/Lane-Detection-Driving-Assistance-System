import cv2
import numpy as np
import os

def create_fallback_video(output_path='static/fallback.mp4', duration=5, fps=30, width=640, height=360):
    """
    Create a simple fallback video with text.
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Generate frames
    total_frames = duration * fps
    
    for i in range(total_frames):
        # Create a blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (240, 240, 240)  # Light gray background
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text1 = "Lane Detection System"
        text2 = "Upload a video to begin"
        text3 = "or create a demo video"
        
        # Calculate text sizes
        text_size1 = cv2.getTextSize(text1, font, 1, 2)[0]
        text_size2 = cv2.getTextSize(text2, font, 0.8, 2)[0]
        text_size3 = cv2.getTextSize(text3, font, 0.8, 2)[0]
        
        # Calculate positions
        text_x1 = (width - text_size1[0]) // 2
        text_y1 = height // 2 - 40
        text_x2 = (width - text_size2[0]) // 2
        text_y2 = height // 2 + 20
        text_x3 = (width - text_size3[0]) // 2
        text_y3 = height // 2 + 60
        
        # Add pulsing effect
        alpha = 0.7 + 0.3 * np.sin(i * 0.1)  # Pulsing between 0.7 and 1.0
        
        # Draw text
        cv2.putText(frame, text1, (text_x1, text_y1), font, 1, (70, 70, 70), 2)
        cv2.putText(frame, text2, (text_x2, text_y2), font, 0.8, 
                   (int(70*alpha), int(70*alpha), int(70*alpha)), 2)
        cv2.putText(frame, text3, (text_x3, text_y3), font, 0.8, 
                   (int(70*alpha), int(70*alpha), int(70*alpha)), 2)
        
        # Write the frame
        writer.write(frame)
    
    # Release resources
    writer.release()
    print(f"Fallback video created at {output_path}")

if __name__ == "__main__":
    create_fallback_video() 