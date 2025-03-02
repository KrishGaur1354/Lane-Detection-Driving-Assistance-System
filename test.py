import cv2
import numpy as np
import tensorflow as tf
# Make sure TensorFlow is properly imported
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
except ImportError:
    # Alternative import path for newer TensorFlow versions
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import time
import math

class LaneDetectionSystem:
    def __init__(self):
        # Initialize parameters for lane detection
        self.lane_model = self.build_lane_detection_model()
        self.object_detection_model = self.build_object_detection_model()
        self.current_lane = None
        self.safe_distance = 30  # Safe distance in meters
        self.lane_change_safe = False
        
    def build_lane_detection_model(self):
        """Build a CNN model for lane detection"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(720, 1280, 3)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(3, activation='softmax')  # Left lane, center lane, right lane
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def build_object_detection_model(self):
        """Build a CNN model for object detection (cars)"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(720, 1280, 3)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(4, activation='sigmoid')  # [x, y, width, height] for bounding box
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        return model
    
    def detect_lanes(self, frame):
        """Detect lanes in the input frame"""
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blur, 50, 150)
        
        # Define region of interest (ROI)
        height, width = edges.shape
        mask = np.zeros_like(edges)
        polygon = np.array([[
            (0, height), 
            (width, height), 
            (width // 2 + 200, height // 2),
            (width // 2 - 200, height // 2)
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # Apply Hough transform to detect lines
        lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        
        # Separate left and right lane lines
        left_lines = []
        right_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Adding small value to avoid division by zero
                
                # Filter out horizontal lines
                if abs(slope) < 0.1:
                    continue
                    
                if slope < 0:  # Negative slope for left lane
                    left_lines.append(line[0])
                else:  # Positive slope for right lane
                    right_lines.append(line[0])
        
        # Create line image
        line_image = np.zeros_like(frame)
        
        # Draw left lane
        if left_lines:
            left_line = self.average_lines(left_lines, frame)
            if left_line is not None:
                cv2.line(line_image, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 0, 255), 5)
        
        # Draw right lane
        if right_lines:
            right_line = self.average_lines(right_lines, frame)
            if right_line is not None:
                cv2.line(line_image, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 0, 255), 5)
        
        # Determine current lane
        if left_lines and right_lines:
            # Car is in center lane
            self.current_lane = 'center'
        elif left_lines and not right_lines:
            # Car is in right lane
            self.current_lane = 'right'
        elif not left_lines and right_lines:
            # Car is in left lane
            self.current_lane = 'left'
        else:
            # Cannot determine lane
            self.current_lane = 'unknown'
            
        # Blend the lines with the original frame
        result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
        
        # Add current lane information
        cv2.putText(result, f"Current Lane: {self.current_lane}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        return result
    
    def average_lines(self, lines, frame):
        """Calculate average line from multiple detected lines"""
        if not lines:
            return None
            
        x_coords = []
        y_coords = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
            
        if not x_coords or not y_coords:
            return None
            
        # Calculate line equation: y = mx + b
        A = np.vstack([x_coords, np.ones(len(x_coords))]).T
        m, b = np.linalg.lstsq(A, y_coords, rcond=None)[0]
        
        # Get start and end points for the line
        height, width, _ = frame.shape
        y1 = height
        y2 = int(height * 0.6)  # Draw up to 60% of the frame height
        x1 = int((y1 - b) / m)
        x2 = int((y2 - b) / m)
        
        # Ensure points are within frame
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        
        return [x1, y1, x2, y2]
    
    def detect_vehicles(self, frame):
        """Detect vehicles in the input frame"""
        # For demonstration, we'll use a simple color-based detection
        # In a real application, you would use the trained object detection model
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color range for vehicles (this is a simplified approach)
        lower_bound = np.array([0, 0, 100])
        upper_bound = np.array([180, 30, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        vehicles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                vehicles.append((x, y, w, h))
                
        return vehicles
    
    def calculate_distance(self, vehicle_box):
        """Calculate approximate distance to vehicle (simplified)"""
        # In a real application, you would use camera parameters and vehicle size
        # For demonstration, we'll use the inverse of bounding box height as a proxy for distance
        _, _, _, h = vehicle_box
        # Assuming the higher the box, the closer the vehicle
        distance = 100000 / (h + 1)  # Avoid division by zero
        return distance
    
    def check_lane_change_safety(self, frame, vehicles):
        """Check if it's safe to change lanes"""
        # Determine which lane we want to change to
        target_lane = 'right' if self.current_lane == 'left' or self.current_lane == 'center' else 'left'
        
        # Check for vehicles in the target lane
        for vehicle in vehicles:
            x, y, w, h = vehicle
            vehicle_x_center = x + w/2
            frame_width = frame.shape[1]
            
            # Determine which lane the vehicle is in (simplified)
            vehicle_lane = 'left' if vehicle_x_center < frame_width/3 else ('right' if vehicle_x_center > 2*frame_width/3 else 'center')
            
            # If vehicle is in target lane and close, it's not safe to change lanes
            if vehicle_lane == target_lane:
                distance = self.calculate_distance(vehicle)
                if distance < self.safe_distance:
                    self.lane_change_safe = False
                    return
        
        # If we didn't find any close vehicles in the target lane
        self.lane_change_safe = True
    
    def process_frame(self, frame):
        """Process a single frame"""
        # Detect lanes
        lane_result = self.detect_lanes(frame)
        
        # Detect vehicles
        vehicles = self.detect_vehicles(frame)
        
        # Draw bounding boxes around detected vehicles
        for vehicle in vehicles:
            x, y, w, h = vehicle
            cv2.rectangle(lane_result, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Calculate and display distance
            distance = self.calculate_distance(vehicle)
            cv2.putText(lane_result, f"{distance:.1f}m", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Check if lane change is safe
        self.check_lane_change_safety(frame, vehicles)
        
        # Display lane change recommendation
        if self.current_lane != 'unknown':
            if self.lane_change_safe:
                message = f"Safe to change to {'right' if self.current_lane == 'left' or self.current_lane == 'center' else 'left'} lane"
                color = (0, 255, 0)  # Green
            else:
                message = "Not safe to change lanes"
                color = (0, 0, 255)  # Red
                
            cv2.putText(lane_result, message, (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
        return lane_result
    
    def process_video(self, input_path, output_path=None):
        """Process a video file"""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create video writer if output path is specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process the frame
            result_frame = self.process_frame(frame)
            
            # Write to output video if specified
            if writer:
                writer.write(result_frame)
            
            # Display the result
            cv2.imshow('Lane Detection', result_frame)
            
            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

# Function to train the models (would require labeled data)
def train_models(lane_detection_system, lane_data_path, vehicle_data_path):
    """Train the lane detection and object detection models"""
    # This would load training data and train the models
    # For demonstration purposes, we're skipping actual training
    print("Training models (simulated)...")
    time.sleep(2)  # Simulate training time
    print("Models trained successfully!")

# Example usage
if __name__ == "__main__":
    system = LaneDetectionSystem()
    # Uncomment these lines to train the models (with actual data)
    # train_models(system, 'path/to/lane/data', 'path/to/vehicle/data')
    
    # Process a video file
    system.process_video('input_video.mp4', 'output_video.mp4')