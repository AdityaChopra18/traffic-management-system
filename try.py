import cv2
import numpy as np
import time
import os
from datetime import datetime
import threading
import requests

class TrafficSignalController:
    """
    A class to control traffic signals when emergency vehicles are detected.
    """
    
    def __init__(self, intersection_id="INT001", api_endpoint="http://localhost:5000/traffic/control"):
        """
        Initialize the traffic signal controller.
        
        Parameters:
        -----------
        intersection_id : str
            Unique identifier for the intersection
        api_endpoint : str
            Endpoint for the traffic management system API
        """
        self.intersection_id = intersection_id
        self.api_endpoint = api_endpoint
        self.current_signal_state = "NORMAL"  # Can be NORMAL or EMERGENCY
        self.emergency_active = False
        self.emergency_start_time = None
        self.emergency_duration = 20  # Duration to keep green signal (seconds) - reduced for testing
        self.last_state_change = time.time()
        self.lock = threading.Lock()
        self.debug_mode = True  # Enable debug logging
    
    def activate_emergency_corridor(self, direction="NORTH"):
        """
        Activate emergency corridor by setting green signal for the specified direction.
        
        Parameters:
        -----------
        direction : str
            Direction of the emergency vehicle (NORTH, SOUTH, EAST, WEST)
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        with self.lock:
            if self.emergency_active:
                # Already in emergency mode, just update time
                self.emergency_start_time = time.time()
                return True
            
            try:
                # Construct payload for traffic management system
                payload = {
                    "intersection_id": self.intersection_id,
                    "command": "EMERGENCY",
                    "direction": direction,
                    "priority": "HIGH"
                }
                
                # Try to connect to the traffic management system API
                try:
                    response = requests.post(self.api_endpoint, json=payload, timeout=2)
                    if response.status_code == 200:
                        print(f"Successfully sent emergency signal command to direction: {direction}")
                    else:
                        print(f"Failed to send command. Status code: {response.status_code}")
                except requests.RequestException as e:
                    print(f"API connection error: {e}")
                    # Continue with local simulation even if API fails
                
                # Activate emergency mode
                self.emergency_active = True
                self.emergency_start_time = time.time()
                self.current_signal_state = "EMERGENCY"
                
                print(f"EMERGENCY MODE ACTIVATED - Green signal for {direction}")
                return True
                
            except Exception as e:
                print(f"Error activating emergency corridor: {e}")
                return False

    def manual_override_to_normal(self):
        """
        Manually override the traffic signal state back to normal operation
        regardless of emergency duration.
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        with self.lock:
            try:
                # Return to normal operation
                payload = {
                    "intersection_id": self.intersection_id,
                    "command": "NORMAL"
                }
                
                try:
                    response = requests.post(self.api_endpoint, json=payload, timeout=2)
                    if response.status_code == 200:
                        print("Successfully sent command to return to normal operation")
                    else:
                        print(f"Failed to send command. Status code: {response.status_code}")
                except requests.RequestException as e:
                    print(f"API connection error: {e}")
                
                self.emergency_active = False
                self.current_signal_state = "NORMAL"
                print("MANUAL OVERRIDE: NORMAL MODE RESTORED - Regular traffic signal pattern")
                return True
                
            except Exception as e:
                print(f"Error restoring normal operation: {e}")
                return False
    
    def check_and_update_state(self):
        """
        Check current state and update if necessary.
        Should be called periodically.
        
        Returns:
        --------
        str
            Current state of the traffic signal
        """
        with self.lock:
            current_time = time.time()
            
            if self.debug_mode and self.emergency_active:
                elapsed_time = current_time - self.emergency_start_time
                print(f"Checking state: Emergency active = {self.emergency_active}, " 
                      f"Time elapsed = {elapsed_time:.2f}s, " 
                      f"Timeout = {self.emergency_duration}s, "
                      f"Remaining = {self.emergency_duration - elapsed_time:.2f}s")
            
            # Check if emergency mode should be deactivated
            if self.emergency_active and \
               (current_time - self.emergency_start_time) > self.emergency_duration:
                
                try:
                    # Return to normal operation
                    payload = {
                        "intersection_id": self.intersection_id,
                        "command": "NORMAL"
                    }
                    
                    try:
                        response = requests.post(self.api_endpoint, json=payload, timeout=2)
                        if response.status_code == 200:
                            print("Successfully sent command to return to normal operation")
                        else:
                            print(f"Failed to send command. Status code: {response.status_code}")
                    except requests.RequestException as e:
                        print(f"API connection error: {e}")
                    
                    self.emergency_active = False
                    self.current_signal_state = "NORMAL"
                    print("AUTO TIMEOUT: NORMAL MODE RESTORED - Regular traffic signal pattern")
                    
                except Exception as e:
                    print(f"Error restoring normal operation: {e}")
            
            return self.current_signal_state
    
    def get_current_state(self):
        """
        Get the current state of the traffic signal.
        
        Returns:
        --------
        str
            Current state of the traffic signal
        """
        with self.lock:
            return self.current_signal_state


class EmergencyVehicleDetector:
    """
    A class for detecting emergency vehicles in video streams using computer vision techniques.
    
    This detector uses multiple methods to identify emergency vehicles:
    1. Color detection for emergency vehicle colors (red, blue)
    2. Flashing light pattern detection
    3. Vehicle type detection (using a pre-trained model)
    """
    
    def __init__(self, model_path="models/emergency_vehicle_model.weights", 
                 config_path="models/emergency_vehicle.cfg",
                 classes_path="models/emergency_vehicle.names",
                 intersection_id="INT001"):
        """
        Initialize the emergency vehicle detector.
        
        Parameters:
        -----------
        model_path : str
            Path to the pre-trained weights for the YOLO model
        config_path : str
            Path to the configuration file for the YOLO model
        classes_path : str
            Path to the file containing class names
        intersection_id : str
            Unique identifier for the intersection
        """
        self.net = None
        self.classes = []
        self.is_model_loaded = False
        
        # Try to load the YOLO model if files exist
        try:
            if os.path.exists(config_path) and os.path.exists(model_path):
                self.net = cv2.dnn.readNetFromDarknet(config_path, model_path)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.is_model_loaded = True
                
                # Load classes
                if os.path.exists(classes_path):
                    with open(classes_path, 'r') as f:
                        self.classes = [line.strip() for line in f.readlines()]
        except Exception as e:
            print(f"Warning: Could not load YOLO model: {e}")
            print("Falling back to color and light pattern detection only.")
        
        # Initialize parameters for light flash detection
        self.flash_history = []
        self.flash_threshold = 20  # Brightness threshold for a flash
        self.flash_time_window = 2  # Time window to detect flashing in seconds
        self.last_detection_time = None
        self.cooldown_period = 5  # Seconds between alert triggers
        
        # Color detection parameters (HSV ranges for red and blue)
        # Red has two ranges in HSV (wraps around 180)
        self.red_lower1 = np.array([0, 100, 100])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([160, 100, 100])
        self.red_upper2 = np.array([180, 255, 255])
        
        self.blue_lower = np.array([100, 150, 0])
        self.blue_upper = np.array([140, 255, 255])
        
        # Initialize traffic signal controller
        self.signal_controller = TrafficSignalController(intersection_id=intersection_id)
        
        # Direction detection variables
        self.direction_history = []
        self.direction_window = 10  # Number of frames to determine direction
        
        # Last known emergency vehicle positions for direction estimation
        self.last_positions = []
        self.position_history_size = 10
        
        # Time tracking for state checking
        self.last_state_check = time.time()
    
    def detect_emergency_colors(self, frame):
        """
        Detect emergency vehicle colors (red and blue) in the frame.
        
        Parameters:
        -----------
        frame : numpy.ndarray
            Input frame to analyze
            
        Returns:
        --------
        bool
            True if emergency colors are detected above threshold
        float
            Percentage of frame with emergency colors
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create masks for red and blue
        mask_red1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        mask_red2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        mask_blue = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
        
        # Combine masks
        mask_combined = cv2.bitwise_or(mask_red, mask_blue)
        
        # Calculate percentage of emergency colors
        total_pixels = frame.shape[0] * frame.shape[1]
        emergency_pixels = cv2.countNonZero(mask_combined)
        percentage = (emergency_pixels / total_pixels) * 100
        
        # Set a threshold for detection
        color_threshold = 1.0  # 1% of frame
        
        return percentage > color_threshold, percentage
    
    def detect_flashing_lights(self, frame, timestamp):
        """
        Detect flashing light patterns typical of emergency vehicles.
        
        Parameters:
        -----------
        frame : numpy.ndarray
            Input frame to analyze
        timestamp : float
            Current timestamp for temporal analysis
            
        Returns:
        --------
        bool
            True if flashing lights pattern is detected
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Threshold to find bright spots
        _, bright_regions = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Count bright pixels
        bright_pixel_count = cv2.countNonZero(bright_regions)
        total_pixels = gray.shape[0] * gray.shape[1]
        brightness_percentage = (bright_pixel_count / total_pixels) * 100
        
        # Store brightness with timestamp
        self.flash_history.append((timestamp, brightness_percentage))
        
        # Keep only recent history
        self.flash_history = [x for x in self.flash_history 
                             if timestamp - x[0] < self.flash_time_window]
        
        # Check for flashing pattern (alternating high/low brightness)
        if len(self.flash_history) < 4:
            return False
        
        # Calculate standard deviation of brightness values
        brightness_values = [x[1] for x in self.flash_history]
        std_dev = np.std(brightness_values)
        
        # High standard deviation suggests flashing (changing brightness)
        return std_dev > 5.0
    
    def detect_emergency_vehicles_yolo(self, frame):
        """
        Detect emergency vehicles using YOLO object detection.
        
        Parameters:
        -----------
        frame : numpy.ndarray
            Input frame to analyze
            
        Returns:
        --------
        bool
            True if emergency vehicles are detected
        list
            List of detected emergency vehicle bounding boxes
        """
        if not self.is_model_loaded:
            return False, []
        
        height, width = frame.shape[:2]
        
        # Create a blob from the image
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Get output layer names
        layer_names = self.net.getLayerNames()
        try:
            # OpenCV API changed over versions, handle different APIs
            try:
                output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
            except:
                output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        except:
            # Fallback if we can't get layers
            return False, []
        
        # Forward pass through the network
        outputs = self.net.forward(output_layers)
        
        # Process outputs
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5:  # Confidence threshold
                    # Scale bounding box coordinates back to original image size
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Non-maximum suppression to remove overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        # Check if any emergency vehicle classes are detected
        # Assuming emergency vehicle classes are indexed as [0, 1, 2] in the classes list
        emergency_classes = ["ambulance", "police_car", "fire_truck"]
        emergency_detected = False
        detected_boxes = []
        
        if len(indexes) > 0:
            for i in indexes.flatten():
                class_name = self.classes[class_ids[i]] if class_ids[i] < len(self.classes) else "unknown"
                if class_name in emergency_classes:
                    emergency_detected = True
                    detected_boxes.append(boxes[i])
        
        return emergency_detected, detected_boxes
    
    def estimate_direction(self, boxes, frame_shape):
        """
        Estimate the direction of the emergency vehicle based on position and movement.
        
        Parameters:
        -----------
        boxes : list
            List of bounding boxes of detected vehicles
        frame_shape : tuple
            Shape of the video frame (height, width)
            
        Returns:
        --------
        str
            Estimated direction (NORTH, SOUTH, EAST, WEST)
        """
        if not boxes:
            return "UNKNOWN"
        
        # Get frame dimensions
        height, width = frame_shape[:2]
        
        # For each detected emergency vehicle, find the center point
        centers = []
        for box in boxes:
            x, y, w, h = box
            center_x = x + w // 2
            center_y = y + h // 2
            centers.append((center_x, center_y))
        
        # Use the largest box as the main vehicle (usually closest)
        main_box = max(boxes, key=lambda box: box[2] * box[3])
        main_x, main_y, main_w, main_h = main_box
        main_center_x = main_x + main_w // 2
        main_center_y = main_y + main_h // 2
        
        # Add the main center to position history
        self.last_positions.append((main_center_x, main_center_y))
        
        # Keep only recent history
        if len(self.last_positions) > self.position_history_size:
            self.last_positions.pop(0)
        
        # If we have enough history, determine movement direction
        if len(self.last_positions) >= 2:
            # Get the oldest and newest positions
            oldest_x, oldest_y = self.last_positions[0]
            newest_x, newest_y = self.last_positions[-1]
            
            # Calculate movement vectors
            dx = newest_x - oldest_x
            dy = newest_y - oldest_y
            
            # Determine predominant direction of movement
            if abs(dx) > abs(dy):
                # Horizontal movement is greater
                if dx > 0:
                    return "EAST"
                else:
                    return "WEST"
            else:
                # Vertical movement is greater
                if dy > 0:
                    return "SOUTH"  # Note: y increases downward in images
                else:
                    return "NORTH"
        
        # If we don't have enough history, use position in frame
        # Divide the frame into quadrants and determine which quadrant the vehicle is in
        if main_center_x < width / 2:
            if main_center_y < height / 2:
                return "NORTH"
            else:
                return "SOUTH"
        else:
            if main_center_y < height / 2:
                return "EAST"
            else:
                return "WEST"
    
    def process_frame(self, frame):
        """
        Process a video frame to detect emergency vehicles using multiple detection methods.
        
        Parameters:
        -----------
        frame : numpy.ndarray
            Input frame from video stream
            
        Returns:
        --------
        frame : numpy.ndarray
            Processed frame with detection visualization
        is_emergency : bool
            True if an emergency vehicle is detected
        confidence : float
            Detection confidence level (0-100)
        """
        current_time = time.time()
        
        # Apply each detection method
        color_detected, color_percentage = self.detect_emergency_colors(frame)
        flash_detected = self.detect_flashing_lights(frame, current_time)
        yolo_detected, boxes = self.detect_emergency_vehicles_yolo(frame)
        
        # Combine detection results (weighted decision)
        color_weight = 0.3
        flash_weight = 0.3
        yolo_weight = 0.4
        
        # Calculate confidence score
        confidence = (color_percentage * color_weight * 100 / 20) + \
                     (flash_detected * flash_weight * 100) + \
                     (yolo_detected * yolo_weight * 100)
        
        # Threshold for positive detection
        is_emergency = confidence > 30
        
        # Forced state check every 2 seconds regardless of detections
        if current_time - self.last_state_check > 2:
            self.signal_controller.check_and_update_state()
            self.last_state_check = current_time
        
        # Get current traffic signal state
        signal_state = self.signal_controller.get_current_state()
        
        # If emergency vehicle detected, activate emergency corridor
        if is_emergency:
            # Determine direction of the emergency vehicle
            direction = self.estimate_direction(boxes, frame.shape)
            
            # Only activate if not already in emergency mode or if it's time for a new alert
            if (signal_state != "EMERGENCY") or \
               (self.last_detection_time is None) or \
               (current_time - self.last_detection_time > self.cooldown_period):
                
                # Update last detection time
                self.last_detection_time = current_time
                
                # Activate emergency corridor in the determined direction
                self.signal_controller.activate_emergency_corridor(direction)
                
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                     f"ALERT: Emergency vehicle detected from {direction}! "
                     f"Confidence: {confidence:.1f}%")
        
        # Draw YOLO detection boxes if available
        if yolo_detected:
            for box in boxes:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Emergency Vehicle", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add detection information overlay
        cv2.putText(frame, f"Emergency Vehicle: {'YES' if is_emergency else 'NO'}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if is_emergency else (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.1f}%", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Signal State: {signal_state}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 0, 255) if signal_state == "EMERGENCY" else (0, 255, 0), 2)
        
        # Add instructions for manual override
        remaining_time = 0
        if signal_state == "EMERGENCY" and self.signal_controller.emergency_start_time:
            elapsed = current_time - self.signal_controller.emergency_start_time
            remaining_time = max(0, self.signal_controller.emergency_duration - elapsed)
            cv2.putText(frame, f"Time remaining: {remaining_time:.1f}s", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(frame, "Press 'n' to return to NORMAL mode", 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame, is_emergency, confidence


def main():
    """
    Main function to run the emergency vehicle detection system.
    """
    # Initialize detector
    detector = EmergencyVehicleDetector(intersection_id="INT001")
    
    # Open video capture from camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Starting emergency vehicle detection...")
    print("Press 'q' to quit.")
    print("Press 'n' to manually return to NORMAL traffic signal state.")
    
    while True:
        # Read a frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Process the frame
        processed_frame, is_emergency, confidence = detector.process_frame(frame)
        
        # Display the frame
        cv2.imshow('Emergency Vehicle Detection', processed_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        # 'q' to quit
        if key == ord('q'):
            break
        # 'n' to manually switch to normal mode
        elif key == ord('n'):
            print("Manual override requested - switching to NORMAL mode")
            detector.signal_controller.manual_override_to_normal()
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()


# Run the detection system if script is executed directly
if __name__ == "__main__":
    main()