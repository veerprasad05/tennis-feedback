from ultralytics import YOLO
import logging
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import gc
import traceback

# Suppress YOLO logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Constants
PADDING = 200
CROP_SIZE = 640

class TennisStrokeDetector:
    def __init__(self, sequence_length=30):  # Analyze last 30 frames (1 second at 30fps)
        self.sequence_length = sequence_length
        
        # Store sequences of data
        self.pose_history = deque(maxlen=sequence_length)
        self.racket_history = deque(maxlen=sequence_length)
        self.ball_history = deque(maxlen=sequence_length)
        self.frame_numbers = deque(maxlen=sequence_length)
        
        # Current stroke state
        self.current_stroke = None
        self.stroke_start_frame = None
        self.stroke_phase = "idle"  # idle, preparation, execution, follow_through
        
        # Completed strokes
        self.detected_strokes = []
        
    def add_frame_data(self, frame_num, pose_landmarks, racket_bbox, ball_detections):
        """Add new frame data to the sequence"""
        self.frame_numbers.append(frame_num)
        self.pose_history.append(self.extract_pose_features(pose_landmarks))
        self.racket_history.append(self.extract_racket_features(racket_bbox))
        self.ball_history.append(self.extract_ball_features(ball_detections))
        
        # Analyze if we have enough frames
        if len(self.pose_history) >= 10:  # Need minimum frames to analyze
            self.analyze_stroke_sequence()
    
    def extract_pose_features(self, pose_landmarks):
        """Extract key pose features from MediaPipe landmarks"""
        if not pose_landmarks:
            return None
            
        features = {}
        
        # Key landmarks (MediaPipe indices)
        landmark_indices = {
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24
        }
        
        for name, idx in landmark_indices.items():
            if idx < len(pose_landmarks.landmark):
                landmark = pose_landmarks.landmark[idx]
                features[name] = {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z if hasattr(landmark, 'z') else 0,
                    'visibility': landmark.visibility if hasattr(landmark, 'visibility') else 1
                }
        
        # Calculate derived features
        if 'right_shoulder' in features and 'left_shoulder' in features:
            features['shoulder_angle'] = self.calculate_angle(
                features['left_shoulder'], features['right_shoulder']
            )
        
        if 'right_wrist' in features and 'right_elbow' in features and 'right_shoulder' in features:
            features['right_arm_angle'] = self.calculate_arm_angle(
                features['right_shoulder'], features['right_elbow'], features['right_wrist']
            )
            
        return features
    
    def extract_racket_features(self, racket_bbox):
        """Extract racket position and movement features"""
        if not racket_bbox:
            return None
            
        x1, y1, x2, y2 = racket_bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        return {
            'center_x': center_x,
            'center_y': center_y,
            'width': x2 - x1,
            'height': y2 - y1,
            'area': (x2 - x1) * (y2 - y1)
        }
    
    def extract_ball_features(self, ball_detections):
        """Extract ball position features"""
        if not ball_detections:
            return None
            
        # Use the highest confidence ball detection
        best_ball = max(ball_detections, key=lambda x: x['weighted_confidence'])
        x1, y1, x2, y2 = best_ball['bbox']
        
        return {
            'center_x': (x1 + x2) / 2,
            'center_y': (y1 + y2) / 2,
            'confidence': best_ball['weighted_confidence']
        }
    
    def calculate_angle(self, point1, point2):
        """Calculate angle between two points"""
        dx = point2['x'] - point1['x']
        dy = point2['y'] - point1['y']
        return np.arctan2(dy, dx) * 180 / np.pi
    
    def calculate_arm_angle(self, shoulder, elbow, wrist):
        """Calculate arm angle at elbow joint"""
        # Vector from elbow to shoulder
        v1 = np.array([shoulder['x'] - elbow['x'], shoulder['y'] - elbow['y']])
        # Vector from elbow to wrist
        v2 = np.array([wrist['x'] - elbow['x'], wrist['y'] - elbow['y']])
        
        # Calculate angle between vectors
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
        return angle
    
    def analyze_stroke_sequence(self):
        """Main function to analyze the current sequence for stroke patterns"""
        if len(self.pose_history) < 10:
            return
            
        # Check for stroke initiation
        if self.stroke_phase == "idle":
            if self.detect_stroke_preparation():
                self.stroke_phase = "preparation"
                self.stroke_start_frame = self.frame_numbers[-10]  # Start 10 frames ago
                print(f"Stroke preparation detected at frame {self.stroke_start_frame}")
        
        # Check for stroke execution
        elif self.stroke_phase == "preparation":
            stroke_type = self.classify_stroke_type()
            if stroke_type:
                self.current_stroke = stroke_type
                self.stroke_phase = "execution"
                print(f"{stroke_type} execution detected")
        
        # Check for stroke completion
        elif self.stroke_phase == "execution":
            if self.detect_stroke_completion():
                self.complete_stroke()
    
    def detect_stroke_preparation(self):
        """Detect if player is preparing for a stroke"""
        recent_poses = list(self.pose_history)[-10:]
        recent_rackets = list(self.racket_history)[-10:]
        
        # Remove None values
        valid_poses = [p for p in recent_poses if p is not None]
        valid_rackets = [r for r in recent_rackets if r is not None]
        
        if len(valid_poses) < 5 or len(valid_rackets) < 3:
            return False
        
        # Check for racket movement (preparation usually involves racket moving back)
        racket_movement = self.analyze_racket_movement(valid_rackets)
        
        # Check for body rotation
        body_rotation = self.analyze_body_rotation(valid_poses)
        
        # Check for weight shift
        weight_shift = self.analyze_weight_shift(valid_poses)
        
        # Preparation detected if multiple indicators are present
        indicators = [racket_movement, body_rotation, weight_shift]
        return sum(indicators) >= 2
    
    def classify_stroke_type(self):
        """Classify the type of stroke being performed"""
        recent_poses = list(self.pose_history)[-15:]
        recent_rackets = list(self.racket_history)[-15:]
        recent_balls = list(self.ball_history)[-15:]
        
        valid_poses = [p for p in recent_poses if p is not None]
        valid_rackets = [r for r in recent_rackets if r is not None]
        valid_balls = [b for b in recent_balls if b is not None]
        
        if len(valid_poses) < 8:
            return None
        
        # Check for serve (ball toss + overhead motion)
        if self.detect_serve_pattern(valid_poses, valid_balls):
            return "serve"
        
        # Check for forehand vs backhand
        if self.detect_forehand_pattern(valid_poses, valid_rackets):
            return "forehand"
        elif self.detect_backhand_pattern(valid_poses, valid_rackets):
            return "backhand"
        
        return None
    
    def detect_serve_pattern(self, poses, balls):
        """Detect serve pattern (ball toss + overhead swing)"""
        # Check for ball toss (ball moving upward)
        if len(balls) >= 5:
            ball_heights = [b['center_y'] for b in balls[-5:]]
            if len(ball_heights) >= 3:
                # Ball should move up then down
                if ball_heights[0] > ball_heights[1] and ball_heights[1] < ball_heights[2]:
                    return True
        
        # Check for overhead arm motion
        if len(poses) >= 5:
            right_wrist_heights = []
            for pose in poses[-5:]:
                if pose and 'right_wrist' in pose:
                    right_wrist_heights.append(pose['right_wrist']['y'])
            
            if len(right_wrist_heights) >= 3:
                # Wrist should move up (lower y values in image coordinates)
                if right_wrist_heights[-1] < right_wrist_heights[0] - 0.1:
                    return True
        
        return False
    
    def detect_forehand_pattern(self, poses, rackets):
        """Detect forehand pattern (right-handed player)"""
        if len(poses) < 5:
            return False
        
        # Check shoulder rotation (right shoulder should move forward)
        shoulder_rotations = []
        for pose in poses[-5:]:
            if pose and 'shoulder_angle' in pose:
                shoulder_rotations.append(pose['shoulder_angle'])
        
        if len(shoulder_rotations) >= 3:
            # Shoulder angle should change significantly
            angle_change = abs(shoulder_rotations[-1] - shoulder_rotations[0])
            if angle_change > 15:  # Threshold for significant rotation
                # Check if right wrist moves across body (left to right)
                if len(rackets) >= 3:
                    racket_x_positions = [r['center_x'] for r in rackets[-3:]]
                    if racket_x_positions[-1] > racket_x_positions[0]:
                        return True
        
        return False
    
    def detect_backhand_pattern(self, poses, rackets):
        """Detect backhand pattern"""
        if len(poses) < 5:
            return False
        
        # For backhand, left shoulder typically leads (right-handed player)
        # Check for different shoulder rotation pattern than forehand
        
        if len(rackets) >= 3:
            racket_x_positions = [r['center_x'] for r in rackets[-3:]]
            # Backhand typically moves from right to left
            if racket_x_positions[-1] < racket_x_positions[0]:
                return True
        
        return False
    
    def detect_stroke_completion(self):
        """Detect when stroke is completed (follow-through phase)"""
        recent_poses = list(self.pose_history)[-10:]
        recent_rackets = list(self.racket_history)[-10:]
        
        valid_poses = [p for p in recent_poses if p is not None]
        valid_rackets = [r for r in recent_rackets if r is not None]
        
        if len(valid_rackets) < 5:
            return False
        
        # Check if racket movement has slowed down (end of follow-through)
        racket_speeds = []
        for i in range(1, len(valid_rackets)):
            dx = valid_rackets[i]['center_x'] - valid_rackets[i-1]['center_x']
            dy = valid_rackets[i]['center_y'] - valid_rackets[i-1]['center_y']
            speed = np.sqrt(dx*dx + dy*dy)
            racket_speeds.append(speed)
        
        if len(racket_speeds) >= 3:
            # If speed has decreased significantly, stroke is likely complete
            recent_speed = np.mean(racket_speeds[-3:])
            earlier_speed = np.mean(racket_speeds[:3]) if len(racket_speeds) >= 6 else recent_speed
            
            if recent_speed < earlier_speed * 0.5:  # Speed reduced by 50%
                return True
        
        return False
    
    def complete_stroke(self):
        """Complete the current stroke and record it"""
        end_frame = self.frame_numbers[-1]
        
        stroke_info = {
            'type': self.current_stroke,
            'start_frame': self.stroke_start_frame,
            'end_frame': end_frame,
            'duration_frames': end_frame - self.stroke_start_frame
        }
        
        self.detected_strokes.append(stroke_info)
        print(f"Completed {self.current_stroke}: frames {self.stroke_start_frame}-{end_frame}")
        
        # Reset for next stroke
        self.stroke_phase = "idle"
        self.current_stroke = None
        self.stroke_start_frame = None
    
    def analyze_racket_movement(self, rackets):
        """Analyze racket movement pattern"""
        if len(rackets) < 3:
            return False
        
        # Calculate movement vector
        start_pos = (rackets[0]['center_x'], rackets[0]['center_y'])
        end_pos = (rackets[-1]['center_x'], rackets[-1]['center_y'])
        
        movement_distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        
        return movement_distance > 50  # Threshold for significant movement
    
    def analyze_body_rotation(self, poses):
        """Analyze body rotation"""
        if len(poses) < 3:
            return False
        
        shoulder_angles = []
        for pose in poses:
            if 'shoulder_angle' in pose:
                shoulder_angles.append(pose['shoulder_angle'])
        
        if len(shoulder_angles) >= 3:
            angle_change = abs(shoulder_angles[-1] - shoulder_angles[0])
            return angle_change > 10  # Threshold for rotation
        
        return False
    
    def analyze_weight_shift(self, poses):
        """Analyze weight shift through hip movement"""
        if len(poses) < 3:
            return False
        
        hip_centers = []
        for pose in poses:
            if 'left_hip' in pose and 'right_hip' in pose:
                center_x = (pose['left_hip']['x'] + pose['right_hip']['x']) / 2
                hip_centers.append(center_x)
        
        if len(hip_centers) >= 3:
            hip_movement = abs(hip_centers[-1] - hip_centers[0])
            return hip_movement > 0.05  # Threshold for weight shift
        
        return False
    
    def get_stroke_summary(self):
        """Get summary of all detected strokes"""
        return {
            'total_strokes': len(self.detected_strokes),
            'strokes': self.detected_strokes,
            'stroke_types': {
                'forehand': len([s for s in self.detected_strokes if s['type'] == 'forehand']),
                'backhand': len([s for s in self.detected_strokes if s['type'] == 'backhand']),
                'serve': len([s for s in self.detected_strokes if s['type'] == 'serve'])
            }
        }

class TennisBallDetector:
    def __init__(self, custom_model_path=None):
        # Load models
        self.yolo_model = YOLO('yolov8s.pt')
        self.custom_model = YOLO(custom_model_path) if custom_model_path else None
        
        # Tennis ball color ranges (HSV)
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([40, 255, 255])
        
        # Detection parameters
        self.min_area = 50
        self.max_area = 2000
        self.min_circularity = 0.6
        
    def detect_by_color(self, frame):
        """Detect tennis balls using color and shape analysis"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        
        # Morphological operations to clean up the mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > self.min_circularity:
                        x, y, w, h = cv2.boundingRect(contour)
                        # Check aspect ratio (should be close to square)
                        aspect_ratio = w / h if h > 0 else 0
                        if 0.7 < aspect_ratio < 1.3:
                            candidates.append({
                                'bbox': (x, y, x+w, y+h),
                                'confidence': circularity,
                                'method': 'color'
                            })
        
        return candidates
    
    def detect_by_yolo(self, frame):
        """Detect tennis balls using YOLO sports ball class"""
        try:
            results = self.yolo_model(frame, conf=0.2, iou=0.5)
            
            candidates = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    if int(box.cls[0]) == 32:  # Sports ball class
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        candidates.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': confidence,
                            'method': 'yolo'
                        })
            
            return candidates
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return []
    
    def detect_by_custom_model(self, frame):
        """Detect tennis balls using custom trained model"""
        if self.custom_model is None:
            return []
            
        try:
            results = self.custom_model(frame, conf=0.3)
            
            candidates = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    candidates.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'method': 'custom'
                    })
            
            return candidates
        except Exception as e:
            print(f"Custom model detection error: {e}")
            return []
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def merge_detections(self, all_candidates, iou_threshold=0.3):
        """Merge overlapping detections from different methods"""
        if not all_candidates:
            return []
        
        # Sort by confidence
        all_candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        merged = []
        for candidate in all_candidates:
            overlaps = False
            for existing in merged:
                if self.calculate_iou(candidate['bbox'], existing['bbox']) > iou_threshold:
                    # If overlap, keep the one with higher confidence
                    if candidate['confidence'] > existing['confidence']:
                        merged.remove(existing)
                        merged.append(candidate)
                    overlaps = True
                    break
            
            if not overlaps:
                merged.append(candidate)
        
        return merged
    
    def detect_tennis_balls(self, frame):
        """Main detection method combining all approaches"""
        all_candidates = []
        
        # Method 1: Color-based detection
        color_candidates = self.detect_by_color(frame)
        all_candidates.extend(color_candidates)
        
        # Method 2: YOLO sports ball detection
        yolo_candidates = self.detect_by_yolo(frame)
        all_candidates.extend(yolo_candidates)
        
        # Method 3: Custom model (if available)
        if self.custom_model:
            custom_candidates = self.detect_by_custom_model(frame)
            all_candidates.extend(custom_candidates)
        
        # Merge overlapping detections
        final_detections = self.merge_detections(all_candidates)
        
        # Filter by confidence (weighted by method reliability)
        filtered_detections = []
        for detection in final_detections:
            method_weight = {
                'custom': 1.0,    # Trust custom model most
                'yolo': 0.8,      # YOLO is quite reliable
                'color': 0.6      # Color detection as backup
            }
            
            weighted_confidence = detection['confidence'] * method_weight[detection['method']]
            if weighted_confidence > 0.3:  # Adjust threshold as needed
                detection['weighted_confidence'] = weighted_confidence
                filtered_detections.append(detection)
        
        return filtered_detections

def crop_and_scale_position(crop_height, crop_width, x1, y1, x2, y2, x1_padded, y1_padded, scale_x, scale_y):
    x1_crop = max(0, x1 - x1_padded)
    y1_crop = max(0, y1 - y1_padded)
    x2_crop = min(crop_width, x2 - x1_padded)
    y2_crop = min(crop_height, y2 - y1_padded)
    
    x1_scaled = int(x1_crop * scale_x)
    y1_scaled = int(y1_crop * scale_y)
    x2_scaled = int(x2_crop * scale_x)
    y2_scaled = int(y2_crop * scale_y)

    return x1_scaled, y1_scaled, x2_scaled, y2_scaled

def main():
    # Initialize resources
    model = None
    cap = None
    ball_detector = None
    
    try:
        print("Loading YOLO model...")
        # Load YOLO model for person and racket detection
        model = YOLO('yolov8s.pt')
        
        print("Initializing tennis ball detector...")
        # Initialize the tennis ball detector with your custom model
        # Change 'best.pt' to your actual model path, or set to None if you don't have one
        ball_detector = TennisBallDetector(custom_model_path='best.pt')
        
        print("Initializing stroke detector...")
        stroke_detector = TennisStrokeDetector(sequence_length=30)
        
        print("Opening video...")
        # Open video - change this to your video path
        cap = cv2.VideoCapture("shots-dataset/test.mp4")
        if not cap.isOpened():
            # Try alternative video path
            cap = cv2.VideoCapture("test.mp4")
            if not cap.isOpened():
                raise ValueError("Could not open video file. Check the path!")
        
        print("Loading MediaPipe...")
        # Load mediapipe modules
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        
        frame_count = 0
        
        print("Starting video processing...")
        
        # Use MediaPipe context manager for proper cleanup
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("End of video or failed to read frame")
                    break
                
                frame_count += 1
                
                try:
                    # Process frame
                    height, width, _ = frame.shape
                    results = model(frame, verbose=False)[0]  # Add verbose=False to reduce output
                    
                    # Use the improved ball detection
                    tennis_balls = ball_detector.detect_tennis_balls(frame)
                    
                    # Check if results.boxes exists and is not None
                    if results.boxes is None or len(results.boxes) == 0:
                        if frame_count % 30 == 0:  # Print less frequently
                            print(f"Frame {frame_count}: No detections")
                        continue
                    
                    # Filter detections by class (person and racket only)
                    person_boxes = [box for box in results.boxes if int(box.cls[0]) == 0]
                    racket_boxes = [box for box in results.boxes if int(box.cls[0]) == 38]
                    
                    if not person_boxes:
                        if frame_count % 30 == 0:  # Print less frequently
                            print(f"Frame {frame_count}: No person detected")
                        continue
                    
                    # Choose the person with the largest area
                    largest_person = max(person_boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
                    x1, y1, x2, y2 = map(int, largest_person.xyxy[0])
                    
                    # Add padding (but clamp to image boundaries)
                    x1_padded = max(0, x1 - PADDING)
                    y1_padded = max(0, y1 - PADDING)
                    x2_padded = min(width, x2 + PADDING)
                    y2_padded = min(height, y2 + PADDING)
                    
                    # Validate crop dimensions
                    if y2_padded <= y1_padded or x2_padded <= x1_padded:
                        print(f"Frame {frame_count}: Invalid crop dimensions")
                        continue
                    
                    # Crop the player with padding
                    player_crop = frame[y1_padded:y2_padded, x1_padded:x2_padded].copy()
                    
                    # Validate crop is not empty
                    if player_crop.size == 0:
                        print(f"Frame {frame_count}: Empty crop")
                        continue
                    
                    # Get original crop dimensions
                    crop_height, crop_width = player_crop.shape[:2]
                    
                    # Resize to 640x640 pixels
                    player_crop_resized = cv2.resize(player_crop, (CROP_SIZE, CROP_SIZE))
                    
                    # MediaPipe pose detection
                    player_crop_resized_rgb = cv2.cvtColor(player_crop_resized, cv2.COLOR_BGR2RGB)
                    pose_results = pose.process(player_crop_resized_rgb)
                    
                    if pose_results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            player_crop_resized, 
                            pose_results.pose_landmarks, 
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                        )
                        
                        # Add stroke detection
                        racket_bbox = None
                        if racket_boxes:
                            largest_racket = max(racket_boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
                            racket_bbox = list(map(int, largest_racket.xyxy[0]))
                        
                        stroke_detector.add_frame_data(
                            frame_count, 
                            pose_results.pose_landmarks, 
                            racket_bbox, 
                            tennis_balls
                        )
                        
                        # Display current stroke info
                        if stroke_detector.stroke_phase != "idle":
                            cv2.putText(player_crop_resized, 
                                    f"Stroke: {stroke_detector.current_stroke or 'detecting...'} ({stroke_detector.stroke_phase})", 
                                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                    # Calculate scaling factors
                    scale_x = CROP_SIZE / crop_width
                    scale_y = CROP_SIZE / crop_height
                    
                    # Draw player box on resized crop
                    player_x1_scaled, player_y1_scaled, player_x2_scaled, player_y2_scaled = crop_and_scale_position(
                        crop_height, crop_width, x1, y1, x2, y2, x1_padded, y1_padded, scale_x, scale_y
                    )
                    
                    cv2.rectangle(player_crop_resized, (player_x1_scaled, player_y1_scaled), 
                                (player_x2_scaled, player_y2_scaled), (0, 255, 0), 2)
                    cv2.putText(player_crop_resized, "Player", (player_x1_scaled, player_y1_scaled - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Process racket detection
                    if racket_boxes:
                        largest_racket = max(racket_boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
                        x1r, y1r, x2r, y2r = map(int, largest_racket.xyxy[0])
                        
                        if (x1r < x2_padded and x2r > x1_padded and y1r < y2_padded and y2r > y1_padded):
                            racket_x1_scaled, racket_y1_scaled, racket_x2_scaled, racket_y2_scaled = crop_and_scale_position(
                                crop_height, crop_width, x1r, y1r, x2r, y2r, x1_padded, y1_padded, scale_x, scale_y
                            )
                            
                            cv2.rectangle(player_crop_resized, (racket_x1_scaled, racket_y1_scaled), 
                                        (racket_x2_scaled, racket_y2_scaled), (0, 0, 255), 2)
                            cv2.putText(player_crop_resized, "Racket", (racket_x1_scaled, racket_y1_scaled - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # Process improved ball detection results
                    for ball in tennis_balls:
                        x1b, y1b, x2b, y2b = ball['bbox']
                        confidence = ball['weighted_confidence']
                        method = ball['method']
                        
                        # Check if ball is within the crop area
                        if (x1b < x2_padded and x2b > x1_padded and y1b < y2_padded and y2b > y1_padded):
                            ball_x1_scaled, ball_y1_scaled, ball_x2_scaled, ball_y2_scaled = crop_and_scale_position(
                                crop_height, crop_width, x1b, y1b, x2b, y2b, x1_padded, y1_padded, scale_x, scale_y
                            )
                            
                            # Different colors based on detection method
                            method_colors = {
                                'custom': (255, 0, 0),    # Blue for custom model
                                'yolo': (0, 255, 255),    # Yellow for YOLO
                                'color': (255, 0, 255)    # Magenta for color detection
                            }
                            color = method_colors.get(method, (255, 0, 0))
                            
                            cv2.rectangle(player_crop_resized, (ball_x1_scaled, ball_y1_scaled), 
                                        (ball_x2_scaled, ball_y2_scaled), color, 2)
                            cv2.putText(player_crop_resized, f"Ball ({method}): {confidence:.2f}", 
                                      (ball_x1_scaled, ball_y1_scaled - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
                    # Add detection statistics overlay
                    stats_text = f"Frame: {frame_count} | Balls detected: {len(tennis_balls)}"
                    cv2.putText(player_crop_resized, stats_text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Show detection methods used
                    if tennis_balls:
                        methods_used = list(set([ball['method'] for ball in tennis_balls]))
                        methods_text = f"Methods: {', '.join(methods_used)}"
                        cv2.putText(player_crop_resized, methods_text, (10, 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    # Display the result
                    cv2.imshow("Player Crop (640x640)", player_crop_resized)
                    
                    # Check for exit
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Quit requested by user")
                        break
                    
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    # Don't print full traceback for every frame error
                    continue
                
                finally:
                    # Clean up frame-specific variables
                    locals_to_clean = ['results', 'tennis_balls', 'player_crop', 'player_crop_resized', 
                                     'player_crop_resized_rgb', 'pose_results']
                    for var_name in locals_to_clean:
                        if var_name in locals():
                            del locals()[var_name]
                    
                    # Force garbage collection every 100 frames
                    if frame_count % 100 == 0:
                        gc.collect()
                        print(f"Processed {frame_count} frames, garbage collected")
        
        # At the end, print stroke summary:
        stroke_summary = stroke_detector.get_stroke_summary()
        print("Stroke Analysis Summary:")
        print(f"Total strokes detected: {stroke_summary['total_strokes']}")
        for stroke in stroke_summary['strokes']:
            print(f"{stroke['type']}: frames {stroke['start_frame']}-{stroke['end_frame']}")
    
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
    
    finally:
        # Ensure cleanup happens
        print("Cleaning up resources...")
        
        if cap is not None:
            cap.release()
            print("Video capture released")
        
        cv2.destroyAllWindows()
        print("OpenCV windows destroyed")
        
        # Final garbage collection
        gc.collect()
        print("Final cleanup completed")

if __name__ == "__main__":
    main()