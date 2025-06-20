from collections import deque
import numpy as np

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