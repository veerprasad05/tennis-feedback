from collections import deque
import numpy as np
import cv2

class ReadyPositionDetector:
    def __init__(self, sequence_length=60):  # Store last 60 frames (2 seconds at 30fps)
        self.sequence_length = sequence_length
        
        # Store sequences of data
        self.pose_history = deque(maxlen=sequence_length)
        self.frame_numbers = deque(maxlen=sequence_length)
        
        # Ready position detection
        self.valid_ready_positions = []  # List of valid ready position sequences
        self.raw_ready_frames = []  # All raw ready position frames for debugging
        
        # Ready position validation tracking
        self.current_ready_sequence = []  # Current sequence of consecutive ready position frames
        
    def add_frame_data(self, frame_num, pose_landmarks):
        """Add new frame data and detect ready position"""
        self.frame_numbers.append(frame_num)
        
        # Extract pose features
        pose_features = self.extract_pose_features(pose_landmarks)
        
        # Store in history
        self.pose_history.append(pose_features)
        
        # Check for ready position
        is_ready_position = self.detect_ready_position(pose_features)
        
        if is_ready_position:
            # Add to current ready position sequence
            ready_info = {
                'frame': frame_num,
                'pose_features': pose_features
            }
            self.current_ready_sequence.append(ready_info)
            self.raw_ready_frames.append(frame_num)
        else:
            # No ready position - check if we should finalize current sequence
            if self.current_ready_sequence:
                self.finalize_ready_sequence()
    
    def finalize_ready_sequence(self):
        """Finalize current ready position sequence and validate it"""
        if not self.current_ready_sequence:
            return
        
        sequence_length = len(self.current_ready_sequence)
        
        # Validation rules:
        # 1. At least 2 consecutive frames
        # 2. If more than many frames, still counts as 1 ready position period
        if sequence_length >= 2:
            # This is a valid ready position period
            start_frame = self.current_ready_sequence[0]['frame']
            end_frame = self.current_ready_sequence[-1]['frame']
            
            # Use the middle frame as the representative ready position frame
            middle_index = len(self.current_ready_sequence) // 2
            representative_ready = self.current_ready_sequence[middle_index]
            
            valid_ready = {
                'ready_position_id': len(self.valid_ready_positions) + 1,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'duration_frames': sequence_length,
                'representative_frame': representative_ready['frame'],
                'pose_features': representative_ready['pose_features'],
                'all_ready_frames': [ready['frame'] for ready in self.current_ready_sequence]
            }
            
            self.valid_ready_positions.append(valid_ready)
            print(f"Valid ready position detected: frames {start_frame}-{end_frame} ({sequence_length} frames)")
        
        # Reset current sequence
        self.current_ready_sequence = []
    
    def extract_pose_features(self, pose_landmarks):
        """Extract key pose features from MediaPipe landmarks"""
        if not pose_landmarks:
            return None
            
        features = {}
        
        # Key landmarks (MediaPipe indices)
        landmark_indices = {
            'left_shoulder': 11, 'right_shoulder': 12,
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
        
        return features
    
    def detect_ready_position(self, pose_features):
        """Detect if player is in ready position (both wrists within extended body polygon)"""
        if not pose_features:
            return False
        
        # Check if all required landmarks are present and visible
        required_landmarks = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 
                            'left_wrist', 'right_wrist']
        
        for landmark_name in required_landmarks:
            if landmark_name not in pose_features:
                return False
        
        # Extract coordinates
        left_shoulder = (pose_features['left_shoulder']['x'], pose_features['left_shoulder']['y'])
        right_shoulder = (pose_features['right_shoulder']['x'], pose_features['right_shoulder']['y'])
        left_hip = (pose_features['left_hip']['x'], pose_features['left_hip']['y'])
        right_hip = (pose_features['right_hip']['x'], pose_features['right_hip']['y'])
        
        left_wrist = (pose_features['left_wrist']['x'], pose_features['left_wrist']['y'])
        right_wrist = (pose_features['right_wrist']['x'], pose_features['right_wrist']['y'])
        
        # Create extended polygon:
        # - Bottom: vertical lines from shoulders down to hip y-level
        # - Top: left_shoulder to right_shoulder (original positions)
        left_hip_y = left_hip[1]
        right_hip_y = right_hip[1]
        left_shoulder_x, left_shoulder_y = left_shoulder
        right_shoulder_x, right_shoulder_y = right_shoulder
        
        # Create the four corners of the extended polygon
        # Bottom left: left shoulder x-coordinate, left hip y coordinate
        # Bottom right: right shoulder x-coordinate, right hip y coordinate  
        # Top right: right shoulder
        # Top left: left shoulder
        extended_body_polygon = np.array([
            (left_shoulder_x, left_hip_y),    # Top left: left shoulder x, left hip y
            (right_shoulder_x, right_hip_y),  # Top right: right shoulder x, right hip y
            (right_shoulder_x, right_shoulder_y),       # Bottom right: right shoulder
            (left_shoulder_x, left_shoulder_y)          # Bottom left: left shoulder
        ], dtype=np.float32)
        
        # Check if both wrists are inside the extended polygon
        left_wrist_inside = self.point_in_polygon(left_wrist, extended_body_polygon)
        right_wrist_inside = self.point_in_polygon(right_wrist, extended_body_polygon)
        
        # Ready position: both wrists are within the extended body polygon
        return left_wrist_inside and right_wrist_inside
    
    def point_in_polygon(self, point, polygon):
        """Check if a point is inside a polygon using OpenCV"""
        point = np.array(point, dtype=np.float32)
        
        # Use OpenCV's pointPolygonTest
        # Returns positive if inside, negative if outside, 0 if on the edge
        result = cv2.pointPolygonTest(polygon, tuple(point), False)
        
        return result >= 0  # Inside or on the edge
    
    def get_ready_position_summary(self):
        """Get summary of all detected ready positions"""
        return {
            'total_valid_ready_positions': len(self.valid_ready_positions),
            'valid_ready_positions': self.valid_ready_positions,
            'total_raw_ready_frames': len(self.raw_ready_frames),
            'raw_ready_frames': self.raw_ready_frames
        }
    
    def print_ready_position_results(self):
        """Print all ready position detection results"""
        # Finalize any remaining ready position sequence
        if self.current_ready_sequence:
            self.finalize_ready_sequence()
        
        if not self.valid_ready_positions:
            print("No valid ready positions detected.")
            print(f"Raw ready position frames detected: {len(self.raw_ready_frames)}")
            if self.raw_ready_frames:
                print(f"Raw frames: {self.raw_ready_frames}")
                print("(These didn't meet the minimum 2 consecutive frames requirement)")
            return
        
        print(f"\nDetected {len(self.valid_ready_positions)} valid ready position periods:")
        
        print("\nSummary:")
        for ready in self.valid_ready_positions:
            if ready['duration_frames'] == 1:
                print(f"Ready Position {ready['ready_position_id']}: Frame {ready['representative_frame']}")
            else:
                print(f"Ready Position {ready['ready_position_id']}: Frames {ready['start_frame']}-{ready['end_frame']} ({ready['duration_frames']} frames)")
        
        print("\nDetailed Ready Position Information:")
        for ready in self.valid_ready_positions:
            pose = ready['pose_features']
            print(f"Ready Position {ready['ready_position_id']}:")
            print(f"  Duration: {ready['start_frame']}-{ready['end_frame']} ({ready['duration_frames']} frames)")
            print(f"  Representative frame: {ready['representative_frame']}")
            
            if pose:
                # Show wrist positions relative to body
                left_wrist = pose.get('left_wrist', {})
                right_wrist = pose.get('right_wrist', {})
                print(f"  Left wrist: ({left_wrist.get('x', 'N/A'):.3f}, {left_wrist.get('y', 'N/A'):.3f})")
                print(f"  Right wrist: ({right_wrist.get('x', 'N/A'):.3f}, {right_wrist.get('y', 'N/A'):.3f})")
            
            print(f"  All ready position frames: {ready['all_ready_frames']}")
            print()
        
        print(f"Total raw ready position frames: {len(self.raw_ready_frames)}")
        if len(self.raw_ready_frames) > sum(rp['duration_frames'] for rp in self.valid_ready_positions):
            invalid_frames = len(self.raw_ready_frames) - sum(rp['duration_frames'] for rp in self.valid_ready_positions)
            print(f"Invalid single-frame ready positions filtered out: {invalid_frames}")
    
    def get_frame_ready_position_list(self):
        """Return simple list of representative frame numbers where valid ready position occurred"""
        # Finalize any remaining ready position sequence
        if self.current_ready_sequence:
            self.finalize_ready_sequence()
        
        return [ready['representative_frame'] for ready in self.valid_ready_positions]
    
    def get_all_valid_ready_position_frames(self):
        """Return all frame numbers from valid ready position sequences"""
        # Finalize any remaining ready position sequence
        if self.current_ready_sequence:
            self.finalize_ready_sequence()
        
        all_frames = []
        for ready in self.valid_ready_positions:
            all_frames.extend(ready['all_ready_frames'])
        return sorted(all_frames)