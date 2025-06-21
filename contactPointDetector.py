from collections import deque
import numpy as np

class ContactPointDetector:
    def __init__(self, sequence_length=60):  # Store last 60 frames (2 seconds at 30fps)
        self.sequence_length = sequence_length
        
        # Store sequences of data
        self.ball_history = deque(maxlen=sequence_length)
        self.racket_history = deque(maxlen=sequence_length)
        self.frame_numbers = deque(maxlen=sequence_length)
        
        # Contact detection
        self.valid_contact_points = []  # List of valid contact points (groups of consecutive frames)
        self.raw_contact_frames = []  # All raw contact frames for debugging
        
        # Contact validation tracking
        self.current_contact_sequence = []  # Current sequence of consecutive contact frames
        
        # Last known positions (for estimation when not detected)
        self.last_ball_position = None
        self.last_racket_position = None
        
    def add_frame_data(self, frame_num, racket_bbox, ball_detections):
        """Add new frame data and detect contact"""
        self.frame_numbers.append(frame_num)
        
        # Extract current ball position (with estimation if not detected)
        current_ball = self.extract_ball_position(ball_detections)
        
        # Extract current racket position (with estimation if not detected)
        current_racket = self.extract_racket_position(racket_bbox)
        
        # Store in history
        self.ball_history.append(current_ball)
        self.racket_history.append(current_racket)
        
        # Check for contact
        is_contact = self.detect_contact(current_ball, current_racket)
        
        if is_contact:
            # Add to current contact sequence
            contact_info = {
                'frame': frame_num,
                'ball_position': current_ball,
                'racket_bbox': current_racket
            }
            self.current_contact_sequence.append(contact_info)
            self.raw_contact_frames.append(frame_num)
        else:
            # No contact - check if we should finalize current sequence
            if self.current_contact_sequence:
                self.finalize_contact_sequence()
    
    def finalize_contact_sequence(self):
        """Finalize current contact sequence and validate it"""
        if not self.current_contact_sequence:
            return
        
        sequence_length = len(self.current_contact_sequence)
        
        # Validation rules:
        # 1. At least 2 consecutive frames
        # 2. If more than 5 frames, still counts as 1 contact point
        if sequence_length >= 2:
            # This is a valid contact point
            start_frame = self.current_contact_sequence[0]['frame']
            end_frame = self.current_contact_sequence[-1]['frame']
            
            # Use the middle frame as the representative contact frame
            middle_index = len(self.current_contact_sequence) // 2
            representative_contact = self.current_contact_sequence[middle_index]
            
            valid_contact = {
                'contact_point_id': len(self.valid_contact_points) + 1,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'duration_frames': sequence_length,
                'representative_frame': representative_contact['frame'],
                'ball_position': representative_contact['ball_position'],
                'racket_bbox': representative_contact['racket_bbox'],
                'all_contact_frames': [contact['frame'] for contact in self.current_contact_sequence]
            }
            
            self.valid_contact_points.append(valid_contact)
            print(f"Valid contact point detected: frames {start_frame}-{end_frame} ({sequence_length} frames)")
        
        # Reset current sequence
        self.current_contact_sequence = []
    
    def extract_ball_position(self, ball_detections):
        """Extract ball center position with estimation when not detected - using largest ball"""
        if ball_detections:
            # Find the ball with the largest area (not highest confidence)
            largest_ball = None
            max_area = 0
            
            for ball in ball_detections:
                x1, y1, x2, y2 = ball['bbox']
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    largest_ball = ball
            
            if largest_ball:
                x1, y1, x2, y2 = largest_ball['bbox']
                
                ball_position = {
                    'center_x': (x1 + x2) / 2,
                    'center_y': (y1 + y2) / 2,
                    'area': max_area,
                    'confidence': largest_ball['weighted_confidence'],
                    'detected': True
                }
                
                # Update last known position
                self.last_ball_position = ball_position
                return ball_position
        
        # Ball not detected - use last known position
        if self.last_ball_position is not None:
            estimated_position = {
                'center_x': self.last_ball_position['center_x'],
                'center_y': self.last_ball_position['center_y'],
                'area': self.last_ball_position.get('area', 0),
                'confidence': 0.0,  # No confidence since it's estimated
                'detected': False
            }
            return estimated_position
        else:
            # No previous position available
            return None
    
    def extract_racket_position(self, racket_bbox):
        """Extract racket position with estimation when not detected"""
        if racket_bbox:
            x1, y1, x2, y2 = racket_bbox
            
            racket_position = {
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'center_x': (x1 + x2) / 2,
                'center_y': (y1 + y2) / 2,
                'width': x2 - x1,
                'height': y2 - y1,
                'detected': True
            }
            
            # Update last known position
            self.last_racket_position = racket_position
            return racket_position
        
        else:
            # Racket not detected - use last known position
            if self.last_racket_position is not None:
                estimated_position = {
                    'x1': self.last_racket_position['x1'],
                    'y1': self.last_racket_position['y1'],
                    'x2': self.last_racket_position['x2'],
                    'y2': self.last_racket_position['y2'],
                    'center_x': self.last_racket_position['center_x'],
                    'center_y': self.last_racket_position['center_y'],
                    'width': self.last_racket_position['width'],
                    'height': self.last_racket_position['height'],
                    'detected': False
                }
                return estimated_position
            else:
                # No previous position available
                return None
    
    def detect_contact(self, ball_position, racket_position):
        """Detect if ball center is within racket bounding box"""
        if ball_position is None or racket_position is None:
            return False
        
        ball_x = ball_position['center_x']
        ball_y = ball_position['center_y']
        
        # Check if ball center is within racket bounding box
        if (racket_position['x1'] <= ball_x <= racket_position['x2'] and
            racket_position['y1'] <= ball_y <= racket_position['y2']):
            return True
        
        return False
    
    def get_contact_summary(self):
        """Get summary of all detected contacts"""
        return {
            'total_valid_contacts': len(self.valid_contact_points),
            'valid_contact_points': self.valid_contact_points,
            'total_raw_contact_frames': len(self.raw_contact_frames),
            'raw_contact_frames': self.raw_contact_frames
        }
    
    def print_contact_results(self):
        """Print all contact detection results"""
        # Finalize any remaining contact sequence
        if self.current_contact_sequence:
            self.finalize_contact_sequence()
        
        if not self.valid_contact_points:
            print("No valid ball-racket contacts detected.")
            print(f"Raw contact frames detected: {len(self.raw_contact_frames)}")
            if self.raw_contact_frames:
                print(f"Raw frames: {self.raw_contact_frames}")
                print("(These didn't meet the minimum 2 consecutive frames requirement)")
            return
        
        print(f"\nDetected {len(self.valid_contact_points)} valid ball-racket contact points:")
        
        print("\nSummary:")
        for contact in self.valid_contact_points:
            if contact['duration_frames'] == 1:
                print(f"Contact {contact['contact_point_id']}: Frame {contact['representative_frame']}")
            else:
                print(f"Contact {contact['contact_point_id']}: Frames {contact['start_frame']}-{contact['end_frame']} ({contact['duration_frames']} frames)")
        
        print("\nDetailed Contact Information:")
        for contact in self.valid_contact_points:
            ball = contact['ball_position']
            racket = contact['racket_bbox']
            print(f"Contact Point {contact['contact_point_id']}:")
            print(f"  Duration: {contact['start_frame']}-{contact['end_frame']} ({contact['duration_frames']} frames)")
            print(f"  Representative frame: {contact['representative_frame']}")
            print(f"  Ball position: ({ball['center_x']:.1f}, {ball['center_y']:.1f})")
            print(f"  Ball area: {ball.get('area', 'N/A'):.1f} pixels²")
            print(f"  Ball detected: {ball['detected']}")
            print(f"  Racket center: ({racket['center_x']:.1f}, {racket['center_y']:.1f})")
            print(f"  Racket detected: {racket['detected']}")
            print(f"  All contact frames: {contact['all_contact_frames']}")
            print()
        
        print(f"Total raw contact frames: {len(self.raw_contact_frames)}")
        if len(self.raw_contact_frames) > sum(cp['duration_frames'] for cp in self.valid_contact_points):
            invalid_frames = len(self.raw_contact_frames) - sum(cp['duration_frames'] for cp in self.valid_contact_points)
            print(f"Invalid single-frame contacts filtered out: {invalid_frames}")
    
    def get_frame_contact_list(self):
        """Return simple list of representative frame numbers where valid contact occurred"""
        # Finalize any remaining contact sequence
        if self.current_contact_sequence:
            self.finalize_contact_sequence()
        
        return [contact['representative_frame'] for contact in self.valid_contact_points]
    
    def get_all_valid_contact_frames(self):
        """Return all frame numbers from valid contact sequences"""
        # Finalize any remaining contact sequence
        if self.current_contact_sequence:
            self.finalize_contact_sequence()
        
        all_frames = []
        for contact in self.valid_contact_points:
            all_frames.extend(contact['all_contact_frames'])
        return sorted(all_frames)