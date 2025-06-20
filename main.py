from ultralytics import YOLO
import logging
import cv2
import numpy as np
import mediapipe as mp
import gc
import traceback

from balldetector import TennisBallDetector
from strokedetector import TennisStrokeDetector

# Suppress YOLO logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Constants
PADDING = 200
CROP_SIZE = 640

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