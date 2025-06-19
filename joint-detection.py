from ultralytics import YOLO
import logging
import cv2
import numpy as np
import mediapipe as mp

# Suppress YOLO logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Load YOLO model
model = YOLO('yolov8n.pt')
# ball_model = YOLO('best.pt')

# Open video
cap = cv2.VideoCapture("shots-dataset/test.mp4")

PADDING = 200  # number of pixels to expand around the detected box
CROP_SIZE = 640  # target size for player crop (640x640)

def ready_position():
    
    return

def crop_and_scale_position(crop_height, crop_width, x1, y1,x2, y2, x1_padded, y1_padded, scale_x, scale_y):
    
    x1_crop = max(0, x1 - x1_padded)
    y1_crop = max(0, y1 - y1_padded)
    x2_crop = min(crop_width, x2 - x1_padded)
    y2_crop = min(crop_height, y2 - y1_padded)
    
    x1_scaled = int(x1_crop * scale_x)
    y1_scaled = int(y1_crop * scale_y)
    x2_scaled = int(x2_crop * scale_x)
    y2_scaled = int(y2_crop * scale_y)

    return x1_scaled, y1_scaled, x2_scaled, y2_scaled

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    results = model(frame)[0]
    # ball_results = ball_model(frame)[0]

    # Filter detections by class
    person_boxes = [box for box in results.boxes if int(box.cls[0]) == 0]  # person class
    racket_boxes = [box for box in results.boxes if int(box.cls[0]) == 38]  # tennis racket class
    # ball_boxes = [box for box in ball_results.boxes if int(box.cls[0]) == 0] # ball class in trained model
    
    # Use own-trained model for tennis ball detection tennis ball

    if person_boxes:
        # Choose the person with the largest area
        largest_person = max(person_boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
        x1, y1, x2, y2 = map(int, largest_person.xyxy[0])

        # Add padding (but clamp to image boundaries)
        x1_padded = max(0, x1 - PADDING)
        y1_padded = max(0, y1 - PADDING)
        x2_padded = min(width, x2 + PADDING)
        y2_padded = min(height, y2 + PADDING)

        # Crop the player with padding
        player_crop = frame[y1_padded:y2_padded, x1_padded:x2_padded]
        
        # Get original crop dimensions
        crop_height, crop_width = player_crop.shape[:2]
        
        # Resize to 640x640 pixels
        player_crop_resized = cv2.resize(player_crop, (CROP_SIZE, CROP_SIZE))

        # Calculate scaling factors
        scale_x = CROP_SIZE / crop_width
        scale_y = CROP_SIZE / crop_height

        # Draw player box on resized crop (transform coordinates)
        player_x1_scaled, player_y1_scaled, player_x2_scaled, player_y2_scaled = crop_and_scale_position(crop_height, crop_width, x1, y1, x2, y2, x1_padded, y1_padded, scale_x, scale_y)
        
        cv2.rectangle(player_crop_resized, (player_x1_scaled, player_y1_scaled), (player_x2_scaled, player_y2_scaled), (0, 255, 0), 2)
        cv2.putText(player_crop_resized, "Player", (player_x1_scaled, player_y1_scaled - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Racket boxes (red) - transform and scale coordinates
        if racket_boxes:
            # Choose the racket with the largest area
            largest_racket = max(racket_boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
            x1r, y1r, x2r, y2r = map(int, largest_racket.xyxy[0])
            
            # Check if racket is within the crop area
            if (x1r < x2_padded and x2r > x1_padded and y1r < y2_padded and y2r > y1_padded):
                # Transform to crop coordinates
                racket_x1_scaled, racket_y1_scaled, racket_x2_scaled, racket_y2_scaled = crop_and_scale_position(crop_height, crop_width, x1r, y1r, x2r, y2r, x1_padded, y1_padded, scale_x, scale_y)
                
                cv2.rectangle(player_crop_resized, (racket_x1_scaled, racket_y1_scaled), (racket_x2_scaled, racket_y2_scaled), (0, 0, 255), 2)
                cv2.putText(player_crop_resized, "Racket", (racket_x1_scaled, racket_y1_scaled - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else :
                print("racket not detected")
                
        # Ball boxes using color detection (blue) - transform and scale coordinates
        """ if ball_positions:
            for (x1b, y1b, x2b, y2b) in ball_positions:
                # Check if ball is within the crop area
                if (x1b < x2_padded and x2b > x1_padded and y1b < y2_padded and y2b > y1_padded):
                    # Transform to crop coordinates
                    ball_x1_crop = max(0, x1b - x1_padded)
                    ball_y1_crop = max(0, y1b - y1_padded)
                    ball_x2_crop = min(crop_width, x2b - x1_padded)
                    ball_y2_crop = min(crop_height, y2b - y1_padded)
                    
                    # Scale to resized dimensions
                    ball_x1_scaled = int(ball_x1_crop * scale_x)
                    ball_y1_scaled = int(ball_y1_crop * scale_y)
                    ball_x2_scaled = int(ball_x2_crop * scale_x)
                    ball_y2_scaled = int(ball_y2_crop * scale_y)
                    
                    cv2.rectangle(player_crop_resized, (ball_x1_scaled, ball_y1_scaled), (ball_x2_scaled, ball_y2_scaled), (255, 0, 0), 2)
                    cv2.putText(player_crop_resized, "Ball", (ball_x1_scaled, ball_y1_scaled - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2) """
        
        # Display the cropped and resized player
        cv2.imshow("Player Crop (640x640)", player_crop_resized)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()