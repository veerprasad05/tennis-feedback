from ultralytics import YOLO
import logging
import cv2

# Suppress YOLO logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Load YOLOv8 pretrained model
model = YOLO('yolov8n.pt')

# Open video
cap = cv2.VideoCapture("shots-dataset/test.mp4")

PADDING = 200  # number of pixels to expand around the detected box
CROP_SIZE = 360  # target size for player crop (360x360)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    results = model(frame)[0]

    # Filter detections by class
    person_boxes = [box for box in results.boxes if int(box.cls[0]) == 0]  # person class
    racket_boxes = [box for box in results.boxes if int(box.cls[0]) == 38]  # tennis racket class
    ball_boxes = [box for box in results.boxes if int(box.cls[0]) == 32]  # sports ball class
    
    # Ball boxes (blue) - transform and scale coordinates
    if ball_boxes:
        x1b, y1b, x2b, y2b = map(int, ball_boxes[0].xyxy[0])
        
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
            cv2.putText(player_crop_resized, "Ball", (ball_x1_scaled, ball_y1_scaled - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

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
        
        # Resize to 360x360 pixels
        player_crop_resized = cv2.resize(player_crop, (CROP_SIZE, CROP_SIZE))

        # Calculate scaling factors
        scale_x = CROP_SIZE / crop_width
        scale_y = CROP_SIZE / crop_height

        # Draw player box on resized crop (transform coordinates)
        # Player box coordinates relative to the crop
        player_x1_crop = max(0, x1 - x1_padded)
        player_y1_crop = max(0, y1 - y1_padded)
        player_x2_crop = min(crop_width, x2 - x1_padded)
        player_y2_crop = min(crop_height, y2 - y1_padded)
        
        # Scale to resized dimensions
        player_x1_scaled = int(player_x1_crop * scale_x)
        player_y1_scaled = int(player_y1_crop * scale_y)
        player_x2_scaled = int(player_x2_crop * scale_x)
        player_y2_scaled = int(player_y2_crop * scale_y)
        
        cv2.rectangle(player_crop_resized, (player_x1_scaled, player_y1_scaled), (player_x2_scaled, player_y2_scaled), (0, 255, 0), 2)
        cv2.putText(player_crop_resized, "Player", (player_x1_scaled, player_y1_scaled - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Racket boxes (red) - transform and scale coordinates
        if racket_boxes:
            # Choose the person with the largest area
            largest_racket = max(racket_boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
            x1r, y1r, x2r, y2r = map(int, largest_racket.xyxy[0])
            
            # Check if racket is within the crop area
            if (x1r < x2_padded and x2r > x1_padded and y1r < y2_padded and y2r > y1_padded):
                # Transform to crop coordinates
                racket_x1_crop = max(0, x1r - x1_padded)
                racket_y1_crop = max(0, y1r - y1_padded)
                racket_x2_crop = min(crop_width, x2r - x1_padded)
                racket_y2_crop = min(crop_height, y2r - y1_padded)
                
                # Scale to resized dimensions
                racket_x1_scaled = int(racket_x1_crop * scale_x)
                racket_y1_scaled = int(racket_y1_crop * scale_y)
                racket_x2_scaled = int(racket_x2_crop * scale_x)
                racket_y2_scaled = int(racket_y2_crop * scale_y)
                
                cv2.rectangle(player_crop_resized, (racket_x1_scaled, racket_y1_scaled), (racket_x2_scaled, racket_y2_scaled), (0, 0, 255), 2)
                cv2.putText(player_crop_resized, "Racket", (racket_x1_scaled, racket_y1_scaled - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Display the cropped and resized player
        cv2.imshow("Player Crop (360x360)", player_crop_resized)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()