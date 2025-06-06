from ultralytics import YOLO
import logging
import cv2
import mediapipe as mp
from collections import deque

logging.getLogger("ultralytics").setLevel(logging.ERROR)
# Load YOLOv8 pretrained model
model = YOLO('yolov8n.pt')
# racket_model = YOLO('best.pt')

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Open video
cap = cv2.VideoCapture("shots-dataset/test.mp4")

PADDING = 150  # number of pixels to expand around the detected box

# ── NEW helper ─────────────────────────────────────────────────────────
def print_and_highlight_right_arm(pose_landmarks, crop_img):
    """
    Print (x, y) pixel coords of right wrist, elbow, shoulder
    and draw green circles on those points.
    """
    # MediaPipe landmark indices for the right arm
    RIGHT_WRIST, RIGHT_ELBOW, RIGHT_SHOULDER = 16, 14, 12
    indices = [RIGHT_WRIST, RIGHT_ELBOW, RIGHT_SHOULDER]
    names   = ["Right wrist", "Right elbow", "Right shoulder"]

    h, w = crop_img.shape[:2]

    for idx, name in zip(indices, names):
        lm   = pose_landmarks.landmark[idx]
        x_px = int(lm.x * w)
        y_px = int(lm.y * h)
        print(f"{name} coords: {x_px}, {y_px}")
        # draw a green filled circle
        cv2.circle(crop_img, (x_px, y_px), 6, (0, 255, 0), -1)
    print("new frame\n")
# ───────────────────────────────────────────────────────────────────────

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    results = model(frame)[0]
    # racket_results = racket_model(frame)[0]

    # Filter for 'person' detections only
    person_boxes = [box for box in results.boxes if int(box.cls[0]) == 0]
    racket_boxes = [box for box in results.boxes if int(box.cls[0]) == 38]

    if person_boxes:
        # Get the largest detected person
        largest_box = max(person_boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
        x1, y1, x2, y2 = map(int, largest_box.xyxy[0])

        # Add padding (but clamp to image size)
        x1 = max(0, x1 - PADDING)
        y1 = max(0, y1 - PADDING)
        x2 = min(width, x2 + PADDING)
        y2 = min(height, y2 + PADDING)

        # Crop with padding
        player_crop = frame[y1:y2, x1:x2]
        # cv2.imshow("Padded Player Frame", player_crop)
        
        # Convert cropped image to RGB for MediaPipe
        crop_rgb = cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB)

        # Run pose estimation
        results = pose.process(crop_rgb)

        # Stroke analysis to find which stroke is being played
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(player_crop, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            print_and_highlight_right_arm(results.pose_landmarks, player_crop)
            
            # Draw tennis racket(s) on player_crop if detected
            for box in racket_boxes:
                largest_box = max(racket_boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
                x1r, y1r, x2r, y2r = map(int, largest_box.xyxy[0])

                # Check if racket is within the player_crop bounds
                if x1 <= x1r <= x2 and y1 <= y1r <= y2:
                    # Adjust coordinates to the crop
                    crop_x1r = x1r - x1
                    crop_y1r = y1r - y1
                    crop_x2r = x2r - x1
                    crop_y2r = y2r - y1

                    # Draw on player_crop
                    cv2.rectangle(player_crop, (crop_x1r, crop_y1r), (crop_x2r, crop_y2r), (0, 0, 255), 2)
                    cv2.putText(player_crop, "Racket", (crop_x1r, crop_y1r - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Show the pose overlay
        cv2.imshow("Pose on Player", player_crop)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()