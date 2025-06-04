from ultralytics import YOLO
import logging
import cv2
import mediapipe as mp

logging.getLogger("ultralytics").setLevel(logging.ERROR)
# Load YOLOv8 pretrained model
model = YOLO('yolov8n.pt')

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Open video
cap = cv2.VideoCapture("shots-dataset/test.mp4")

PADDING = 80  # number of pixels to expand around the detected box

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    results = model(frame)[0]

    # Filter for 'person' detections only
    person_boxes = [box for box in results.boxes if int(box.cls[0]) == 0]

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

        # Draw pose landmarks on the cropped image
        # if results.pose_landmarks:
        #    mp_drawing.draw_landmarks(player_crop, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Show the pose overlay
        # cv2.imshow("Pose on Player", player_crop)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()