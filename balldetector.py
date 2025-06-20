from ultralytics import YOLO
import cv2
import numpy as np

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