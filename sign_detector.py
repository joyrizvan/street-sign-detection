import logging
import cv2
import torch
import pyttsx3
import pytesseract
import numpy as np
from ultralytics import YOLO

# Suppress YOLO logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Load custom-trained YOLO model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('best.pt').to(device)

# Initialize TTS
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# Tesseract path (update for your system)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Speed limit class name (adjust to your model)
SPEED_CLASS = "regulatory--maximum-speed-limit"

# Track last announced sign
last_announcement = None
last_speed = None

def preprocess_for_ocr(image):
    """Enhanced preprocessing for speed limit signs"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    thresh = cv2.adaptiveThreshold(enhanced, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 4)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return processed

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.6)
    current_detections = set()  # Track detected signs in current frame

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id].lower()

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            if class_name == SPEED_CLASS:
                # Crop and preprocess speed sign
                sign_roi = frame[y1:y2, x1:x2]
                processed = preprocess_for_ocr(sign_roi)
                
                # Try multiple OCR configurations
                configs = [
                    '--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789',
                    '--psm 8 --oem 3',
                    '--psm 7 --oem 3'
                ]
                
                best_speed = None
                for config in configs:
                    text = pytesseract.image_to_string(processed, config=config)
                    digits = ''.join(filter(str.isdigit, text))
                    if digits:
                        if best_speed is None or len(digits) > len(best_speed):
                            best_speed = digits
                
                if best_speed:
                    current_sign = f"Speed limit {best_speed}"
                    cv2.putText(frame, f"{best_speed} km/h", (x1, y1-40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                    
                    # Only announce if different from last announcement
                    if current_sign != last_announcement:
                        print(current_sign)
                        engine.say(current_sign)
                        last_announcement = current_sign
                        last_speed = best_speed
                else:
                    current_sign = "Speed limit sign"
                    if current_sign != last_announcement:
                        print("Speed sign detected")
                        engine.say(current_sign)
                        last_announcement = current_sign
            else:
                current_sign = class_name
                if current_sign != last_announcement:
                    print(f"Detected: {class_name}")
                    engine.say(class_name)
                    last_announcement = class_name
            
            current_detections.add(last_announcement)

    # Reset announcement if no signs are detected
    if not current_detections and last_announcement is not None:
        last_announcement = None

    engine.runAndWait()
    cv2.imshow("Traffic Sign Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()