import logging
import cv2
import torch
import pyttsx3
import pytesseract
import numpy as np
from ultralytics import YOLO

class TrafficSignReader:
    def __init__(self, model_path='best.pt', speed_class="regulatory--maximum-speed-limit"):
        # Suppress Ultralytics logging
        logging.getLogger("ultralytics").setLevel(logging.WARNING)

        # Load YOLO model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path).to(self.device)
        self.speed_class = speed_class.lower()

        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1.0)

        # Set tesseract path (update this if necessary)
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        # Track last announcement to prevent repetition
        self.last_announcement = None

    def preprocess_for_ocr(self, image):
        """Apply grayscale, CLAHE, adaptive thresholding, and morphology for OCR"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        thresh = cv2.adaptiveThreshold(enhanced, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 4)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return processed

    def extract_speed_text(self, image):
        """Try different OCR configurations and return the best digit result"""
        processed = self.preprocess_for_ocr(image)
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
        return best_speed

    def speak(self, message):
        """Speak out a message if it hasn’t been said already"""
        if message != self.last_announcement:
            print(message)
            self.engine.say(message)
            self.engine.runAndWait()
            self.last_announcement = message

    def process_frame(self, frame):
        """Detect signs in a frame and handle TTS announcements"""
        results = self.model(frame, conf=0.6)
        current_detections = set()

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id].lower()

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, class_name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Handle speed limit signs
                if class_name == self.speed_class:
                    sign_roi = frame[y1:y2, x1:x2]
                    best_speed = self.extract_speed_text(sign_roi)
                    if best_speed:
                        message = f"Speed limit {best_speed}"
                        cv2.putText(frame, f"{best_speed} km/h", (x1, y1 - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    else:
                        message = "Speed limit sign"
                    self.speak(message)
                else:
                    self.speak(class_name)

                current_detections.add(class_name)

        # Reset last announcement if no detections
        if not current_detections:
            self.last_announcement = None

        return frame

    def run(self):
        """Capture video from webcam and process each frame"""
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = self.process_frame(frame)
            cv2.imshow("Traffic Sign Detection", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    reader = TrafficSignReader()
    reader.run()
