import logging
from ultralytics import YOLO
import cv2
import torch

# Suppress YOLO logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Load custom-trained traffic sign detection model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('traffic-signs.pt').to(device)  # Use your trained model

# Open webcam
cap = cv2.VideoCapture(0)  # 0 = default webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if there's an issue with the webcam

    # Run YOLO detection on the frame
    results = model(frame, conf=0.6)  # Adjust confidence threshold if needed

    # Process detections
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            class_id = int(box.cls[0])  # Get class ID
            class_name = model.names[class_id]  # Get detected class name

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display sign name
            cv2.putText(frame, class_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Print detected sign in console
            print(f"Detected: {class_name}")

    # Show video feed with detections
    cv2.imshow("Traffic Sign Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
