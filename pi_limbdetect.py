import cv2
import time
import winsound
from ultralytics import YOLO
import threading

# Load your trained YOLOv8 model
model = YOLO(r"C:\Users\Hrishikesh\Soruce Codes\Major\best.pt")

# Open the UDP stream from the Pi
cap = cv2.VideoCapture("udp://0.0.0.0:5000", cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

frame_skip = 2
frame_count = 0

# Detection stability tracking
detection_start_time = None
detection_active = False

def alert():
    winsound.Beep(1000, 500)  # Beep sound at 1000 Hz for 500 ms
    print("ALERT: Stable detection confirmed!")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        continue

    if frame_count % frame_skip == 0:
        # Run inference
        results = model.predict(source=frame, conf=0.2, imgsz=640, verbose=False)
        annotated_frame = results[0].plot()

        # Check for detections above threshold
        detected = False
        for box in results[0].boxes:
            if box.conf.item() > 0.5:
                detected = True
                break

        current_time = time.time()

        if detected:
            if detection_start_time is None:
                detection_start_time = current_time
            elif current_time - detection_start_time >= 3 and not detection_active:
                detection_active = True
                threading.Thread(target=alert).start()
        else:
            detection_start_time = None
            detection_active = False

        # Show the annotated frame
        cv2.imshow("Face Detection - YOLOv8", annotated_frame)

    frame_count += 1

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
