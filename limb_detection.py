import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO(r"C:\Users\Hrishikesh\Soruce Codes\Major\best.pt")

# Start webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the frame
    results = model.predict(source=frame, conf=0.5, imgsz=640, verbose=False)

    # Visualize results on frame
    annotated_frame = results[0].plot()

    # Show the annotated frame
    cv2.imshow("Limb Detection - YOLOv8", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
