import cv2
import torch
import numpy as np
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from scipy.spatial.distance import cosine
import threading
import queue
import time
import os
import winsound
from datetime import datetime



# pi code >>> libcamera-vid -t 0 --inline --profile baseline --width 640 --height 480 --framerate 30 -o udp://192.168.29.15:5000

# Load the models
yolo_model = YOLO(r"C:\Users\Hrishikesh\Soruce Codes\Major\yolov8m.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load known faces
known_faces = np.load("face_embeddings.npy", allow_pickle=True).item()

def get_embedding(face):
    face = cv2.resize(face, (160, 160))  # Resize to FaceNet input size
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    face = torch.tensor(face, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0  
    with torch.no_grad():
        embedding = facenet_model(face).cpu().numpy().flatten()
    return embedding

def recognize_face(face_embedding, threshold=0.4):
    best_match = "Unknown"
    min_distance = float("inf")

    for name, known_embedding in known_faces.items():
        distance = cosine(face_embedding, known_embedding)
        if distance < min_distance:
            min_distance = distance
            best_match = name

    return best_match if min_distance < threshold else "Unknown"

# Threaded frame capture to avoid blocking the main loop
frame_queue = queue.Queue(maxsize=1)
stop_event = threading.Event()

def frame_reader(cap, frame_queue, stop_event):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        # Put the latest frame into the queue
        if frame_queue.full():
            frame_queue.get_nowait()  # Discard the old frame
        frame_queue.put_nowait(frame)

# Initialize video stream from Pi
udp_uri = "udp://0.0.0.0:5000?overrun_nonfatal=1&analyzeduration=100000&probesize=32"
cap = cv2.VideoCapture(udp_uri, cv2.CAP_FFMPEG)
if not cap.isOpened():
    print(f"Error: Could not open video stream from {udp_uri}")
    exit()

# Start frame reader thread
reader_thread = threading.Thread(target=frame_reader, args=(cap, frame_queue, stop_event), daemon=True)
reader_thread.start()

# Play beep sound
def play_beep():
    try:
        winsound.Beep(1000, 500)
        
    except Exception as e:
        print(f"Error playing beep: {e}")

# Save screenshot to Desktop
def save_screenshot(frame, name):
    desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    screenshot_path = os.path.join(desktop_path, f'{name}_{timestamp}.png')
    cv2.imwrite(screenshot_path, frame)
    print(f"Screenshot saved to {screenshot_path}")

# Main processing loop
while True:
    if frame_queue.empty():
        time.sleep(0.001)  # Small delay if no new frame is available
        continue

    # Get the latest frame from the queue
    frame = frame_queue.get_nowait()

    # Run YOLO detection
    results = yolo_model(frame, conf=0.4)
    detected_faces = []
    detected_objects = []
    context_info = []

    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):  
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls.item())

            detected_label = yolo_model.names[class_id]
            if detected_label.lower() in ["person", "face"]:
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                # Run FaceNet on the detected face
                embedding = get_embedding(face)
                name = recognize_face(embedding)

                detected_faces.append(name)

                # Bounding box and label
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if name != "Unknown":
                    # If the person is recognized, play beep and save screenshot
                    play_beep()
                    save_screenshot(frame, name)

            else:  # Object detection
                detected_objects.append(detected_label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, detected_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Prepare context info
    if detected_faces:
        context_info.append(f"Faces Detected: {', '.join(detected_faces)}")

    if detected_objects:
        context_info.append(f"Objects in scene: {', '.join(detected_objects)}")

    if len(detected_faces) > 1:
        context_info.append(f"Multiple people detected: {len(detected_faces)}")

    context_text = "\n".join(context_info) if context_info else "No faces or objects detected"
    context_frame = np.ones((300, 500, 3), dtype=np.uint8) * 255

    y0, dy = 20, 30  
    for i, line in enumerate(context_text.split("\n")):
        cv2.putText(context_frame, line, (10, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Show the frame with annotations and context window
    cv2.imshow("Face vs Object Classification", frame)
    cv2.imshow("Context Analysis", context_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_event.set()  # Stop the frame reader thread
        break

cap.release()
cv2.destroyAllWindows()
