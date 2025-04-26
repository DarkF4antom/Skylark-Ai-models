import cv2
import torch
import numpy as np
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from scipy.spatial.distance import cosine

yolo_model = YOLO(r"C:\Users\Hrishikesh\Soruce Codes\Major\yolov8m.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
known_faces = np.load("face_embeddings.npy", allow_pickle=True).item()

def get_embedding(face):
    face = cv2.resize(face, (160, 160))  # Resize to FaceNet input size
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    face = torch.tensor(face, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0  
    with torch.no_grad():
        embedding = facenet_model(face).cpu().numpy().flatten()
    return embedding




def recognize_face(face_embedding, threshold=0.4):
    
    # best_match = "Unknown"
    # min_distance = float("inf")
    
    # for name, known_embedding in known_faces.items():
    #     distance = cosine(face_embedding, known_embedding)
    #     if distance < min_distance:
    #         min_distance = distance
    #         best_match = name
    
    # return best_match if min_distance < threshold else "Unknown"
    
    best_match = "Unknown"
    min_distance = float("inf")

    for name, known_embedding in known_faces.items():
        distance = cosine(face_embedding, known_embedding)
        print(f"Comparing with {name}: Distance = {distance}")  

        if distance < min_distance:
            min_distance = distance
            best_match = name

    print(f"Best match: {best_match} (Distance: {min_distance})")
    return best_match if min_distance < threshold else "Unknown"




cap = cv2.VideoCapture(0) #<webcam image
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    

    results = yolo_model(frame, conf=0.4)
    
    context_info = []  
    
    detected_faces = []
    detected_objects = []
    
    
    
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):  
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls.item())  

            
            detected_label = yolo_model.names[class_id]

            if detected_label.lower() in ["person", "face"]:
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                
                embedding = get_embedding(face)

               
                name = recognize_face(embedding)
                detected_faces.append(name)

                #bounding box and label
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            else:  #object
                detected_objects.append(detected_label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, detected_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # displaying stuff 
     
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

    
    cv2.imshow("Face vs Object Classification", frame)
    cv2.imshow("Context Analysis", context_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
