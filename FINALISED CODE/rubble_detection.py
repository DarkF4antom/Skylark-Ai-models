import cv2
import time
import winsound
from ultralytics import YOLO
import threading
import queue # Import queue for thread-safe communication
import sys 


# code for starting stream>> libcamera-vid -t 0 --inline --profile baseline --width 640 --height 480 --framerate 30 -o udp://192.168.29.15:5000


IS_WINDOWS = sys.platform.startswith('win')


model = YOLO(r"C:\Users\Hrishikesh\Soruce Codes\Major\best.pt")


latest_frame_queue = queue.Queue(maxsize=1)


stop_event = threading.Event()

def frame_reader(cap, frame_queue, stop_event):
    """Thread function to read frames from the video capture as fast as possible."""
    print("Frame reader thread started")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
           
            time.sleep(0.01)
            continue


        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            
             try:
                 frame_queue.put_nowait(frame)
             except queue.Full:
                
                 pass



udp_uri = "udp://0.0.0.0:5000?overrun_nonfatal=1&analyzeduration=100000&probesize=32"


cap = cv2.VideoCapture(udp_uri, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print(f"Error: Could not open video stream from {udp_uri}")
    print("Please ensure the Pi is streaming to udp://YOUR_LAPTOP_IP:5000")
    print("and that port 5000 is open on your laptop's firewall.")
    exit()

# Start the frame reader thread
reader_thread = threading.Thread(target=frame_reader, args=(cap, latest_frame_queue, stop_event), daemon=True)
reader_thread.start()

# --- Alert Function ---
def alert():
    """Plays a beep sound."""
    
    if IS_WINDOWS:
        try:
            winsound.Beep(1000, 500)
            winsound.Beep(2000,500)
            winsound.Beep(2500,500)
            
            
            # Beep sound at 1000 Hz for 500 ms
        except Exception as e:
            print(f"Error playing beep: {e}")
            print("ALERT: Stable detection confirmed! (Beep failed)")
    else:
         print("ALERT: Stable detection confirmed!") 


# --- Main Processing Loop ---

detection_start_time = None
detection_active = False

print("Main processing loop started")
while True:
    # Get the latest frame from the queue without waiting indefinitely
    try:
        
        frame = latest_frame_queue.get_nowait()
    except queue.Empty:
        
        time.sleep(0.001) # Sleep for 1ms
        continue 

    
    results = model.predict(source=frame, conf=0.2, imgsz=640, verbose=False)

    # Annotate the frame
    annotated_frame = results[0].plot()

    
    detected = False
    
    for box in results[0].boxes:
        
        if box.conf.item() > 0.5: 
            detected = True
            
            break 

    # --- Detection Stability Logic ---
    current_time = time.time()

    if detected:
        if detection_start_time is None:
            
            detection_start_time = current_time
            
        elif (current_time - detection_start_time >= 2) and not detection_active:
            detection_active = True
            print("Stable detection threshold reached (2 seconds). Triggering alert.") # Optional: debug
            
            threading.Thread(target=alert, daemon=True).start() 
    else:
        # If no detection, reset the tracking
        if detection_start_time is not None:
            
            pass 
        detection_start_time = None
        detection_active = False 

    # Show the annotated frame
    cv2.imshow("Face Detection - YOLOv8", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quit requested. Stopping threads...")
        stop_event.set() # Signal the reader thread to stop
        break # Exit the main loop

# --- Cleanup ---
print("Releasing video capture...")
cap.release()
print("Waiting for reader thread to finish...")
reader_thread.join()
print("Destroying windows...")
cv2.destroyAllWindows()
print("Program finished.")