import cv2
import numpy as np
from picamera2 import Picamera2
import time
from ultralytics import YOLO

# Initialize Picamera2
picam2 = Picamera2()

# Configure preview settings
picam2.preview_configuration.main.size = (1920,1080) 
picam2.preview_configuration.main.format = "BGR888"
picam2.preview_configuration.align()
picam2.configure("preview")

# Start the preview
picam2.start()

# Define the path to the YOLO model
model_path = "/home/admin/Desktop/Pothole/best.pt"

# Initialize YOLO model
model = YOLO(model_path)

# Define the desired FPS
desired_fps = 4
delay_time = 1.0 / desired_fps

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, desired_fps, (640, 360))

# Loop for capturing frames
while True:
    start_time = time.time()
    
    # Capture frame
    im = picam2.capture_array()

    # Perform object detection
    results = model(im, conf=0.4)
    
    # If multiple results are returned, take the first one
    if isinstance(results, list):
        results = results[0]
    
    # Plot bounding boxes on the frame
    annotated_frame = results.plot()

    # Convert the annotated frame to BGR format for compatibility with OpenCV
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    
    # Write the frame to the output video
    out.write(annotated_frame)
    
    # Show the annotated frame in real-time
    cv2.imshow('Real-Time Detection', annotated_frame)
    
    # Check for quit command
    if cv2.waitKey(1) == ord('q'):
        break
    
    # Ensure desired FPS
    elapsed_time = time.time() - start_time
    remaining_time = max(0, delay_time - elapsed_time)
    time.sleep(remaining_time)

# Release video writer and clean up
out.release()
cv2.destroyAllWindows()
