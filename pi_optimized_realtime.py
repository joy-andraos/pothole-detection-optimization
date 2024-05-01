import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# Initialize Picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 640)  # Adjust the resolution here
picam2.preview_configuration.main.format = "BGR888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Define the path to the YOLO model
model_path = "/home/admin/Desktop/Pothole/best.pt"

# Initialize YOLO model
ov_model = YOLO(model_path)

# Export the model to OpenVINO
ov_model.export(format='openvino', imgsz=(640, 640), half=False)

# Load the OpenVINO model
model = YOLO('best_openvino_model/')

# Define pruning parameters
prune_amount = 0.2  # Percentage of weights to prune

# Specify prunable layers
prunable_modules = [module for module in model.modules() if isinstance(module, nn.Conv2d)]


# Define pruning strategy
for module in prunable_modules:
    if module.weight.requires_grad:  # Skip layers that shouldn't be pruned
        prune.l1_unstructured(module, name='weight', amount=prune_amount)

# Apply pruning to the model
for module in prunable_modules:
    if module.weight.requires_grad:  # Skip layers that shouldn't be pruned
        prune.remove(module, 'weight')  # Remove pruned weights from the layer


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 30, (640, 640))  # Adjust FPS and resolution if needed

# Loop for capturing frames
while True:
    # Capture frame
    im = picam2.capture_array()
    
    # Resize image to match model input shape
    im_resized = cv2.resize(im, (640, 640))

    # Perform object detection
    results = model(im_resized, conf=0.4)
    
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

# Release video writer and clean up
out.release()
cv2.destroyAllWindows()
