import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO


root = tk.Tk()
root.withdraw() 

model_path = "C:\\Users\\User\\Desktop\\archive\\best.pt"

model = YOLO(model_path)

def upload_picture(image):
    results = model(image, conf=0.4)
    if isinstance(results, list):
        results = results[0]
    annotated_frame = results.plot()
    cv2.imshow('Object Detection', annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def upload_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    desired_fps = cap.get(cv2.CAP_PROP_FPS)
    delay_time = 1.0 / desired_fps

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, desired_fps, (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.4)
        if isinstance(results, list):
            results = results[0]

        annotated_frame = results.plot()

        out.write(annotated_frame)

        cv2.imshow('Real-Time Detection', annotated_frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        time.sleep(delay_time)

    out.release()
    cap.release()
    cv2.destroyAllWindows()

file_path = filedialog.askopenfilename(title="Select file", filetypes=(("Image files", "*.jpg;*.jpeg;*.png"),
                                                                       ("Video files", "*.mp4;*.avi")))

if file_path:
    if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        image = cv2.imread(file_path)
        upload_picture(image)
    elif file_path.lower().endswith(('.mp4', '.avi')):
        upload_video(file_path)
