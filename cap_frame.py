import numpy as np
import cv2
import os

cap = cv2.VideoCapture('assets/sample3.mp4')

output_folder = 'output_images'
os.makedirs(output_folder, exist_ok=True)

existing_files = [f for f in os.listdir(output_folder) if f.startswith('frame_') and f.endswith('.png')]
frame_count = len(existing_files)  

while True:
    ret, frame = cap.read()
    if not ret:
        break  
    frame_count += 1 
    output_path = os.path.join(output_folder, f'frame_{frame_count}.png')
    cv2.imwrite(output_path, frame)
    print(f"フレームを保存しました: {output_path}")

cap.release()
