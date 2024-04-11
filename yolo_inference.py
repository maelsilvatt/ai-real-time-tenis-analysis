from ultralytics import YOLO
import torch

model = YOLO('yolov8x')

# Check for CUDA device and set it
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

model.predict('input_videos\input_video.mp4')  
