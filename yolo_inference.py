from ultralytics import YOLO
import torch

# using our trained model
model = YOLO('models/yolov5_last.pt')

# setting inference device as CUDA (if available)
device = 'cuda' if torch.cuda.is_available else 'cpu'

# store and predict
conf = 0.2 # this confidence value gave us the best results

result = model.predict('input_videos/input_video.mp4', device=device, save=True, conf=conf)