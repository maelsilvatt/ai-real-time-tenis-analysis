import cv2

# Returns a list of frames given a video
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)

    cap.release()
    return frames

# Saves a list of frames as a video
def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    width = output_video_frames[0].shape[1]
    heigth = output_video_frames[0].shape[0]
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, heigth))

    for frame in output_video_frames:
        out.write(frame)
    
    out.release()