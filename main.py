from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from keypoints_detector import KeyPointsDetector
from minicourt import MiniCourt
import cv2


def main():
    # Read source video
    video_frames =  read_video("input_videos/input_video.mp4")

    # Detect court lines
    keypoints_detector = KeyPointsDetector("models/keypoints_model.pth")
    court_keypoints = keypoints_detector.predict(video_frames[0])

    # Detect players
    player_tracker = PlayerTracker('models/yolov8x.pt')
    player_detections = player_tracker.detect_frames(video_frames, court_keypoints, read_from_stub=True, stub_path="tracker_stubs/player_detections.pkl")
    
    # Detect ball
    ball_tracker = BallTracker('models/yolov5_last.pt')
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/ball_detections.pkl")

    # Draw bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    # Draw mini court and its elements
    mini_court = MiniCourt(video_frames[0])
    output_video_frames = mini_court.draw_mini_court(output_video_frames, player_detections, ball_detections, court_keypoints)

    # Draw frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"frame: {i}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    # Save output video
    save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()