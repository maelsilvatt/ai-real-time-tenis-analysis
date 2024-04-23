from utils.video_utils import read_video, save_video
from trackers.player_tracker import PlayerTracker
from trackers.ball_tracker import BallTracker


def main():
    # Read source video
    input_video_path = "input_videos/input_video.mp4"
    video_frames =  read_video(input_video_path)

    # Detect players
    player_tracker = PlayerTracker(model_path='yolov8x')
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/player_detections.pkl")
    
    # Detect ball
    ball_tracker = BallTracker(model_path='models/yolov5_last.pt')
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=False, stub_path="tracker_stubs/ball_detections.pkl")

    # Draw bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = player_tracker.draw_bboxes(video_frames, ball_detections)

    # Save output video
    save_video(video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()