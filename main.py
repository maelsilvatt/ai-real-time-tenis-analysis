from utils.video_utils import read_video, save_video
from trackers.player_tracker import PlayerTracker


def main():
    # Read source video
    input_video_path = "input_videos/input_video.mp4"
    video_frames =  read_video(input_video_path)

    # Detect players
    player_tracker = PlayerTracker(model_path='yolov8x')
    player_detection = player_tracker.detect_frames(video_frames)

    # Draw bouding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detection)

    # Save output video
    save_video(video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()