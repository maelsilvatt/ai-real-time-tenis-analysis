from ultralytics import YOLO
import pandas as pd
import pickle
import cv2

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    # Given a source video, detects and return every ball positions
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
                ball_detections = self.interpolate_ball_detections(ball_detections)
            return ball_detections

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)
        
        ball_detections = self.interpolate_ball_detections(ball_detections)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        
        return ball_detections

    # Given a frame, detects and returns ball position
    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.2)[0]

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result

        return ball_dict
    
    # Interpolate intermediate ball positions to get a better position estimative
    def interpolate_ball_detections(self, ball_detections):
        # Obtains all values from keys equal to 1
        ball_detections = [x.get(1, []) for x in ball_detections]

        # To proceed, convert ball positions list into a dataframe
        df_ball_detections = pd.DataFrame(ball_detections, columns=['x1','y1','x2','y2'])

        # Use interpolate to fill missing values between estimated ball positions
        df_ball_detections = df_ball_detections.interpolate() 
        df_ball_detections = df_ball_detections.bfill()

        # Convert the dataframe to a list again
        ball_detections = [{1:x} for x in df_ball_detections.to_numpy().tolist()]

        return ball_detections

    # Given ball detections, returns the frames where the ball was hit
    def get_ball_shot_frames(self, ball_detections):
        ball_positions = [x.get(1, []) for x in ball_detections]

        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Creates a column to set if the ball was hit in a particular frame
        df_ball_positions['ball_hit'] = 0

        # Gets ball center
        df_ball_positions['mean_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2

        # Computes the mean value of 5 ball positions in y axis
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        
    # Given a frame, returns bounding boxes with texts on detected players, if there is any
    def draw_bboxes(self, video_frames, ball_detections):
        output_video_frames = []

        for frame, ball_dict in zip(video_frames, ball_detections):
            # Draw bounding boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox

                # Draw a text alongside bounding boxes
                text = f'Tennis Ball'
                org = (int(bbox[0]), int(bbox[1] - 10))
                color = (152, 251, 152)

                cv2.putText(frame, text, org, cv2.FONT_HERSHEY_COMPLEX, 0.9, color, 2)

                rec = (int(x1), int(y1)), (int(x2), int(y2))
                cv2.rectangle(frame, rec[0], rec[1], color, 2)
            
            # Draw new data into actual frame
            output_video_frames.append(frame)

        return output_video_frames