from ultralytics import YOLO
import cv2
import pickle

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)
        
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
    
    # Given a frame, returns bounding boxes with texts on detected players, if there is any
    def draw_bboxes(self, video_frames, ball_detections):
        output_video_frames = []

        for frame, ball_dict in zip(video_frames, ball_detections):
            # Draw bounding boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox

                # Draw a text alongside bounding boxes
                text = f'Ball ID: {track_id}'
                org = (int(bbox[0]), int(bbox[1] - 10))

                cv2.putText(img=frame, text=text, org=org, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.9, color=(0, 0, 255), thickness=2)

                rec = (int(x1), int(y1), int(x2), int(y2))

                frame = cv2.rectangle(img=frame, rec=rec, color=(0, 0, 255), thickness=2)
            
            # Draw new data into actual frame
            output_video_frames.append(frame)

        return output_video_frames