from ultralytics import YOLO
import cv2
import pickle

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        return player_detections

    # Given a frame, detects players and returns their position
    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_class_id = box.cls.tolist()[0]
            object_class_name = id_name_dict[object_class_id]

            if object_class_name == "person":
                player_dict[track_id] = result

        return player_dict
    
    # Given a frame, returns bounding boxes with texts on detected players, if there is any
    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []

        for frame, player_dict in zip(video_frames, player_detections):
            # Draw bounding boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox

                # Draw a text alongside bounding boxes
                text = f'Player ID: {track_id}'
                org = (int(bbox[0]), int(bbox[1] - 10))

                cv2.putText(img=frame, text=text, org=org, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.9, color=(0, 0, 255), thickness=2)

                cv2.rectangle(img=frame, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 0, 255), thickness=2)
            
            # Draw new data into actual frame
            output_video_frames.append(frame)

        return output_video_frames