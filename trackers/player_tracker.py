from ultralytics import YOLO
import cv2
import pickle

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames, court_keypoints, read_from_stub=False, stub_path=None):
        player_detections = []

        # Reads from stub if there is any 
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        # Otherwise, creates a new detection stub
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        # Filter frames to keep players only
        player_detections = self.filter_players(court_keypoints, player_detections)

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

    # Filter detected persons and keeps players only
    def filter_players(self, court_keypoints, player_detections):
        # Gets first frame detections
        first_frame = player_detections[0] 

        # A tuple list for every entity and their minimum distances to the keypoints
        distances = []

        # Search for every entity and calculate their distance to the court keypoints
        for track_id, bbox in first_frame.items():

            # Calculates players bbox center
            x1, y1, x2, y2 = bbox
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            p1 = (center_x, center_y)  # player bbox center

            # Calculates the distance between the key point and the player
            min_distance = float('inf')
            for i in range(0, len(court_keypoints), 2):
                p2 = (court_keypoints[i], court_keypoints[i+1])  # key point
                distance = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
                
                if distance < min_distance:
                    min_distance = distance

            distances.append((track_id, min_distance))

        # Get the first two entities with the minimal distance to a key point
        distances.sort(key=lambda x: x[1])

        filtered_players = [distances[0][0], distances[1][0]]

        # Loop over all detections and keep only the filtered players
        filtered_players_detections = []

        for player_dict in player_detections:
            filtered_players_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in filtered_players}
            filtered_players_detections.append(filtered_players_dict)

        return filtered_players_detections
    
    # Given a frame, returns bounding boxes with texts on detected players, if there is any
    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []

        for frame, player_dict in zip(video_frames, player_detections):
            # Draw bounding boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox

                # Draw a text alongside bounding boxes
                text = f'Player {track_id}'
                org = (int(bbox[0]), int(bbox[1] - 10))
                color = (0, 150, 255)

                cv2.putText(frame, text, org, cv2.FONT_HERSHEY_COMPLEX, 0.9, color, 2)

                rec = (int(x1), int(y1)), (int(x2), int(y2))
                
                cv2.rectangle(frame, rec[0], rec[1], color, 2)
            
            # Draw new data into actual frame
            output_video_frames.append(frame)

        return output_video_frames