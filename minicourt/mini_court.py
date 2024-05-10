import cv2
import numpy as np

# Court dimensions
SINGLE_LINE_WIDTH = 8.23
DOUBLE_LINE_WIDTH = 10.97
HALF_COURT_LINE_HEIGHT = 11.88
SERVICE_LINE_WIDTH = 6.4
DOUBLE_ALLEY_DIFF = 1.37
NO_MANS_LAND_HEIGHT = 5.48

# Players heights
PLAYER_1_HEIGHT_METERS = 1.88
PLAYER_2_HEIGHT_METERS = 1.91

class MiniCourt():
    def __init__(self, frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50
        self.padding = 20

        self.set_background_position(frame)
        self.set_mini_court_position()
        self.set_mini_court_keypoints()
        self.set_mini_court_lines()
        self.set_background_position(frame)

    # Sets mini court background position
    def set_background_position(self, frame):
        frame = frame.copy()

        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height

        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height
    
    # Sets mini court position on a frame
    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding
        self.court_start_y = self.start_y + self.padding
        
        self.court_end_x = self.end_x - self.padding
        self.court_end_y = self.end_y - self.padding

        self.court_width = self.court_end_x - self.court_start_x

    # Converts meters to pixels for inner computations
    def meters_to_pixels(self, meters, ref1=self.court_width, ref2=DOUBLE_LINE_WIDTH):
        return (meters * ref1) / ref2

    # Sets every key point coordinates
    def set_mini_court_keypoints(self):
        keypoints = [0] * 28

        # Point 0
        keypoints[0], keypoints[1] = int(self.court_start_x), int(self.court_start_y)

        # Point 1
        keypoints[2], keypoints[3] = int(self.court_end_x), int(self.court_start_y)

        # Point 2
        keypoints[4] = int(self.court_start_x)
        keypoints[5] = self.court_start_y + self.meters_to_pixels(HALF_COURT_LINE_HEIGHT * 2)

        # Point 3
        keypoints[6] = keypoints[0] + self.court_width
        keypoints[7] = keypoints[5] 

        # Point 4
        keypoints[8] = keypoints[0] +  self.meters_to_pixels(DOUBLE_ALLEY_DIFF)
        keypoints[9] = keypoints[1] 

        # Point 5
        keypoints[10] = keypoints[4] + self.meters_to_pixels(DOUBLE_ALLEY_DIFF)
        keypoints[11] = keypoints[5] 

        # Point 6
        keypoints[12] = keypoints[2] - self.meters_to_pixels(DOUBLE_ALLEY_DIFF)
        keypoints[13] = keypoints[3] 

        # Point 7
        keypoints[14] = keypoints[6] - self.meters_to_pixels(DOUBLE_ALLEY_DIFF)
        keypoints[15] = keypoints[7] 

        # Point 8
        keypoints[16] = keypoints[8] 
        keypoints[17] = keypoints[9] + self.meters_to_pixels(NO_MANS_LAND_HEIGHT)

        # Point 9
        keypoints[18] = keypoints[16] + self.meters_to_pixels(SINGLE_LINE_WIDTH)
        keypoints[19] = keypoints[17] 

        # Point 10
        keypoints[20] = keypoints[10] 
        keypoints[21] = keypoints[11] - self.meters_to_pixels(NO_MANS_LAND_HEIGHT)

        # Point 11
        keypoints[22] = keypoints[20] +  self.meters_to_pixels(SINGLE_LINE_WIDTH)
        keypoints[23] = keypoints[21] 

        # Point 12
        keypoints[24] = int((keypoints[16] + keypoints[18]) / 2)
        keypoints[25] = keypoints[17] 

        # Point 13
        keypoints[26] = int((keypoints[20] + keypoints[22]) / 2)
        keypoints[27] = keypoints[21] 

        self.keypoints = keypoints

    # Sets mini court lines to be drawn
    def set_mini_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6, 7),
            (1, 3),

            (0, 1),
            (8, 9),
            (10, 11),
            (2, 3)
        ]
    
    # Draws mini court keypoints
    def draw_keypoints(self, frame):
        for i in range(0, len(self.keypoints), 2):
            x = int(self.keypoints[i])
            y = int(self.keypoints[i+1])

            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        return frame

    # Draws mini court lines 
    def draw_mini_court_lines(self, frame):
        for line in self.lines:
            start = (int(self.keypoints[line[0]*2]), int(self.keypoints[line[0]*2+1]))
            end = (int(self.keypoints[line[1]*2]), int(self.keypoints[line[1]*2+1]))

            cv2.line(frame, start, end, (0, 0, 0), 2)
        
        # Draw net
        net_start = (self.keypoints[0], int((self.keypoints[1] + self.keypoints[5]) / 2))
        net_end = (self.keypoints[2], int((self.keypoints[1] + self.keypoints[5]) / 2))

        cv2.line(frame, net_start, net_end, (255, 0, 0), 2)

        return frame

    # Draws mini court 
    def draw_mini_court(self, frames):
        output_frames = []

        for frame in frames:
            # Draw background rectangle for every frame
            shapes = np.zeros_like(frame, np.uint8)
            cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), -1)

            output_frame = frame.copy()
            alpha = 0.5
            mask = shapes.astype(bool)
            output_frame[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

            # Draw minicourt keypoints
            output_frame = self.draw_keypoints(output_frame)
            
            # Draw minicourt lines
            output_frame = self.draw_mini_court_lines(output_frame)

            # Save frame
            output_frames.append(output_frame)
        
        return output_frames

    # Mini court position getter
    def get_mini_court_start_point(self):
        return (self.court_start_x, self.court_start_y)

    # Mini court width getter
    def get_mini_court_width(self):
        return self.court_width
    
    # Mini court keypoints getter
    def get_mini_court_keypoints(self):
        return self.keypoints
    
    # Gets players and ball bounding boxes coordinates
    def get_bbox_coordinates(self, player_bboxes, ball_bboxes, real_court_keypoints):
        player_heights = {
            1: PLAYER_1_HEIGHT_METERS,
            2: PLAYER_2_HEIGHT_METERS
        }

        output_player_bbox = []
        output_ball_bbox = []


        for frame_idx, player_bbox in enumerate(player_bboxes):
            # Gets ball position
            ball_bbox = ball_bboxes[frame_idx]
            x1, y2, x2, y2 = ball_bbox
            ball_position = (int((x1 + x2) /2), int((y1 + y2) / 2))

            # Gets players foot position
            get_foot_position = lambda bbox: (int((bbox[0] + bbox[2]) / 2), bbox[3])
            players_foot_position = list(map(get_foot_position, player_bboxes))

            # Gets ball to players distances
            ball_to_players_distance = []

            measure_distance = lambda p1, p2: ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
            ball_to_players_distance = list(map(calculate_distance, ball_position, players_foot_position))
            
            ball_closest_player = min(player_bboxes.keys(), ball_to_players_distance)

            output_player_bbox_dict = {}
            for player_id, bbox in player_bbox.items():
                # Gets player foot position
                x1, y1, x2, y2 = bbox
                foot_position = int((x1 + x2) /2, y2)

                # Get the closest key point to the player
                ref_keypoints = [0, 2, 12, 13]  # referential keypoints

                closest_key_point = float('inf')
                closest_key_point_idx = ref_keypoints[0]

                for idx in ref_keypoints:
                    key_point = (real_court_keypoints(idx * 2), real_court_keypoints(idx * 2 + 1))
                    distance = abs(foot_position[1] - key_point[1])

                    if distance < closest_key_point:
                        closest_key_point = distance
                        closest_key_point_idx = idx

                # Gets real court closest keypoint to the player
                real_court_closest_key_point = (real_court_keypoints[closest_key_point_idx*2], 
                                                real_court_keypoints[closest_key_point_idx*2+1])

                # Get player height in pixels
                min_frame_idx = max(0, frame_idx - 20)
                max_frame_idx = min(len(player_bboxes), frame_idx + 50)

                # Gets player bbox height in pixels for each frame, and then the max height
                bboxes_heights_in_pixels = [(player_bbox[i][3] - player_bbox[i][1]) for i in range(min_frame_idx, max_frame_idx)]
                player_heigth_in_pixels = max(bboxes_heights_in_pixels)

                # Gets mini court coordinates
                with object_position, closest_key_point as p1, p2:
                    key_point_x_distance_in_pixels, key_point_y_distance_in_pixels = abs(p1[0] - p2[0]), abs(p1[1] - p2[1])

                # Converts pixel distance to meters, doing the opposite operation
                key_point_x_distance_in_meters = meters_to_pixels(key_point_x_distance_in_pixels, ref1=player_height_in_pixels, ref2=player_height_in_meters)
                key_point_y_distance_in_meters = meters_to_pixels(key_point_y_distance_in_pixels, ref1=player_height_in_pixels, ref2=player_height_in_meters)

                # Converts to mini court coordinates
                mini_court_x_distance_in_pixels = self.meters_to_pixels(key_point_x_distance_in_meters)
                mini_court_y_distance_in_pixels = self.meters_to_pixels(key_point_y_distance_in_meters)

                # Gets player position on mini court
                closest_mini_court_key_point = (self.keypoints[closest_key_point_idx*2], self.keypoints[closest_key_point_idx*2+1])
                mini_court_player_position = (closest_mini_court_key_point[0] + mini_court_x_distance_in_pixels,
                                              closest_mini_court_key_point[1] + mini_court_y_distance_in_pixels)

                output_player_bbox_dict[player_id] = mini_court_player_position