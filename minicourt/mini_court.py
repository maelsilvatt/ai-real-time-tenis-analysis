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
    def meters_to_pixels(self, meters, ref1=None, ref2=DOUBLE_LINE_WIDTH):
        if ref1 is None:
            ref1 = self.court_width

        return (meters * ref1) / ref2

    # Sets every key point coordinates
    def set_mini_court_keypoints(self):
        keypoints = [0] * 28

        meters_to_pixels =  lambda  value : value * 3779.527559

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

            cv2.circle(frame, (x, y), 5, (0, 0, 0), -1)

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

    # Draws mini court elements
    def draw_mini_court(self, frames, player_detections, ball_detections, real_court_keypoints):
        output_frames = []

        # Gets main elements positions
        player_positions, ball_positions = self.get_element_positions(player_detections, ball_detections, real_court_keypoints)
        frame_idx = 0

        for frame in frames:
            # Draw background rectangle for every frame
            shapes = np.zeros_like(frame, np.uint8)
            cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), -1)

            output_frame = frame.copy()
            alpha = 0.5
            mask = shapes.astype(bool)
            output_frame[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

            # Draw mini court keypoints
            output_frame = self.draw_keypoints(output_frame)
            
            # Draw mini court lines
            output_frame = self.draw_mini_court_lines(output_frame)

            # Draw mini court elements
            output_frame = self.draw_mini_court_elements(output_frame, player_positions[frame_idx], ball_positions[frame_idx])

            # Save frame
            output_frames.append(output_frame)

            # Update frame counter
            frame_idx += 1
        
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

    # Given original elements bboxes, returns their mini court coordinates
    def get_mini_court_coords(self, 
                              element_position, 
                              closest_key_point, 
                              closest_key_point_idx, 
                              element_height_in_pixels, 
                              element_height_in_meters):  
        
        # Gets mini court coordinates
        abs_distance = lambda p1, p2: (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
        key_point_x_distance_in_pixels, key_point_y_distance_in_pixels = abs_distance(element_position, closest_key_point)

        # Converts pixel distance to meters
        meters_to_pixels =  lambda  value : value * 3779.527559
        pixel_to_meters = lambda value : value * 0.0003

        key_point_x_distance_in_meters = pixel_to_meters(key_point_x_distance_in_pixels)
        key_point_y_distance_in_meters = pixel_to_meters(key_point_y_distance_in_pixels)

        # Converts to mini court coordinates
        mini_court_x_distance_in_pixels = meters_to_pixels(key_point_x_distance_in_meters)
        mini_court_y_distance_in_pixels = meters_to_pixels(key_point_y_distance_in_meters)

        # Gets player position on mini court
        closest_mini_court_key_point = (self.keypoints[closest_key_point_idx*2], self.keypoints[closest_key_point_idx*2+1])
        mini_court_element_position = (closest_mini_court_key_point[0] + mini_court_x_distance_in_pixels,
                                        closest_mini_court_key_point[1] + mini_court_y_distance_in_pixels)
        
        return mini_court_element_position

    # Given an element position, returns its closest key point
    def get_closest_key_point_idx(self, element_position, real_keypoints):
        ref_keypoints = [0, 2, 12, 13]  # referential keypoints

        closest_key_point = float('inf')
        closest_key_point_idx = ref_keypoints[0]

        for idx in ref_keypoints:
            key_point = (real_keypoints[idx * 2], real_keypoints[idx * 2 + 1])
            distance = abs(element_position[1] - key_point[1])

            if distance < closest_key_point:
                closest_key_point = distance
                closest_key_point_idx = idx
        
        return closest_key_point_idx
    
    # Gets players and ball positions to draw them
    def get_element_positions(self, player_bboxes, ball_bboxes, real_keypoints):
        player_heights = {
            1: PLAYER_1_HEIGHT_METERS,
            2: PLAYER_2_HEIGHT_METERS
        }

        player_positions = []
        ball_positions = []

        for frame_idx, player_bbox in enumerate(player_bboxes):
            # Gets ball position
            ball_bbox = ball_bboxes[frame_idx][1]
            get_bbox_center = lambda bbox: (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3])))
            ball_position = get_bbox_center(ball_bbox)

            measure_distance = lambda p1, p2: ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
            
            ball_closest_player = min(player_bboxes[frame_idx].keys(), key=lambda x: measure_distance(ball_position, get_bbox_center(player_bbox[x])))

            output_player_bbox_dict = {}
            for player_id, bbox in player_bbox.items():
                # Gets player foot position
                x1, _, x2, y2 = bbox
                foot_position = ((x1 + x2) /2, y2)

                # Gets real court closest keypoint to the player
                closest_key_point_idx = self.get_closest_key_point_idx(foot_position, real_keypoints)
                closest_key_point = (real_keypoints[closest_key_point_idx*2], 
                                     real_keypoints[closest_key_point_idx*2+1])

                # Get player height in pixels
                min_frame_idx = max(0, frame_idx - 20)
                max_frame_idx = min(len(player_bboxes), frame_idx + 50)

                # Gets player bbox height in pixels for each frame, and then the max height
                get_bbox_height = lambda bbox: bbox[3]-bbox[1]
                bboxes_heights_in_pixels = [get_bbox_height(player_bboxes[i][player_id]) for i in range(min_frame_idx, max_frame_idx)]
                player_height_in_pixels = max(bboxes_heights_in_pixels)

                mini_court_player_position = self.get_mini_court_coords(foot_position,
                                                                               closest_key_point,
                                                                               closest_key_point_idx,
                                                                               player_height_in_pixels,
                                                                               player_heights[player_id])
                
                output_player_bbox_dict[player_id] = mini_court_player_position

                if ball_closest_player == player_id:
                    closest_key_point_idx = self.get_closest_key_point_idx(ball_position, real_keypoints)
                    closest_key_point = (real_keypoints[closest_key_point_idx*2],
                                             real_keypoints[closest_key_point_idx*2+1])
                    
                    mini_court_player_position = self.get_mini_court_coords(ball_position,
                                                                            closest_key_point,
                                                                            closest_key_point_idx,
                                                                            player_height_in_pixels,
                                                                            player_heights[player_id])


                    #ball_positions.append({1:mini_court_player_position})
                    ball_positions.append(mini_court_player_position)

            player_positions.append(output_player_bbox_dict)

        return player_positions, ball_positions
    
    # Draws each moving element on the mini court
    def draw_mini_court_elements(self, frame, players_positions, ball_positions):
        # Gets each player position values
        p1_positions = players_positions[1]
        p2_positions = players_positions[2]

        # Draws player 1
        cv2.circle(frame, (int(p1_positions[0]), int(p1_positions[1])), 5, (255, 0, 0), -1)

        # Draws player 2
        cv2.circle(frame, (int(p2_positions[0]), int(p2_positions[1])), 5, (0, 0, 255), -1)

        # Draws ball
        cv2.circle(frame, (int(ball_positions[0]), int(ball_positions[1])), 5, (0, 255, 0), -1)

        return frame