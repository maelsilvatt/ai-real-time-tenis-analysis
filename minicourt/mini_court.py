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
        self.set_court_keypoints()
        self.set_background_position(frame)

    def set_background_position(self, frame):
        frame = frame.copy()

        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height

        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height
    
    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding
        self.court_start_y = self.start_y + self.padding
        
        self.court_end_x = self.end_x - self.padding
        self.court_end_y = self.end_y - self.padding

        self.court_width = self.court_end_x - self.court_start_x

    def meters_to_pixels(self, meters):
        return (meters * self.court_width) / DOUBLE_LINE_WIDTH

    def set_court_keypoints(self):
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

    def set_court_lines(self):
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
    
    def draw_keypoints(self, frame):
        for i in range(0, len(self.keypoints), 2):
            x = int(self.keypoints[i])
            y = int(self.keypoints[i+1])

            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        return frame

    def draw_minicourt(self, frames):
        output_frames = []

        for frame in frames:
            # Draw background rectangle for every frame
            shapes = np.zeros_like(frame, np.uint8)
            cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), -1)

            output_frame = frame.copy()
            alpha = 0.5
            mask = shapes.astype(bool)
            output_frame[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

            # Draw court keypoints
            output_frame = self.draw_keypoints(output_frame)

            output_frames.append(output_frame)
        
        return output_frames

    