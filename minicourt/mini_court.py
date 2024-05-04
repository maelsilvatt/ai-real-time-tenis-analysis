import cv2

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
        self.drawing_rectangle_height = 450
        self.buffer = 50
        self.padding = 20

        self.set_canvas_background_position(frame)
        self.set_mini_court_position()

    def set_canvas_background_position(self, frame):
        frame = frame.copy()

        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height

        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height
    
    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court

        self.court_width = self.court_end_x - self.court_start_x

    def set_court_keypoints(self):
        keypoints = [0] * 28

        # Point 0
        keypoints[0], keypoints[1] = int(self.court_start_x), int(self.court_start_y)

        # Point 1
        keypoints[2], keypoints[3] = int(self.court_end_x), int(self.court_start_y)

        # Point 2
        keypoints[4] = int(self.court_start_x)
        keypoints[5] = self.court_start_y + meters_to_pixels(HALF_COURT_LINE_HEIGHT * 2)

        # Point 3
        keypoints[6] = keypoints[0] + self.court_width
        keypoints[7] = keypoints[5] 

        # Point 4
        keypoints[8] = keypoints[0] +  self.meters_to_pixels(DOUBLE_ALLEY_DIFFERENCE)
        keypoints[9] = keypoints[1] 

        # Point 5
        keypoints[10] = keypoints[4] + self.meters_to_pixels(DOUBLE_ALLEY_DIFFERENCE)
        keypoints[11] = keypoints[5] 

        # Point 6
        keypoints[12] = keypoints[2] - self.meters_to_pixels(DOUBLE_ALLEY_DIFFERENCE)
        keypoints[13] = keypoints[3] 

        # Point 7
        keypoints[14] = keypoints[6] - self.meters_to_pixels(DOUBLE_ALLEY_DIFFERENCE)
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
        
        # Meters to pixels conversion
        def meters_to_pixels(meters_distance):
            return (meters_distance * DOUBLE_LINE_WIDTH) / court_width


        