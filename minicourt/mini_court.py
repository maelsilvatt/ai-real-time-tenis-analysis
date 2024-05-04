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
        keypoints[5] = self.court_start_y + meters_to_pixels(HALF_COURT_LINE_HEIGHT * 2, DOUBLE_LINE_WIDTH, court_width)

        # Pixel to meters conversion
        def pixel_to_meters(pixel_distance, ref_height_meters, ref_height_pixels):
            return (pixel_distance * ref_height_meters) / ref_height_pixels
        
        # Meters to pixels conversion
        def meters_to_pixels(meters_distance, ref_height_meters, ref_height_pixels):
            return (meters_distance * ref_height_pixels) / ref_height_meters


        