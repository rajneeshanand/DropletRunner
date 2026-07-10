
#Detects blue corner markers on the board. Used to estimate board boundaries and tilt angles.


import cv2
import numpy as np


class MarkerDetector:
    def __init__(self):
        #marker HSV range (measured: H=111-116, S=255, V=71-165)
        self.lower_blue = np.array([100, 150, 50])
        self.upper_blue = np.array([130, 255, 255])

        #expected 4 corners in order: TL, TR, BR, BL
        self.corners = None

    def detect(self, image):

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if 30 < area < 3000:
                M = cv2.moments(c)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    candidates.append((cx, cy))

        if len(candidates) < 4:
            return self.corners  #return last known if not enough found

        #sort into TL, TR, BR, BL order
        #sort by y to get top 2 and bottom 2
        candidates.sort(key=lambda p: p[1])
        top_two = sorted(candidates[:2], key=lambda p: p[0])
        bottom_two = sorted(candidates[2:4], key=lambda p: p[0])

        self.corners = [
            top_two[0],      # Top-left
            top_two[1],      # Top-right
            bottom_two[1],   # Bottom-right
            bottom_two[0],   # Bottom-left
        ]

        return self.corners

    def estimate_tilt(self, corners):
        if corners is None:
            return (0.0, 0.0)

        tl, tr, br, bl = corners

        #Alpha (pitch) — difference in y between left and right sides
        left_y = (tl[1] + bl[1]) / 2
        right_y = (tr[1] + br[1]) / 2
        alpha = (right_y - left_y) / 10.0  # Scale factor, tune later

        #Beta (roll) — difference in y between top and bottom
        top_y = (tl[1] + tr[1]) / 2
        bottom_y = (bl[1] + br[1]) / 2
        board_height = bottom_y - top_y

        top_x = (tl[0] + tr[0]) / 2
        bottom_x = (bl[0] + br[0]) / 2
        beta = (bottom_x - top_x) / 10.0  #scale factor, tune later

        return (alpha, beta)