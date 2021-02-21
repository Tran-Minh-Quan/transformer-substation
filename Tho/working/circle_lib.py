import cv2
import numpy as np
import math


class CircleDetect:
    def __init__(self, low_canny, high_canny, step_size, hough_param, min_radius, max_radius):
        self.NO_ERROR = 0
        self.INVALID_INPUT_ERROR = 1
        self.NON_CIRCLE_ERROR = 2   # Circle undetectable
        self.MULTIPLE_CIRCLES_ERROR = 3
        self.low_canny = low_canny
        self.high_canny = high_canny
        self.step_size = step_size
        self.hough_param = hough_param  # This value should varies between 20-70 (30 is best for some cases)
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.first_detect = 1
        self.redetect = 1
        self.last_canny_param = 450

    def calculate(self, img, top_left, bot_right, extended_ratio, mode):
        try:
            img_out = img.copy()
        except Exception:
            self.redetect = 1
            return [np.array([-1, -1, -1]), np.zeros([500, 500]), self.INVALID_INPUT_ERROR]
        width_box = bot_right[1] - top_left[1]
        height_box = bot_right[0] - top_left[0]
        # Check for invalid input
        if width_box <= 0 or height_box <= 0 \
                or max(bot_right[0], top_left[0]) > img.shape[0] or max(bot_right[1], top_left[1]) > img.shape[1]:
            return [np.array([-1, -1, -1]), img_out, self.INVALID_INPUT_ERROR]
        # Calculate x axis extended
        x_axis_extended = [max(0, int(top_left[0] - extended_ratio * width_box)),
                           min(img.shape[0], int(bot_right[0] + extended_ratio * width_box))]
        # Calculate y axis extended
        y_axis_extended = [max(0, int(top_left[1] - extended_ratio * height_box)),
                           min(img.shape[1], int(bot_right[1] + extended_ratio * height_box))]
        # Crop image
        crop_img = img[x_axis_extended[0]: x_axis_extended[1], y_axis_extended[0]:y_axis_extended[1]]
        cv2.rectangle(img_out, (y_axis_extended[0], x_axis_extended[0]), (y_axis_extended[1], x_axis_extended[1]),
                      (255, 0, 0), 2)
        # cv2.imwrite("img_test.jpg", crop_img)
        # Grayscale image
        if crop_img.ndim > 2:
            gray_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)   # Convert for color image
        else:
            gray_img = crop_img     # Gray scale image
        # Binary search algorithm to find RIGHTMOST Canny parameter
        if self.redetect == 1:
            rm_left = self.low_canny
            rm_right = self.high_canny
        else:
            rm_left = max(self.last_canny_param - 2000, 0)
            rm_right = self.last_canny_param + 2000
        canny_param = rightmost_canny_param_search(gray_img, rm_left, rm_right,
                                                   self.step_size, self.hough_param, self.min_radius, self.max_radius)
        # Detect circle with determined Canny parameter
        circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, 100, param1=canny_param, param2=self.hough_param,
                                   minRadius=self.min_radius, maxRadius=self.max_radius)
        # Return error if circle is undetectable
        if circles is None or circles[0][0][2] == 0:
            self.redetect = 1
            return [np.array([-1, -1, -1]), img_out, self.NON_CIRCLE_ERROR]
        circles_round = np.round(circles[0, :]).astype("int")
        # Mark detected circle in image
        for (y, x, r) in circles_round:
            cv2.circle(img_out, (y + y_axis_extended[0], x + x_axis_extended[0]), r, (0, 255, 0), 4)
            cv2.rectangle(img_out, (y - 2 + y_axis_extended[0], x - 2 + x_axis_extended[0]),
                          (y + 2 + y_axis_extended[0], x + 2 + x_axis_extended[0]), (0, 128, 255), -1)
        if circles.shape[1] > 1:
            self.redetect = 1
            return [np.array([-1, -1, -1]), img_out, self.MULTIPLE_CIRCLES_ERROR]
        radius = circles[0][0][2]
        x_coord = np.round(circles[0][0][0] + y_axis_extended[0])
        y_coord = np.round(circles[0][0][1] + x_axis_extended[0])
        # distance = self.slope * 1/radius_pixel + self.intercept
        self.last_canny_param = canny_param
        self.redetect = 0
        return [np.array([x_coord, y_coord, radius]), img_out, self.NO_ERROR]


def rightmost_canny_param_search(gray_img, rm_left, rm_right, step_size, hough_param, min_radius, max_radius):
    step = 0
    while rm_left < rm_right:
        step += 1
        canny_param = math.floor((rm_right + rm_left) / (2 * step_size)) * step_size
        if canny_param <= 0:
            rm_left = rm_right
            return rm_right
        circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, 100,
                                   param1=canny_param, param2=hough_param, minRadius=min_radius, maxRadius=max_radius)
        if circles is None or circles[0][0][2] == 0:
            rm_right = canny_param
        else:
            rm_left = canny_param + step_size
    # print("Iteration step taken: %d" % step)
    return rm_left - step_size
