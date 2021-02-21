import cv2
import numpy as np
from math import acos
from numpy.linalg import norm
from numpy import dot, pi

def angle_clockwise_with_ox(vector):
    ox = np.array([1, 0])
    if vector[1] >= 0:
        return acos(dot(vector, ox)/norm(vector)/norm(ox)) / pi * 180
    else:
        return 360 - acos(dot(vector, ox)/norm(vector)/norm(ox)) / pi * 180


def angle2vectors(vector1, vector2):
    return acos(dot(vector1, vector2) / norm(vector1) / norm(vector2)) / pi * 180

def read_color_value(img_path):
    while True:
        img = cv2.imread(img_path)
        top_left_x, top_left_y, w, h = cv2.selectROI(img)
        roi = img[top_left_y:top_left_y + h - 1, top_left_x:top_left_x + w - 1].copy()
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        print(roi_hsv)
        print(f'hsv min threshold: ({np.amin(roi_hsv[:, :, 0])}, {np.amin(roi_hsv[:, :, 1])}, {np.amin(roi_hsv[:, :, 2])})')
        print(f'hsv max threshold: ({np.amax(roi_hsv[:, :, 0])}, {np.amax(roi_hsv[:, :, 1])}, {np.amax(roi_hsv[:, :, 2])})')

        cv2.imshow('roi', roi)
        cv2.waitKey(1)

def get_bbox_coordinates(img_path):
    while True:
        img = cv2.imread(img_path)
        top_left_x, top_left_y, w, h = cv2.selectROI(img)
        print(f'Center coordinate x: {int(top_left_x + w/2 - 1)}')
        print(f'Center coordinate y: {int(top_left_y + h/2 - 1)}')

def sort2arrays(sorted_array, related_array):
    sorted_array, related_array = zip(*sorted(zip(sorted_array, related_array)))
    sorted_array, related_array = (np.array(t) for t in zip(*sorted(zip(sorted_array, related_array))))
    return sorted_array, related_array


# a = np.array([3,4,1,2])
# b = np.array(['three', 'five', 'two', 'one'])
# a, b = sort2arrays(a, b)
# print(a)
# print(b)

# read_color_value(r'D:\WON\THI_GIAC_MAY\Data\Transformer_station\Meter\oil_level_1_ROI_smaller.PNG')

# get_bbox_coordinates(r'D:\WON\THI_GIAC_MAY\Data\Transformer_station\Meter\oil_level_1_ROI_smaller.PNG')