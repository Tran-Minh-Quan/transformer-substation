import cv2
from cv2 import imread, imshow, waitKey, cvtColor, bitwise_and, minAreaRect, boxPoints, drawContours, line, circle
from cv2 import GaussianBlur, HoughCircles, createTrackbar, getTrackbarPos, adaptiveThreshold, findContours, Canny
from cv2 import boundingRect, rectangle, contourArea, getRotationMatrix2D, warpAffine, vconcat, hconcat, arcLength
from cv2 import putText, bilateralFilter, resize, inRange, dilate, threshold
import numpy as np
from math import acos, cos, sin, radians, degrees
from numpy.linalg import norm
from numpy import pi, sqrt, argmin, array, zeros, ones, dot, sum, append, uint8, uint16, rint, count_nonzero, fromiter
from numpy import nonzero, unique, cross
from time import time as this_time
from QuanLib import sort2arrays, angle_clockwise_with_ox
from bisect import bisect
from os import listdir
from os.path import join, splitext
from timeit import timeit


def read_MR_gauge(img, w_bbox):
    # if w_bbox < 289:
    #     return None, None, None
    h_img, w_img = img.shape[:2]
    img_gray = cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_gray = GaussianBlur(img_gray, (3, 3), 0)
    RATIO = w_bbox/386

    # Test Canny edges
    # edges = Canny(img_gray, 30, 60)
    # imshow('edges', edges)

    # Detect circle
    circles = HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=60, param2=10,
                           minRadius=round(177*RATIO), maxRadius=round(180*RATIO))
    if circles is None:
        raise Exception("Cannot find the small circle")
    else:
        circles = rint(circles).astype(int)
        center_1 = array([circles[0][0][0], circles[0][0][1]])
        radius_1 = int(circles[0][0][2])

    # Detect large circle
    circles = HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=60, param2=10,
                           minRadius=round(191*RATIO), maxRadius=round(194*RATIO))
    if circles is None:
        raise Exception("Cannot find the large circle")
    else:
        circles = rint(circles).astype(int)
        center_2 = array([circles[0][0][0], circles[0][0][1]])
        radius_2 = int(circles[0][0][2])

    # Detect middle circle
    center = rint((center_1 + center_2)/2).astype(int)
    radius = round((radius_1 + radius_2)/2)

    # Detect pointer

    # Get ROI of pointers
    empty_img = zeros((h_img, w_img), np.uint8)
    mask_roi = circle(empty_img, tuple(center_1), radius_1, 255, -1)
    mask_roi_bgr = cvtColor(mask_roi, cv2.COLOR_GRAY2BGR)
    roi = bitwise_and(img, mask_roi_bgr)

    # Extract the edge that belong to the black pointer
    roi_hsv = cvtColor(roi, cv2.COLOR_BGR2HSV)
    black_mask = inRange(roi_hsv, array([88, 0, 0]), array([166, 80, 106]))
    black_mask = dilate(black_mask, ones((5, 5), np.uint8))
    edges = Canny(img_gray, 50, 100)
    only_black_edges = bitwise_and(black_mask, edges)

    lines = cv2.HoughLines(only_black_edges, rho=1, theta=0.5/180*pi, threshold=1)

    black_edge_1 = lines[0, 0]
    rho_1, theta_1 = black_edge_1
    end_point_1 = array([np.cos(theta_1), np.sin(theta_1)])*rho_1+10**6*array([-np.sin(theta_1), np.cos(theta_1)])
    end_point_2 = array([np.cos(theta_1), np.sin(theta_1)])*rho_1-10**6*array([-np.sin(theta_1), np.cos(theta_1)])

    # Detect vector of black pointer
    vector_1 = end_point_1 - center
    vector_1 /= norm(vector_1)
    empty_img = zeros((img.shape[0], img.shape[1]), np.uint8)
    vector_1_img = cv2.line(empty_img, tuple(rint(center).astype(int)),
                            tuple(rint(center + radius * vector_1).astype(int)), 255, int(38*RATIO))
    vote_1 = count_nonzero(bitwise_and(vector_1_img, only_black_edges))

    vector_2 = end_point_2 - center
    vector_2 /= norm(vector_2)
    empty_img = zeros((img.shape[0], img.shape[1]), np.uint8)
    vector_2_img = line(empty_img, tuple(rint(center).astype(int)),
                        tuple(rint(center + radius * vector_2).astype(int)), 255, int(38*RATIO))
    vote_2 = count_nonzero(bitwise_and(vector_2_img, only_black_edges))

    if vote_2 >= vote_1:
        black_pointer = vector_2
    else:
        black_pointer = vector_1

    # Calculate the angle of angle_black_pointer
    angle_black_pointer = angle_clockwise_with_ox(black_pointer)

    # Detect red pointers
    lower_red = array([0, 50, 88])
    upper_red = array([29, 255, 255])
    red_mask_1 = cv2.inRange(roi_hsv, lower_red, upper_red)

    lower_red = array([160, 36, 84])
    upper_red = array([179, 255, 255])
    red_mask_2 = inRange(roi_hsv, lower_red, upper_red)

    red_mask = red_mask_1 + red_mask_2

    red_pointer_contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if red_pointer_contours is not None:
        # Filter 2 contour with the largest area
        if len(red_pointer_contours) > 2:
            red_pointer_contours = sorted(red_pointer_contours, key=cv2.contourArea)[-2:]

        # Red pointer 1
        rect = minAreaRect(red_pointer_contours[0])
        box = rint(boxPoints(rect)).astype(int)
        box = sorted(box, key=lambda x: norm(x - box[0]))
        end_points = array([(box[0] + box[1]) / 2, (box[2] + box[3]) / 2])
        end_points = sorted(end_points, key=lambda x: norm(center - x))
        red_pointer_1 = end_points[1] - end_points[0]
        red_pointer_1 = red_pointer_1 / norm(red_pointer_1)

        # Calculate the angle of angle_red_pointer_1
        angle_red_pointer_1 = angle_clockwise_with_ox(red_pointer_1)

        # Red pointer 2
        if len(red_pointer_contours) == 2:
            rect = minAreaRect(red_pointer_contours[1])
            box = rint(boxPoints(rect)).astype(int)
            box = sorted(box, key=lambda x: norm(x - box[0]))
            end_points = array([(box[0] + box[1]) / 2, (box[2] + box[3]) / 2])
            end_points = sorted(end_points, key=lambda x: norm(center - x))
            red_pointer_2 = end_points[1] - end_points[0]
            red_pointer_2 = red_pointer_2 / norm(red_pointer_2)

            # Calculate the angle of angle_red_pointer_2
            angle_red_pointer_2 = angle_clockwise_with_ox(red_pointer_2)
        else:
            angle_red_pointer_2 = None
    else:
        angle_red_pointer_1, angle_red_pointer_2 = None, None

    # Detect marks
    polar_reso = 10
    polar_img = cv2.warpPolar(img_gray, dsize=(radius_2, 360 * polar_reso), center=tuple(center), maxRadius=radius_2,
                              flags=cv2.WARP_FILL_OUTLIERS).astype(np.uint8)

    roi_mark = polar_img[:, radius_1:radius_2]
    roi_mark = GaussianBlur(roi_mark, (15, 15), 0)
    roi_mark_binary = adaptiveThreshold(roi_mark, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 801, 0)

    num_labels, labels_im = cv2.connectedComponents(roi_mark_binary[:, round((radius_2-radius_1)/2)])
    if labels_im[0] != 0 and labels_im[-1] != 0:
        labels_im[labels_im == (num_labels-1)] = 1
        num_labels -= 1
    indices, counts = unique(labels_im, return_index=True, return_counts=True)[1:]
    counts, indices = sort2arrays(counts, indices)
    y_no_mark = indices[-2] + 1

    roi_mark_shifted = vconcat([roi_mark_binary[y_no_mark:], roi_mark_binary[:y_no_mark]])
    edge_mark = Canny(roi_mark_shifted, 50, 100)
    angles_mark = (nonzero(edge_mark[:, round((radius_2-radius_1)/2)])[0]+y_no_mark)/polar_reso
    angles_mark[angles_mark > 360] -= 360

    values = [str(x) for x in range(-7, 9)] + ['9A', '9B', '9C'] + [str(x) for x in range(10, 27)]
    values[-1] = None

    milestones = [(round(x, 1), y) for x, y in zip(angles_mark, values)]
    milestones.sort()

    # Calculate indicating value
    idx = bisect(milestones, (angle_black_pointer, ))
    value_black_pointer = milestones[idx-1][1]
    value_higher_red_pointer, value_lower_red_pointer = None, None

    if angle_red_pointer_1 is not None:
        idx = bisect(milestones, (angle_red_pointer_1, ))
        value = milestones[idx-1][1]
        if cross(black_pointer, red_pointer_1) >= 0:
            value_higher_red_pointer = value
        else:
            value_lower_red_pointer = value
    if angle_red_pointer_2 is not None:
        idx = bisect(milestones, (angle_red_pointer_2, ))
        value = milestones[idx-1][1]
        if cross(black_pointer, red_pointer_2) >= 0:
            value_higher_red_pointer = value
        else:
            value_lower_red_pointer = value

    # Show ROI of mark image, for check
    test_binary = np.zeros((360, 1), dtype=uint8)
    test_gray = np.zeros((360, 1), dtype=uint8)
    for i in range(polar_reso):
        test_binary = hconcat([test_binary, roi_mark_binary[360*i:360*(i+1)]])
        test_gray = hconcat([test_gray, roi_mark[360*i:360*(i+1)]])
    cv2.imshow('test binary', test_binary)
    cv2.imshow('test gray', test_gray)

    # Display
    line(img, tuple(rint(center).astype(int)), tuple(rint(center+radius*black_pointer).astype(int)),
             (255, 255, 255), 1)
    line(img, tuple(rint(center).astype(int)), tuple(rint(center + radius * red_pointer_1).astype(int)),
             (255, 255, 255), 1)
    line(img, tuple(rint(center).astype(int)), tuple(rint(center + radius * red_pointer_2).astype(int)),
             (255, 255, 255), 1)
    for angle in angles_mark:
        coord = (center + array([np.cos(radians(angle)), np.sin(radians(angle))]) * radius).astype(int)
        line(img, tuple(center), tuple(coord), (255, 0, 255), 1)
    circle(img, tuple(center_1), radius_1, (0, 255, 0), 1)
    circle(img, tuple(center_2), radius_2, (0, 255, 0), 1)
    circle(img, tuple(center), radius, (255, 255, 0), 1)

    return value_black_pointer, value_lower_red_pointer, value_higher_red_pointer


def read_oil_level_gauge(img, w_bbox):
    value = None

    img_gray = cvtColor(img, cv2.COLOR_BGR2GRAY)
    ksize = getTrackbarPos('ksize', winName)
    c = getTrackbarPos('c', winName) - 4
    # try:
        # segment_black = adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ksize, c)
    segment_black = threshold(img_gray, 255, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    imshow('segment pointer', segment_black)
    imshow('gray image', img_gray)
    # except:
    #     pass
    return value


# Test

def callback(x):
    pass
winName = 'Panel for calibration'
cv2.namedWindow(winName, 0)
cv2.resizeWindow(winName, 800, 600)
cv2.createTrackbar('ksize', winName, 651, 3001, callback)
cv2.createTrackbar('c', winName, 0, 255, callback)

# dir = r'D:\WON\THI_GIAC_MAY\Data\Test\MR_gauge\crop'
dir = r'D:\WON\THI_GIAC_MAY\Data\Test\oil_level_gauge'
filenames = listdir(dir)
directories = [join(dir, filename) for filename in filenames]


# while 1:
for directory in directories:
    if not directory.endswith('PNG') and not directory.endswith('JPG'):
        continue
        # else:
        #     break
    frame = imread(directory)
    print('filename:', directory.split('\\')[-1])
    start = this_time()


    bbox_dir = splitext(directory)[0] + '.txt'
    bbox = list(map(float, open(bbox_dir, 'r').read().splitlines()[0].split(' ')))

    # Get bounding box, for testing
    top_left = rint(array([(bbox[1] - bbox[3]/2)*frame.shape[1], (bbox[2] - bbox[4]/2)*frame.shape[0]])).astype(int)
    bottom_right = rint(array([(bbox[1] + bbox[3]/2)*frame.shape[1], (bbox[2] + bbox[4]/2)*frame.shape[0]])).astype(int)
    w, h = (bottom_right[0]-top_left[0]+1, bottom_right[1]-top_left[1]+1)

    # Expand bounding box
    RATIO_EXPAND = 1/100
    top_left_expanded = top_left - rint(w * RATIO_EXPAND).astype(uint8)
    top_left_expanded[top_left_expanded < 0] = 0
    bottom_right_expanded = bottom_right + rint(w * RATIO_EXPAND).astype(uint8)
    gauge_img = frame[top_left_expanded[1]:bottom_right_expanded[1]+1, top_left_expanded[0]:bottom_right_expanded[0]+1]

    # Get indicating value
    name = 'oil_level_gauge'
    if name == 'mr_gauge':
        value_1, value_2, value_3 = read_MR_gauge(gauge_img, int(w))
    elif name == 'oil_level_gauge':
        value = read_oil_level_gauge(gauge_img, int(w))

    # Resize image to display
    frame = resize(frame, (800, round(frame.shape[0]/frame.shape[1]*800)))

    # Display values
    img_blank = zeros((200, frame.shape[1], 3), dtype=uint8)
    name = 'oil_level_gauge'
    if name == 'mr_gauge':
        text_1 = 'Black pointer:'
        text_2 = 'Lower red pointer:'
        text_3 = 'Higher red pointer:'

        putText(img_blank, text_1, (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        putText(img_blank, text_2, (20, 108), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        putText(img_blank, text_3, (20, 170), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

        putText(img_blank, value_1, (400, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
        putText(img_blank, value_2, (400, 108), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
        putText(img_blank, value_3, (400, 170), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
    elif name == 'oil_level_gauge':
        text = 'Value:'
        putText(img_blank, text, (20, 108), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        putText(img_blank, value, (400, 108), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)


    img_display = vconcat([img_blank, frame])
    # imshow('Result', img_display)
    print('Process time: ', round((this_time() - start)*1000), 'ms')
    start = this_time()
    waitKey()