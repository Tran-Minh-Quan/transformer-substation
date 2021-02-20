import cv2
from cv2 import imshow, bitwise_and, minAreaRect, boxPoints, drawContours
import numpy as np
from math import acos, cos, sin, radians, degrees
from numpy.linalg import norm
from numpy import pi, sqrt, argmin, array, zeros, ones, dot, sum, append, uint16, rint, count_nonzero, fromiter
from time import time as this_time
from QuanLib import sort2arrays

winName = 'Panel for calibration'
cv2.namedWindow(winName, 0)
cv2.resizeWindow(winName, 800, 600)

color_list = np.random.randint(255, size=(1000, 1, 3)).astype('uint8').tolist()

img = cv2.imread(r'D:\WON\THI_GIAC_MAY\Code\Project\Meter\mr_roi.PNG')


h_img, w_img = img.shape[:2]

TEST_CHANGE_INPUT = 1
TEST_EDGE_DETECTION = 0
TEST_CIRCLE_DETECTION = 1
TEST_MARK_DETECTION = 0
TEST_COLOR_SEGMENTATION = 0
TEST_POINTER_DETECTION = 0
TEST_READ_VALUE = 0


def callback(x):
    pass


if TEST_CHANGE_INPUT:
    cv2.createTrackbar('angle_img', winName, 180, 360, callback)
    cv2.createTrackbar('scale_img', winName, 50, 100, callback)

if TEST_EDGE_DETECTION:
    cv2.createTrackbar('min_thres', winName, 50, 1000, callback)
    cv2.createTrackbar('edgeThres', winName, 100, 1000, callback)

if TEST_CIRCLE_DETECTION:
    cv2.createTrackbar('dp', winName, 1, 200, callback)
    cv2.createTrackbar('minDist', winName, int(h_img//20), 200, callback)
    cv2.createTrackbar('edgeThres', winName, 60, 500, callback)
    cv2.createTrackbar('votes', winName, 40, 500, callback)
    cv2.createTrackbar('minR', winName, 177, 1000, callback)
    cv2.createTrackbar('maxR', winName, 180, 1000, callback)

if TEST_MARK_DETECTION:
    pass

if TEST_COLOR_SEGMENTATION:
    cv2.createTrackbar('lowHue', winName, 88, 179, callback)
    cv2.createTrackbar('lowSat', winName, 0, 255, callback)
    cv2.createTrackbar('lowVal', winName, 0, 255, callback)
    cv2.createTrackbar('highHue', winName, 166, 179, callback)
    cv2.createTrackbar('highSat', winName, 80, 255, callback)
    cv2.createTrackbar('highVal', winName, 106, 255, callback)

if TEST_POINTER_DETECTION:
    cv2.createTrackbar('dis_reso', winName, 1, 10, callback)
    cv2.createTrackbar('angle_reso', winName, 1, 360, callback)
    cv2.createTrackbar('thres', winName, 1, int(sqrt(h_img**2 + w_img**2)//4), callback)
    cv2.createTrackbar('minLen', winName, 103, int(sqrt(h_img**2 + w_img**2)), callback)
    cv2.createTrackbar('maxGap', winName, 9, int(sqrt(h_img**2 + w_img**2)), callback)

start = this_time()
while True:
    img = cv2.imread(r'D:\WON\THI_GIAC_MAY\Code\Project\Meter\mr_roi.PNG')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if TEST_CHANGE_INPUT:
        angle_img = cv2.getTrackbarPos('angle_img', winName) - 180
        scale_img = cv2.getTrackbarPos('scale_img', winName) / 50
        # grab the dimensions of the image
        # determine the center
        h, w = img.shape[:2]
        cX, cY = (w // 2, h // 2)
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle_img, scale_img)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        # perform the actual rotation and return the image
        img = cv2.warpAffine(img, M, (nW, nH))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if TEST_EDGE_DETECTION:
        max_thres = cv2.getTrackbarPos('edgeThres', winName)

        # min_thres = max_thres // 2
        min_thres = cv2.getTrackbarPos('min_thres', winName)
        # min_thres = 0

        polar_image = cv2.linearPolar(img, (254, 256), 192, cv2.WARP_FILL_OUTLIERS)
        img_gray = cv2.cvtColor(polar_image, cv2.COLOR_BGR2GRAY)
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)

        cannied = cv2.Canny(img_gray, min_thres, max_thres, apertureSize=3)
        # cv2.imshow('Edge detection result', cv2.hconcat([img_gray, cannied]))
        cv2.imshow('Edge detection result', cannied)

    if TEST_CIRCLE_DETECTION:
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
        h_img, w_img = img_gray.shape[:2]
        print(f'image width: {w_img}')

        # dp = cv2.getTrackbarPos('dp', winName)
        # minDist = cv2.getTrackbarPos('minDist', winName)
        edgeThres = cv2.getTrackbarPos('edgeThres', winName)
        # votes = cv2.getTrackbarPos('votes', winName)
        # minR = cv2.getTrackbarPos('minR', winName)
        # maxR = cv2.getTrackbarPos('maxR', winName)

        min_thres = edgeThres//2
        canny_edge = cv2.Canny(img_gray, min_thres, edgeThres)
        cv2.imshow('Canny edge detection', canny_edge)

        dp = 1
        minDist = 1
        # edgeThres = 60
        votes = 10
        minR = int(177/424*w_img)
        maxR = int(180/424*w_img)

        circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp, minDist, param1=edgeThres, param2=votes,
                                   minRadius=minR, maxRadius=maxR)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            center = np.array([circles[0][0][0], circles[0][0][1]])
            print(f'center: {center}')
            cv2.circle(img, tuple(center), 2, tuple(color_list[0][0]), 1)
            radius = circles[0][0][2]
            print(f'radius: {radius}')
            cv2.circle(img, tuple(center), radius, tuple(color_list[0][0]), 1)

            # img_float = img.astype(np.float32)
            # polar_img = cv2.linearPolar(img_float, tuple(center), radius + 20, cv2.WARP_FILL_OUTLIERS)
            # polar_img = polar_img.astype(np.uint8)
            # polar_img = cv2.GaussianBlur(polar_img, (7, 7), 0)
            # polar_img = cv2.Canny(polar_img, 30, 60)
            # cv2.imshow("Polar Image", polar_img)

        cv2.imshow('Circle detection result', img)

    if TEST_MARK_DETECTION:
        h_img, w_img = img.shape[:2]

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)

        edges = cv2.Canny(img_gray, 50, 100)
        # cv2.imshow('Canny edges', edges)

        # Tìm đường tròn nhỏ
        circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=60, param2=10,
                                   minRadius=int(177 * scale_img), maxRadius=int(180 * scale_img))
        if circles is None:
            raise Exception("The small circle could not be found")
        else:
            circles = np.uint16(np.around(circles))
            center_1 = np.array([circles[0][0][0], circles[0][0][1]])
            radius_1 = circles[0][0][2]

        show_circle = cv2.circle(img.copy(), tuple(center_1), 1, (0, 255, 0), -1)
        cv2.circle(show_circle, tuple(center_1), radius_1, (0, 255, 0), 1)

        # Tìm đường tròn lớn
        circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=60, param2=10,
                                   minRadius=int(190 * scale_img), maxRadius=int(194 * scale_img))
        if circles is None:
            raise Exception("The big circle could not be found")
        else:
            circles = np.uint16(np.around(circles))
            center_2 = np.array([circles[0][0][0], circles[0][0][1]])
            radius_2 = circles[0][0][2]

        cv2.circle(show_circle, tuple(center_2), 1, (255, 0, 0), -1)
        cv2.circle(show_circle, tuple(center_2), radius_2, (255, 0, 0), 1)

        center = (center_1+center_2)//2
        radius = (radius_1+radius_2)//2
        cv2.circle(show_circle, tuple(center), 1, (0, 0, 255), -1)
        cv2.circle(show_circle, tuple(center), radius, (0, 0, 255), 1)

        # cv2.imshow('circles', show_circle)

        # empty_img = np.zeros((h_img, w_img), np.uint8)
        # only_circle = cv2.circle(empty_img, tuple(center), radius, 255, 2)
        # test = cv2.bitwise_and(only_circle, edges)
        # cv2.imshow('test', test)

        polar_img = cv2.linearPolar(img, tuple(center), radius_2, cv2.WARP_FILL_OUTLIERS).astype(np.uint8)
        polar_reso = 10
        polar_img = cv2.warpPolar(img, dsize=(radius_2, 360*polar_reso), center=tuple(center), maxRadius=radius_2,
                                  flags=cv2.WARP_FILL_OUTLIERS).astype(np.uint8)
        # cv2.imshow("Polar Image", polar_img)

        roi_mark = cv2.Canny(polar_img, 50, 100)[:, radius_1:radius_2]
        cv2.imshow("Canny edges of polar image", roi_mark)

        no_mark_start_angles = array([])
        no_mark_spaces = array([])
        counter = 0
        for idx, row in enumerate(roi_mark):
            if all(row == 0):
                if counter == 0:
                    no_mark_start_angles = append(no_mark_start_angles, idx/polar_reso)
                counter += 1/polar_reso
                if idx == len(roi_mark) - 1:
                    if no_mark_start_angles[0] == 0:
                        no_mark_start_angles[0] = no_mark_start_angles[-1]
                        no_mark_start_angles = no_mark_start_angles[:-1]
                        no_mark_spaces[0] += counter
                    else:
                        no_mark_spaces = append(no_mark_spaces, counter)
            else:
                if counter:
                    no_mark_spaces = append(no_mark_spaces, counter)
                counter = 0

        no_mark_spaces, no_mark_start_angles = sort2arrays(no_mark_spaces, no_mark_start_angles)
        # print(f'no_mark_start_angles: {no_mark_start_angles}')
        # print(f'no_mark_spaces: {no_mark_spaces}')

        zero_mark_angle = no_mark_start_angles[-1] + no_mark_spaces[-1] + 72.7
        print(zero_mark_angle)
        zero_mark = center + array([np.cos(radians(zero_mark_angle)), np.sin(radians(zero_mark_angle))])*radius
        zero_mark_img = cv2.circle(img.copy(), tuple(zero_mark.astype(int)), 1, (0, 255, 0), -1)

        # Show marks, center and circle
        marks_img = img.copy()
        cv2.circle(marks_img, tuple(center), 1, (0, 255, 0), -1)
        cv2.circle(marks_img, tuple(center), radius, (0, 0, 255), 1)
        for i in range(20):
            angle = zero_mark_angle + i*9
            mark = center + array([np.cos(radians(angle)), np.sin(radians(angle))]) * radius
            mark = tuple(uint16(rint(mark)))
            cv2.circle(marks_img, mark, 1, (0, 255, 0), -1)

        cv2.imshow("Marks", marks_img)

    if TEST_COLOR_SEGMENTATION:
        lowHue = cv2.getTrackbarPos('lowHue', winName)
        lowSat = cv2.getTrackbarPos('lowSat', winName)
        lowVal = cv2.getTrackbarPos('lowVal', winName)
        highHue = cv2.getTrackbarPos('highHue', winName)
        highSat = cv2.getTrackbarPos('highSat', winName)
        highVal = cv2.getTrackbarPos('highVal', winName)
        maxVal = cv2.getTrackbarPos('maxVal', winName)
        minVal = cv2.getTrackbarPos('minVal', winName)

        # img = cv2.imread(r'D:\WON\THI_GIAC_MAY\Code\Project\Meter\mr_cropped.PNG')
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img_hsv, (lowHue, lowSat, lowVal), (highHue, highSat, highVal))

        # Range for lower hue red
        # lower_red = np.array([0, 50, 88])
        # upper_red = np.array([29, 255, 255])

        # Range for upper hue red
        # lower_red = np.array([160, 36, 84])
        # upper_red = np.array([179, 255, 255])

        # Range for black
        # lower_red = np.array([88, 0, 0])
        # upper_red = np.array([166, 80, 106])

        imshow('mask', mask)
        imshow('img', img)

    if TEST_POINTER_DETECTION:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
        edges = cv2.Canny(img_gray, 50, 100)
        # cv2.imshow('Canny edges', edges)

        circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=60, param2=10,
                                   minRadius=int(177*scale_img), maxRadius=int(180*scale_img))

        if circles is not None:
            # circles = np.uint16(np.around(circles))
            center = np.array([circles[0][0][0], circles[0][0][1]])
            radius = circles[0][0][2]
            # show_circle = cv2.circle(img.copy(), tuple(rint(center).astype(int)), rint(radius).astype(int), (0, 255, 0), 1)
            # cv2.imshow('circle', show_circle)
            empty_img = np.zeros((img.shape[0], img.shape[1]), np.uint8)
            mask_roi = cv2.circle(empty_img, tuple(rint(center).astype(int)), rint(radius).astype(int), 255, -1)
        else:
            raise Exception('Cannot find any circles')

        mask_roi_bgr = cv2.cvtColor(mask_roi, cv2.COLOR_GRAY2BGR)
        roi = cv2.bitwise_and(img, mask_roi_bgr)
        # cv2.imshow('circle_middle', cv2.circle(img.copy(), (254, 256), 186, 255, 1))

        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        black_mask = cv2.inRange(roi_hsv, array([88, 0, 0]), array([166, 80, 106]))
        black_mask = cv2.dilate(black_mask, ones((5, 5), np.uint8))
        only_black_edges = bitwise_and(black_mask, edges)
        # imshow('only_black_edges', only_black_edges)

        dis_reso = cv2.getTrackbarPos('dis_reso', winName)
        angle_reso = cv2.getTrackbarPos('angle_reso', winName)/180*pi
        try:
            thresh = int(cv2.getTrackbarPos('thres', winName)*scale_img)
        except:
            thresh = cv2.getTrackbarPos('thres', winName)

        lines = cv2.HoughLines(only_black_edges, rho=dis_reso, theta=angle_reso, threshold=thresh)
        if lines is not None:
            # print(f'Number of lines: {len(lines)}')
            edge_pointer = img.copy()
            black_edge_1 = lines[0, 0]
            # print(f'black_edge_1: {black_edge_1}')
            rho_1, theta_1 = black_edge_1
            end_point_1 = array([np.cos(theta_1), np.sin(theta_1)])*rho_1\
                          +10**6*array([-np.sin(theta_1), np.cos(theta_1)])
            end_point_2 = array([np.cos(theta_1), np.sin(theta_1)]) * rho_1\
                          -10**6*array([-np.sin(theta_1), np.cos(theta_1)])
            cv2.line(edge_pointer, tuple(end_point_1.astype(int)), tuple(end_point_2.astype(int)), tuple(color_list[0][0]), 1, cv2.LINE_AA)

            # black_edge_2 = lines[1:4][argmin((lines[1:4, 0, 1]-theta_1)**2)][0]
            # # print(f'black_edge_2: {black_edge_2}')
            # rho_2, theta_2 = black_edge_2
            # a = np.cos(theta_2)
            # b = np.sin(theta_2)
            # x0 = a * rho_2
            # y0 = b * rho_2
            # x1 = int(x0 + 1000 * (-b))
            # y1 = int(y0 + 1000 * a)
            # x2 = int(x0 - 1000 * (-b))
            # y2 = int(y0 - 1000 * a)
            # cv2.line(edge_pointer, (x1, y1), (x2, y2), tuple(color_list[1][0]), 1, cv2.LINE_AA)
        else:
            raise Exception('Cannot find any lines')

        # rho, theta = (black_edge_1 + black_edge_2)/2
        # # print(f'black_pointer: {(black_edge_1 + black_edge_2)/2}')
        # a = np.cos(theta)
        # b = np.sin(theta)
        # x0 = a * rho
        # y0 = b * rho
        # x1 = int(x0 + 1000 * (-b))
        # y1 = int(y0 + 1000 * a)
        # x2 = int(x0 - 1000 * (-b))
        # y2 = int(y0 - 1000 * a)
        # cv2.line(edge_pointer, (x1, y1), (x2, y2), tuple(color_list[2][0]), 1, cv2.LINE_AA)

        # Determine the direction of the pointer

        # Find projected point of black edge 1
        # initial_point = array([np.cos(theta_1), np.sin(theta_1)])*rho_1
        # if all(initial_point == center):
        #     initial_point += 1
        # end_point = initial_point + 10000*array([-np.sin(theta_1), np.cos(theta_1)])
        # ic = center - initial_point
        # ie = end_point - initial_point
        # projected_point = initial_point+ie*(dot(ic, ie)/sum(ie**2))

        # print(f'projected_point: {projected_point}')
        # cv2.circle(edge_pointer, tuple(projected_point.astype(int)), 1, (0, 255, 0), -1)
        # print(tuple(projected_point.astype(int)))
        # cv2.imshow('edge_pointer', edge_pointer)
        # print(f'center: {center}')

        # Find vector of black pointer
        vector_1 = end_point_1 - center
        vector_1 /= norm(vector_1)
        empty_img = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        vector_1_img = cv2.line(empty_img, tuple(rint(center).astype(int)),
                             tuple(rint(center+radius*vector_1).astype(int)), 255, int(38*scale_img))
        vote_1 = count_nonzero(bitwise_and(vector_1_img, only_black_edges))
        # print(f'vote for direction 1: {vote_1}')
        # cv2.imshow('vector 1', vector_1_img)

        vector_2 = end_point_2 - center
        vector_2 /= norm(vector_2)
        empty_img = zeros((img.shape[0], img.shape[1]), np.uint8)
        vector_2_img = cv2.line(empty_img, tuple(rint(center).astype(int)),
                             tuple(rint(center + radius * vector_2).astype(int)), 255, int(38 * scale_img))
        vote_2 = count_nonzero(bitwise_and(vector_2_img, only_black_edges))
        # print(f'vote for direction 2: {vote_2}')
        # cv2.imshow('vector 2', vector_2_img)

        if vote_2 >= vote_1:
            black_pointer = vector_2
        else:
            black_pointer = vector_1

        black_pointer_img = cv2.line(img.copy(), tuple(rint(center).astype(int)),
                            tuple(rint(center + radius * black_pointer).astype(int)), (0, 255,0), 1)
        # cv2.imshow('Vector of pointer', black_pointer_img)

        # Find vector of red pointers

        # Range for lower hue red
        # lower_red = np.array([0, 70, 50])
        # upper_red = np.array([179, 130, 188])
        lower_red = np.array([0, 50, 88])
        upper_red = np.array([29, 255, 255])
        # red_mask_1 = cv2.inRange(roi_hsv, (lowHue, lowSat, lowVal), (highHue, highSat, highVal))
        red_mask_1 = cv2.inRange(roi_hsv, lower_red, upper_red)
        # cv2.imshow('red mask 1', red_mask_1)

        # Range for upper hue red
        lower_red = np.array([160, 36, 84])
        upper_red = np.array([179, 255, 255])
        # red_mask_2 = cv2.inRange(roi_hsv, (lowHue, lowSat, lowVal), (highHue, highSat, highVal))
        red_mask_2 = cv2.inRange(roi_hsv, lower_red, upper_red)
        # cv2.imshow('red mask 2', red_mask_2)

        # The final mask to detect red color
        red_mask = red_mask_1 + red_mask_2
        # kernel = np.ones((5, 5), np.uint8)
        # red_mask = cv2.dilate(red_mask, kernel)
        # cv2.imshow('final red mask', red_mask)
        # red_mask_bgr = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
        # roi_red = cv2.bitwise_and(roi, red_mask_bgr)
        # cv2.imshow('red segmentation', roi_red)

        red_pointer_contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(red_pointer_contours) > 2:
            red_pointer_contours = sorted(red_pointer_contours, key=cv2.contourArea)[-2:]
        if len(red_pointer_contours) == 0:
            raise Exception('Cannot detect any red pointers')

        red_pointer_img = img.copy()
        for contour in red_pointer_contours:
            rect = minAreaRect(contour)
            box = rint(boxPoints(rect)).astype(int)
            # drawContours(red_pointer_img, [box], 0, (0, 0, 255), 2)
            box = sorted(box, key=lambda x: norm(x-box[0]))
            end_points = array([(box[0]+box[1])/2, (box[2]+box[3])/2])
            end_points = sorted(end_points, key=lambda x: norm(center-x))
            red_pointer = end_points[1] - end_points[0]
            red_pointer = red_pointer/norm(red_pointer)
            cv2.line(red_pointer_img, tuple(rint(center).astype(int)),
                                     tuple(rint(center + radius * red_pointer).astype(int)), (0, 255, 0), 1)

        # imshow('red pointer', red_pointer_img)

        # cv2.imshow('pointer edges', edge_pointer)

    # print(f'process time: {this_time() - start:.2f}')
    start = this_time()



    # cv2.imshow('img', img)
    cv2.waitKey(1)