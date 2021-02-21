import cv2
import numpy as np
from math import acos, cos, sin, radians, degrees
from numpy.linalg import norm
from numpy import dot, pi, abs
from QuanLib import angle_clockwise_with_ox, sort2arrays, angle2vectors

# Idea
# Phát hiện tâm kim
# Phát hiện kim
# Phát hiện các vạch và số ứng với mỗi vạch
## Tìm đường tròn đi qua các vạch
## Chỉ xét cung tròn từ chỉ số bé nhất đến chỉ số lớn nhất
## Những điểm màu đen trên cung tròn đó ứng với vạch kim
# Ứng với mỗi tam giác tạo một tia đi từ tâm đường tròn đến tam giác
# Từ góc lệch giữa kim và hai tia gần kim nhất để suy ra giá trị kim

winName = 'Panel for calibration'
cv2.namedWindow(winName, 0)
cv2.resizeWindow(winName, 800, 600)

color_list = np.random.randint(255, size=(1000000, 1, 3)).astype('uint8').tolist()

TEST_EDGE_DETECTION = 0
TEST_CIRCLE_DETECTION = 1
TEST_MARK_DETECTION = 0
TEST_POINTER_DETECTION = 0
TEST_READING_METER = 0

digit_coord_dict = {
    -10: np.array([220, 430]),
    0: np.array([155, 399]),
    30: np.array([89, 308]),
    60: np.array([90, 215]),
    90: np.array([132, 143]),
    115: np.array([241, 95])
    }

def callback(x):
    pass


if TEST_CIRCLE_DETECTION or TEST_MARK_DETECTION or TEST_READING_METER or TEST_POINTER_DETECTION:
    cv2.createTrackbar('dp', winName, 1, 200, callback)
    cv2.createTrackbar('minDist', winName, 115, 200, callback)
    cv2.createTrackbar('edgeThres', winName, 255, 500, callback)
    cv2.createTrackbar('votes', winName, 60, 500, callback)
    cv2.createTrackbar('minR', winName, 151, 1000, callback)
    cv2.createTrackbar('maxR', winName, 207, 1000, callback)

if TEST_POINTER_DETECTION or TEST_READING_METER:
    cv2.createTrackbar('lowHue', winName, 17, 179, callback)
    cv2.createTrackbar('lowSat', winName, 75, 255, callback)
    cv2.createTrackbar('lowVal', winName, 118, 255, callback)
    cv2.createTrackbar('highHue', winName, 35, 179, callback)
    cv2.createTrackbar('highSat', winName, 132, 255, callback)
    cv2.createTrackbar('highVal', winName, 174, 255, callback)

if TEST_EDGE_DETECTION or TEST_CIRCLE_DETECTION or TEST_MARK_DETECTION or TEST_READING_METER or TEST_POINTER_DETECTION:
    # cv2.createTrackbar('min_thres', winName, 1, 1000, callback)
    cv2.createTrackbar('edgeThres', winName, 255, 1000, callback)

if TEST_MARK_DETECTION or TEST_READING_METER:
    pass

while True:
    # img = cv2.imread(r'D:\WON\THI_GIAC_MAY\Data\Transformer_station\Meter\oil_level_1_ROI_smaller.PNG')
    img = cv2.imread(r'D:\WON\THI_GIAC_MAY\Data\Meter\oil_level_2_ROI.PNG')
    h_img, w_img = img.shape[:2]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if TEST_EDGE_DETECTION or TEST_CIRCLE_DETECTION or TEST_MARK_DETECTION or TEST_READING_METER or TEST_POINTER_DETECTION:
        max_thres = cv2.getTrackbarPos('edgeThres', winName)
        min_thres = max_thres // 2
        # min_thres = cv2.getTrackbarPos('min_thres', winName)
        cannied = cv2.Canny(img_gray, min_thres, max_thres)
        # cv2.imshow('Edge detection result', cv2.hconcat([img_gray, cannied]))

    if TEST_CIRCLE_DETECTION or TEST_MARK_DETECTION or TEST_READING_METER or TEST_POINTER_DETECTION:
        dp = cv2.getTrackbarPos('dp', winName)
        minDist = cv2.getTrackbarPos('minDist', winName)
        edgeThres = cv2.getTrackbarPos('edgeThres', winName)
        votes = cv2.getTrackbarPos('votes', winName)
        minR = cv2.getTrackbarPos('minR', winName)
        maxR = cv2.getTrackbarPos('maxR', winName)

        circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp, minDist, param1=edgeThres, param2=votes,
                                   minRadius=minR, maxRadius=maxR)

        print(circles)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for index, i in enumerate(circles[0, :]):
                center = np.array([i[0], i[1]])
                cv2.circle(img, tuple(center), 2, tuple(color_list[index][0]), 2)
                radius = i[2]
                cv2.circle(img, tuple(center), radius, tuple(color_list[index][0]), 2)
        cv2.imshow('Circle detection result', img)

    if TEST_MARK_DETECTION or TEST_POINTER_DETECTION or TEST_READING_METER:
        # for i in digit_coord_dict.values():
        #     cv2.circle(img, tuple(i), 2, (0, 255, 0), -1)
        ox = np.array([1, 0])
        start_angle = angle_clockwise_with_ox(digit_coord_dict[-10] - center) - 10
        end_angle = angle_clockwise_with_ox(digit_coord_dict[115] - center) + 10
        if end_angle < start_angle:
            end_angle += 360
            print('Correct your camera location')
        # Hiển thị đường cong trên ảnh gốc
        # cv2.ellipse(img, center, (radius, radius), 0, start_angle, end_angle, (0, 255, 0), 1)
        # cv2.imshow('curve', img)

        start_vector = np.array([cos(radians(start_angle)), sin(radians(start_angle))])
        empty_img = np.zeros((h_img, w_img), np.uint8)
        radius = radius - 7
        arc = cv2.ellipse(img=empty_img.copy(), center=tuple(center), axes=(radius , radius), angle=0,
                          startAngle=start_angle, endAngle=end_angle, color=255, thickness=2)
        # arc_ = cv2.ellipse(img=empty_img.copy(), center=tuple(center), axes=(radius, radius), angle=0,
        #                    startAngle=88, endAngle=278, color=255, thickness=2)
        # cv2.imshow('smaller circle', arc_)
        # cv2.imshow('arc', arc)
        # cv2.imshow('cannied', cannied)
        edges_mark = cv2.bitwise_and(arc, cannied)
        # cv2.imshow('edges_mark', edges_mark)

        points_mark = cv2.findNonZero(edges_mark)
        edge_angle_array = np.sort(np.array([angle2vectors(point[0] - center, start_vector) for point in points_mark]))
        lower_edge = edge_angle_array[0]
        upper_edge = edge_angle_array[0]
        mark_angle_array = np.array([])
        for angle in edge_angle_array[1:]:
            angle_diff = angle - lower_edge
            if 0 < angle_diff <= 10:
                upper_edge = angle
            elif angle_diff > 10:
                mark_angle_array = np.append(mark_angle_array, (upper_edge + lower_edge) / 2)
                lower_edge = angle
        mark_angle_array = np.append(mark_angle_array, (upper_edge + lower_edge) / 2)
        mark_vectors = np.array([[cos(radians(start_angle + angle)), sin(radians(start_angle + angle))]
                                 for angle in mark_angle_array])
        points_mark = (center + mark_vectors * radius).astype(int)
        for point in points_mark:
            # mark_points_img = cv2.circle(img, tuple(point), 2, (0, 255, 0), -1)
            # cv2.imshow('mark_points_img', mark_points_img)
            cv2.line(img, tuple(center), tuple(point), (0, 255, 0), 2)
        # cv2.imshow('Mark detection result', img)

    if TEST_POINTER_DETECTION or TEST_READING_METER:
        lowHue = cv2.getTrackbarPos('lowHue', winName)
        lowSat = cv2.getTrackbarPos('lowSat', winName)
        lowVal = cv2.getTrackbarPos('lowVal', winName)
        highHue = cv2.getTrackbarPos('highHue', winName)
        highSat = cv2.getTrackbarPos('highSat', winName)
        highVal = cv2.getTrackbarPos('highVal', winName)

        segmented = cv2.inRange(img_hsv, (lowHue, lowSat, lowVal), (highHue, highSat, highVal))
        opened = cv2.morphologyEx(segmented, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        # dilated = cv2.dilate(opened, np.ones((3, 3), np.uint8))

        contours, hier = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        vx, vy = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)[:2]
        pointer_vector = (np.array([vx.item(), vy.item()]) * radius).astype(int)
        pointer = cv2.line(opened, tuple(center - pointer_vector), tuple(center + pointer_vector), 255, 1)
        intersection_point = cv2.findNonZero(cv2.bitwise_and(pointer, arc))[0, 0]
        cv2.line(img, tuple(center), tuple(intersection_point), (255, 255, 255), 2)
        # print([angle2vectors(i, pointer_vector) for i in mark_vectors])
        # cv2.drawContours(img, contours, 0, (0, 255, 0), 3)
        cv2.imshow('segmented', segmented)
        cv2.imshow('opened', opened)
        # cv2.imshow('closed', closed)
        # cv2.imshow('dilated', dilated)
        cv2.imshow('Pointer detection result', pointer)

    if TEST_READING_METER:
        pointer_vector = intersection_point - center
        pointer_angle = angle2vectors(pointer_vector, start_vector)
        print(f'pointer angle: {pointer_angle}')
        print(f'mark_angle_array: {mark_angle_array}')
        digit_array = np.array([-10, 0, 30, 60, 90, 115])
        # print(digit_array[np.argsort(pointer_angle - mark_angle_array)])
        pointer_and_mark_angles = np.abs(pointer_angle - mark_angle_array)
        sorted_indexes = np.argsort(pointer_and_mark_angles)
        lower_digit, upper_digit = digit_array[sorted_indexes][:2]
        angle_with_lower, angle_with_upper = pointer_and_mark_angles[sorted_indexes][:2]
        current_value = lower_digit + angle_with_lower / (angle_with_lower+angle_with_upper) * (upper_digit-lower_digit)
        print(f'angle_with_lower: {angle_with_lower}')
        print(f'angle_with_upper: {angle_with_upper}')
        print(f'current_value: {current_value:.2f}')
        cv2.imshow('Reading result', img)

    cv2.waitKey(1)

