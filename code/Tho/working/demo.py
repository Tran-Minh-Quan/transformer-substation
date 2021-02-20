import numpy as np
import cv2
from gauge_hand_lib import gauge_hand
from gauge_hand_lib import temp_meter
from scipy.interpolate import interp1d
import time


def find_pointing(edges, quantity, thresh_low, min_length, max_gap, step):
    inc_thresh = True
    while inc_thresh:
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, thresh_low,
                                minLineLength=min_length, maxLineGap=max_gap)
        if lines.shape[0] <= quantity:
            inc_thresh = False
        else:
            thresh_low += step
    return lines


if __name__ == "__main__":
    start = time.time()
    img_origin = cv2.imread("./Project/Tho/data/tho_2b.jpg")  # Read image
    img_resized = cv2.resize(img_origin,  # Resize image for faster processing time
                             (np.uint(img_origin.shape[1]/img_origin.shape[0]*500), 500))
    img = cv2.GaussianBlur(img_resized, (7, 7), 0)  # Blur image to reduce noise
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert image to HSV
    mask1 = cv2.inRange(img_hsv, (0, 30, 20), (15, 255, 255))  # Create mask for red color
    mask2 = cv2.inRange(img_hsv, (175, 30, 20), (180, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)
    img = cv2.bitwise_and(img, img, mask=mask)  # Bitwise mask with img
    edges = cv2.Canny(img, 60, 180)  # Find edges in img\
    start = time.time()
    lines = find_pointing(edges, quantity=50, thresh_low=20,
                          min_length=25, max_gap=20, step=5)
    _, gauge_angle, img_show = gauge_hand.find_pointing(edges, img_resized, quantity=20,
                                                        thresh_low=20, min_length=25, max_gap=20,
                                                        step=5, min_angle=5)
    stop = time.time()
    print(stop-start)
    for i in range(lines.shape[0]):
        cv2.line(img_resized, tuple(np.array([lines[i, 0, 0], lines[i, 0, 1]])),
                 tuple(np.array([lines[i, 0, 2], lines[i, 0, 3]])), (0, 0, 255), 1)
    # Interpolate to find value of meter
    temp_interp1d = interp1d(temp_meter.angle_data, temp_meter.value_data)
    read_val = temp_interp1d(gauge_angle)
    cv2.putText(img_show, "Value: %0.4f degree" % read_val, (10, np.int(3/4*500)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow("Threshholed_img", img)
    cv2.imshow("Edges", edges)
    cv2.imshow("Detected segments", img_resized)
    cv2.imshow("Result", img_show)
    cv2.waitKey(0)
