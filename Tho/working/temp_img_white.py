import cv2
from gauge_hand_lib import gauge_hand
import numpy as np
from gauge_hand_lib import temp_meter
from scipy.interpolate import interp1d


def nothing(x):
    pass


if __name__ == "__main__":
    img_origin = cv2.imread("./Project/Tho/data/tho_1a.jpg")
    img_resized = cv2.resize(img_origin,  # Resize image for faster processing time
                             (np.uint(img_origin.shape[1]/img_origin.shape[0]*500), 500))
    img = cv2.GaussianBlur(img_resized, (3, 3), 0)
    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    low_thresh = np.array([15, 100, 28])
    high_thresh = np.array([36, 255, 255])
    mask = cv2.inRange(img_hls, low_thresh, high_thresh)
    img_threshold = cv2.bitwise_and(img, img, mask=mask)
    kernel = np.ones((3, 3), np.uint8)
    img_threshold = cv2.morphologyEx(img_threshold, cv2.MORPH_OPEN, kernel)
    edges = cv2.Canny(img_threshold, 60, 180)
    _, gauge_angle, img_show = gauge_hand.find_pointing(edges, img_resized, quantity=20,
                                                        thresh_low=20, min_length=15, max_gap=10,
                                                        step=5, min_angle=5, max_angle=30)
    temp_interp1d = interp1d(temp_meter.angle_data, temp_meter.value_data)
    # read_val = temp_interp1d(gauge_angle)
    read_val = 0
    cv2.putText(img_show, "Value: %0.4f degree" % read_val, (10, np.int(3/4*500)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow("Win 1", img_threshold)
    cv2.imshow("Win 2", edges)
    cv2.imshow("Win 3", img_show)
    cv2.waitKey(0)

    # mask = cv2.inRange(l_channel, 150, 255)
    # res = cv2.bitwise_and(img, img, mask=mask)
    # mask = cv2.inRange(img_hls, np.array([0, 130, 0]), np.array([255, 255, 255]))
    # cv2.namedWindow("Win 1")
    # cv2.createTrackbar("thresh_h_low", "Win 1", 15, 255, nothing)
    # cv2.createTrackbar("thresh_l_low", "Win 1", 100, 255, nothing)
    # cv2.createTrackbar("thresh_s_low", "Win 1", 28, 255, nothing)
    # cv2.createTrackbar("thresh_h_high", "Win 1", 36, 255, nothing)
    # cv2.createTrackbar("thresh_l_high", "Win 1", 255, 255, nothing)
    # cv2.createTrackbar("thresh_s_high", "Win 1", 255, 255, nothing)
    # while True:
    #     thresh_l_low = cv2.getTrackbarPos("thresh_l_low", "Win 1")
    #     thresh_h_low = cv2.getTrackbarPos("thresh_h_low", "Win 1")
    #     thresh_s_low = cv2.getTrackbarPos("thresh_s_low", "Win 1")
    #     thresh_l_high = cv2.getTrackbarPos("thresh_l_high", "Win 1")
    #     thresh_h_high = cv2.getTrackbarPos("thresh_h_high", "Win 1")
    #     thresh_s_high = cv2.getTrackbarPos("thresh_s_high", "Win 1")
    #     low_thresh = np.array([15, 100, 28])
    #     high_thresh = np.array([36, 255, 255])
    #     # mask = cv2.inRange(img_hls, np.array([thresh_h_low, thresh_l_low, thresh_s_low]),
    #     #                    np.array([thresh_h_high, thresh_l_high, thresh_s_high]))
    #     mask = cv2.inRange(img_hls, low_thresh, high_thresh)
    #     # mask = cv2.inRange(img_hsv, lower_white, upper_white)
    #     res = cv2.bitwise_and(img, img, mask=mask)
    #     kernel = np.ones((3, 3), np.uint8)
    #     res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
    #     edges = cv2.Canny(res, 60, 180)
    #     _, gauge_angle, img_show = gauge_hand.find_pointing(edges, img_resized, quantity=20,
    #                                                         thresh_low=20, min_length=25, max_gap=15,
    #                                                         step=5, min_angle=5)
    #     cv2.imshow("Win 1", mask)
    #     cv2.imshow("Win 2", img_show)
    #     cv2.imshow("Win 3", edges)
    #     cv2.waitKey(100)
