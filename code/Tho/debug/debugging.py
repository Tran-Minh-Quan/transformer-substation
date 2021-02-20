import numpy as np
import cv2
from tho_gauge_lib import get_temp_meter

if __name__ == "__main__":
    img_origin = cv2.imread("./Project/Tho/data/tho_2b.jpg")  # Read image
    bounding_box = np.array([100, 100, 1000, 1500])
    val_1, val_2, img_1, img_2, edges_1, edges_2, all_1, all_2 = get_temp_meter(img=img_origin,
                                                                                resize_ratio=0.5,
                                                                                bounding_box=bounding_box)
    print(val_1)
    print(val_2)
    cv2.imshow("img_1", img_1)
    cv2.imshow("img_2", img_2)
    # cv2.imshow("edges_1", edges_1)
    # cv2.imshow("edges_2", edges_2)
    # cv2.imshow("all_1", all_1)
    # cv2.imshow("all_2", all_2)
    cv2.waitKey(0)
