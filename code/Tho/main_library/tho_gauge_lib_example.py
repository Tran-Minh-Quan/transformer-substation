import numpy as np
import cv2
from tho_gauge_lib import get_temp_meter

if __name__ == "__main__":
    img_origin = cv2.imread("./Project/Tho/data/tho_2a.jpg")  # Read image
    bound_box = np.array([100, 100, 1000, 1500])
    val_1, val_2, img_1, img_2 = get_temp_meter(img=img_origin, resize_ratio=0.5,
                                                bounding_box=bound_box)
    print(val_1)
    print(val_2)
    cv2.imshow("img_1", img_1)
    cv2.imshow("img_2", img_2)