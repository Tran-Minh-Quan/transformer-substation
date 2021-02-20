import cv2
import numpy as np
from circle_lib import CircleDetect
from gauge_hand_lib import gauge_hand
from gauge_hand_lib import press_meter
from scipy.interpolate import interp1d


def display():
    for i in range(lines.shape[0]):
        cv2.line(img_circle, tuple(lines[i, 0]+top_left),
                 tuple(lines[i, 1]+top_left), (0, 0, 255), 1)
    cv2.rectangle(img_circle, tuple(top_left), tuple(bot_right), (255, 0, 0), 1)
    cv2.putText(img_show, "Predicted value: %0.4f" % value, (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(img_show, "True value: %0.4f" % 6.8, (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite("pointer_example.png", img_show)
    cv2.imshow("Img original", img_circle)
    cv2.imshow("Img autocrop", img_show)
    cv2.waitKey(0)


if __name__ == "__main__":
    img_origin = cv2.imread("./Project/Tho/data/Nguyen.jpg")  # Read image
    img = cv2.resize(img_origin, (750, 500))  # Resize image for faster processing time

    circle_machine = CircleDetect(1, 1000, 1, 20, 0, 0)  # Create circle autodetection instance
    # Find strongest circle in image
    circle_params, img_circle, error = circle_machine.calculate(img, (0, 0), (500, 750), 0, 1)
    # Crop part of image based on detected circle
    crop_ratio = np.sqrt(2)/2 * 8/5
    top_left = np.array([max(circle_params[0]-crop_ratio*circle_params[2], 0),
                        max(circle_params[1]-crop_ratio*circle_params[2], 0)], dtype=np.int)
    bot_right = np.array([min(circle_params[0]+crop_ratio*circle_params[2], img.shape[1]),
                         min(circle_params[1]+crop_ratio*circle_params[2], img.shape[0])], dtype=np.int)
    img_cropped = img[top_left[1]:bot_right[1], top_left[0]:bot_right[0]]
    # Find edge in cropped part of img
    edges = cv2.Canny(img_cropped, 80, 160)
    # Find gauge hand in img
    lines, gauge_angle, img_show = gauge_hand.find_pointing(edges, img_cropped, quantity=20, thresh_low=15,
                                                            min_length=25, max_gap=10, step=10, min_angle=3)
    # Interpolate to find value of meter
    temp_interp1d = interp1d(press_meter.angle_data, press_meter.value_data)
    value = temp_interp1d(gauge_angle)
    # Visualize result
    display()
