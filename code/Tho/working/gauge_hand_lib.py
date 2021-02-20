import numpy as np
import cv2
from shapely.geometry import LineString


class gauge_hand:
    # This method is used to find angle between 2 vectors
    def calc_angle_vec(vec_1, vec_2):
        vec_1_unit = vec_1 / np.linalg.norm(vec_1)
        vec_2_unit = vec_2 / np.linalg.norm(vec_2)
        if np.cross(vec_1, vec_2_unit) <= 0:
            angle = np.arccos(np.dot(vec_1_unit, vec_2_unit))
        else:
            angle = 2*np.pi - np.arccos(np.dot(vec_1_unit, vec_2_unit))
        return angle

    # This method is used to find angle between 2 segments
    def calc_angle_seg(seg_1, seg_2):
        vec_1 = np.array([seg_1[0]-seg_1[2], seg_1[1]-seg_1[3]])
        vec_2 = np.array([seg_2[0]-seg_2[2], seg_2[1]-seg_2[3]])
        angle = gauge_hand.calc_angle_vec(vec_1, vec_2)
        return angle

    # This method is used to find gauge hand in image
    # quantity: the amount of line remain for auto threshold of cv2.HoughLinesP
    # thresh_low: the lowest possible threshold for auto threshold of cv2.HoughLinesP
    # min_length: shortest segment not rejected in cv2.HoughLinesP
    # max_gap: the maximum gap between points to link them
    # step: the step size for auto threshold of cv2.HoughLinesP
    # min_angle: the minimum angle for gauge pair not rejected
    # Returns:
    # + gauge_hands: the segments of gauge hand
    # + avg_angle: the angle of gauge hand
    # + img_out: the image after gauge hand is detected
    # Note: img and edges should be cropped and denoise in advanced before passing to function
    def find_pointing(edges, img, quantity, thresh_low, min_length, max_gap, step, min_angle, max_angle):
        img_out = img.copy()
        inc_thresh = True
        while inc_thresh:
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, thresh_low,
                                    minLineLength=min_length, maxLineGap=max_gap)
            if lines.shape[0] <= quantity:
                inc_thresh = False
            else:
                thresh_low += step
        cross_response_min = 1e9
        line_1_min = lines[0, 0, :]
        line_2_min = lines[1, 0, :]
        min_angle_rad = min_angle*np.pi/180
        max_angle_rad = max_angle*np.pi/180
        for i in range(lines.shape[0]):
            for j in range(i+1, lines.shape[0]):
                len_1 = np.sqrt(np.square(lines[i, 0, 0]-lines[i, 0, 2]) + np.square(lines[i, 0, 1]-lines[i, 0, 3]))
                len_2 = np.sqrt(np.square(lines[j, 0, 0]-lines[j, 0, 2]) + np.square(lines[j, 0, 1]-lines[j, 0, 3]))
                line_1 = lines[i, 0, :]
                line_2 = lines[j, 0, :]
                seg_1 = LineString([tuple(line_1[0:2]), tuple(line_1[2:4])])
                seg_2 = LineString([tuple(line_2[0:2]), tuple(line_2[2:4])])
                if seg_1.intersects(seg_2):
                    segments = np.array([np.sum((line_1[0:2] - line_2[0:2])**2),
                                        np.sum((line_1[0:2] - line_2[2:4])**2),
                                        np.sum((line_1[2:4] - line_2[0:2])**2),
                                        np.sum((line_1[2:4] - line_2[2:4])**2)])
                    segment_index = np.argmin(segments)
                    tip_points = np.array([(segment_index - segment_index % 2)/2, segment_index % 2], dtype=np.int)
                    float_points = np.ones(2, dtype=np.int) - tip_points
                    tip_dist_sqr = segments[2*tip_points[0]+tip_points[1]]
                    float_dist_sqr = segments[2*float_points[0]+float_points[1]]
                    cross_response = 2*np.sqrt(tip_dist_sqr) + np.sqrt(float_dist_sqr) - 0.5*(len_1 + len_2)
                    if cross_response_min > cross_response:
                        angle = gauge_hand.calc_angle_seg(lines[i, 0], lines[j, 0])
                        if (angle > min_angle_rad and
                            abs(angle-np.pi) > min_angle_rad and
                                2*np.pi - angle > min_angle_rad and
                                angle < max_angle_rad and
                                abs(angle-np.pi) > max_angle_rad and
                                2*np.pi - angle > max_angle_rad):
                            cross_response_min = cross_response
                            line_1_min = np.array([line_1[2*tip_points[0]:2*tip_points[0]+2],
                                                  line_1[2*float_points[0]:2*float_points[0]+2]])
                            line_2_min = np.array([line_2[2*tip_points[1]:2*tip_points[1]+2],
                                                  line_2[2*float_points[1]:2*float_points[1]+2]])
                else:
                    vec_1 = line_1[0:2] - line_1[2:4]
                    vec_2 = line_2[0:2] - line_2[2:4]
                    if np.cross(vec_1, vec_2) != 0:
                        t1, t2 = np.linalg.solve(np.array([[vec_1[0], -vec_2[0]], [vec_1[1], -vec_2[1]]]),
                                                 np.array([line_1[0] - line_2[0], line_1[1] - line_2[1]]))
                        tip_points = np.array([0, 0])
                        float_points = np.array([1, 1])
                        if t1 > 0:
                            tip_points[0] = 1
                            float_points[0] = 0
                        if t2 > 0:
                            tip_points[1] = 1
                            float_points[1] = 0
                        tip_dist_sqr = np.sum((line_1[2*tip_points[0]:2*tip_points[0]+2] -
                                               line_2[2*tip_points[1]:2*tip_points[1]+2])**2)
                        float_dist_sqr = np.sum((line_1[2*float_points[0]:2*float_points[0]+2] -
                                                line_2[2*float_points[1]:2*float_points[1]+2])**2)
                        cross_response = 2*np.sqrt(tip_dist_sqr) + np.sqrt(float_dist_sqr) - 0.5*(len_1 + len_2)
                        if cross_response_min > cross_response:
                            angle = gauge_hand.calc_angle_seg(lines[i, 0], lines[j, 0])
                            if (angle > min_angle_rad and
                                abs(angle-np.pi) > min_angle_rad and
                                    2*np.pi - angle > min_angle_rad and
                                    angle < max_angle_rad and
                                    abs(angle-np.pi) > max_angle_rad and
                                    2*np.pi - angle > max_angle_rad):
                                cross_response_min = cross_response
                                line_1_min = np.array([line_1[2*tip_points[0]:2*tip_points[0]+2],
                                                      line_1[2*float_points[0]:2*float_points[0]+2]])
                                line_2_min = np.array([line_2[2*tip_points[1]:2*tip_points[1]+2],
                                                      line_2[2*float_points[1]:2*float_points[1]+2]])
        gauge_hands = np.array([line_1_min, line_2_min])

        seg_1 = np.array([gauge_hands[0, 0], gauge_hands[0, 1]]).flatten()
        seg_2 = np.array([gauge_hands[1, 0], gauge_hands[1, 1]]).flatten()
        angle_1 = gauge_hand.calc_angle_seg(seg_1, np.array([0, 1, 0, 0]))
        angle_2 = gauge_hand.calc_angle_seg(seg_2, np.array([0, 1, 0, 0]))
        avg_angle = (angle_1 + angle_2) / 2
        for i in range(gauge_hands.shape[0]):
            cv2.line(img_out, tuple(gauge_hands[i, 0]),
                     tuple(gauge_hands[i, 1]), (0, 0, 255), 1)
        return gauge_hands, avg_angle, img_out


# Temperature meter data
class temp_meter:
    angle_data = np.array([0.8135597303853407, 5.504780036787471])
    value_data = np.array([0, 150])


# Pressure meter data
class press_meter:
    angle_data = np.array([0.7853981633974483, 5.497787143782138])
    value_data = np.array([-1, 10])
