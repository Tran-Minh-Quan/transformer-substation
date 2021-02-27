from numpy.random import randint
from numpy import uint8
import os
from os import listdir
from os.path import join, splitext
from cv2 import imread, imshow, waitKey, resize, selectROI
import cv2
import numpy as np
from ast import literal_eval
import sys
from feature_matching import superglue


def test_data():
    def align(image, template, mask):
        MAX_FEATURES = 1050
        GOOD_MATCH_PERCENT = 0.15
        # MAX_FEATURES = cv2.getTrackbarPos('max_feature', winName)
        # GOOD_MATCH_PERCENT = cv2.getTrackbarPos('good_match_percent', winName)/100

        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create(MAX_FEATURES)
        im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        tem_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        keypoints1, descriptors1 = orb.detectAndCompute(im_gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(tem_gray, mask=mask)

        # Match features.
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)

        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        # Draw top matches
        imMatches = cv2.drawMatches(image, keypoints1, template, keypoints2, matches, None)

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        # Find homography
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        # Use homography
        height, width, channels = template.shape
        aligned = cv2.warpPerspective(image, h, (width, height))
        align_result = cv2.hconcat([aligned, template])

        return imMatches, aligned


    def callback(x):
        pass

    winName = 'Panel for calibration'
    cv2.namedWindow(winName, 0)
    cv2.resizeWindow(winName, 800, 600)

    cv2.createTrackbar('max_feature', winName, 800, 2000, callback)
    cv2.createTrackbar('good_match_percent', winName, 15, 100, callback)

    dir = '..\\Test\\feature_matching'

    template = cv2.imread(join(dir, 'template.JPG'))
    h, w = template.shape[:2]

    bbox_dir = join(dir, 'template.txt')
    bbox = [float(i) for i in open(bbox_dir, 'r').read().splitlines()[0].split(' ')[1:]]
    top_left = (round((bbox[0] - bbox[2]/2)*w), round((bbox[1] - bbox[3]/2)*h))
    bottom_right = (round((bbox[0] + bbox[2]/2)*w), round((bbox[1] + bbox[3]/2)*h))

    template = template[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]

    mask = cv2.imread(join(dir, 'template_mask.PNG'), 0)[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    im_names = [filename for filename in listdir(dir) if filename.endswith('.JPG') and filename != 'template.JPG']
    while 1:
        for im_name in im_names:
            image = cv2.imread(join(dir, im_name))
            h, w = image.shape[:2]

            bbox_dir = join(dir, im_name.replace('JPG', 'txt'))
            bbox = [float(i) for i in open(bbox_dir, 'r').read().splitlines()[0].split(' ')[1:]]
            top_left_1 = (round((bbox[0] - bbox[2]/2)*w), round((bbox[1] - bbox[3]/2)*h))
            bottom_right_1 = (round((bbox[0] + bbox[2]/2)*w), round((bbox[1] + bbox[3]/2)*h))

            image = image[top_left_1[1]:bottom_right_1[1], top_left_1[0]:bottom_right_1[0], :]

            result = align(image, template, mask)[0]
            w_result = 1000
            result = cv2.resize(result, (w_result, round(result.shape[0]/result.shape[1]*w_result)))
            cv2.imshow('feature matching', result)

            result = align(image, template, mask)[1]
            objs = get_center_and_marks(r'd:\WON\THI_GIAC_MAY\Data\Test\feature_matching\center_and_marks.txt')
            template_displace = template.copy()
            for obj in objs:
                if len(obj) == 1:
                    center = (obj[0][0] - top_left[0], obj[0][1] - top_left[1])
                    cv2.circle(template_displace, center, 2, (0, 0, 255), -1)
                    cv2.circle(result, center, 2, (0, 0, 255), -1)
                else:
                    initial_point = (obj[0][0] - top_left[0], obj[0][1] - top_left[1])
                    end_point = (obj[1][0] - top_left[0], obj[1][1] - top_left[1])
                    cv2.line(template_displace, initial_point, end_point, (0, 0, 255), 1)
                    cv2.line(result, initial_point, end_point, (0, 0, 255), 1)
            result = cv2.hconcat([result, template_displace])
            w_result = 1000
            result = cv2.resize(result, (w_result, round(result.shape[0] / result.shape[1] * w_result)))
            cv2.imshow('aligned image (left) and template (right)', result)

            cv2.waitKey(0)
        # image_dir = r"D:\WON\THI_GIAC_MAY\Data\Test\MR_gauge\crop\test_4.PNG"
        # image = cv2.imread(image_dir)
        #
        # template_dir = r"D:\WON\THI_GIAC_MAY\Code\Project\Meter\template.PNG"
        # template = cv2.imread(template_dir)
        #
        # # Convert images to grayscale
        # align_img(image, template)
        cv2.waitKey(0)


def draw_center_and_marks():
    global ix, iy, drawing, img, img_list, obj_list, x_, y_
    def draw(event, x, y, flags, params):
        global ix, iy, drawing, mode, img, img_list, obj_list, x_, y_
        x_ = x
        y_ = y
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                img = img_list[-1].copy()
                cv2.line(img, (ix, iy), (x, y), (0, 0, 255), 1)
                cv2.imshow(win_name, img)
        elif event == cv2.EVENT_LBUTTONUP:
            print('up')
            drawing = False
            if (x, y) != (ix, iy):
                obj_list.append([(round(ix/ratio), round(iy/ratio)), (round(x/ratio), round(y/ratio))])
            else:
                obj_list.append([(round(x/ratio), round(y/ratio))])
                img = img_list[-1].copy()
                cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
                cv2.imshow(win_name, img)
            img_list.append(img)

    x_ = None
    y_ = None
    img_list = []
    obj_list = []
    drawing = False  # true if mouse is pressed
    win_name = 'Draw points and lines'
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, draw)

    img = cv2.imread(r'd:\WON\THI_GIAC_MAY\Data\Test\feature_matching\IMG_0014.JPG')
    h, w = img.shape[:2]
    ratio = 0.91
    img = cv2.resize(img, (round(w*ratio), round(h*ratio)))


    img_list.append(img)
    cv2.imshow(win_name, img)
    while 1:
        print(obj_list)
        k = cv2.waitKeyEx(1)
        if k == 8 and len(img_list) > 1:  # press backspace
            img_list.pop()
            obj_list.pop()
            cv2.imshow(win_name, img_list[-1])
        elif k == ord('s'):
            with open(r'd:\WON\THI_GIAC_MAY\Data\Test\feature_matching\center_and_marks.txt', 'w') as f:
                f.write(str(obj_list))
                f.close()
            break
        elif k == 27:  # press esc
            break


        # print(cv2.waitKeyEx())

def get_center_and_marks(dir):
    with open(dir, 'r') as f:
        return literal_eval(f.read())

if __name__ == '__main__':
    # test_data()
    # draw_center_and_marks()
    # print(get_center_and_marks(r'd:\WON\THI_GIAC_MAY\Data\Test\feature_matching\center_and_marks.txt')[-1])
    # print("Current Directory", os.getcwd())
    # print("Current Directory", os.pardir)
    print('ok')


