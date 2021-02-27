from os import listdir
from os.path import join, splitext
import cv2
import numpy as np
from ast import literal_eval
from time import time
import torch
import matplotlib.cm as cm
import sys
sys.path.append('./SuperGluePretrainedNetwork-master/models')
from matching import Matching
from utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics, make_matching_plot_fast)

def superglue(image, template, mask=None):
    nms_radius = 4  # SuperPoint Non Maximum Suppression radius
    keypoint_threshold = 0.005  # SuperPoint key point detector confidence threshold
    max_keypoints = 1024  # Maximum number of key points detected by Superpoint, -1 all
    weights = 'outdoor'  # SuperGlue weights
    sinkhorn_iterations = 20  # Number of Sinkhorn iterations performed by SuperGlue
    match_threshold = 0.2  # SuperGlue match threshold
    resize = [640, 480]
    resize_float = True
    rotation_int = False

    if rotation_int:  # If a rotation integer is provided (e.g. from EXIF data), use it
        rot0, rot1 = 0, 0  # choose at random because I have no idea about these values
    else:
        rot0, rot1 = 0, 0

    torch.set_grad_enabled(False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': nms_radius,
            'keypoint_threshold': keypoint_threshold,
            'max_keypoints': max_keypoints
        },
        'superglue': {
            'weights': weights,
            'sinkhorn_iterations': sinkhorn_iterations,
            'match_threshold': match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)

    image0, inp0, scales0 = read_image(image, device, resize, rot0, resize_float)
    image1, inp1, scales1 = read_image(template, device, resize, rot1, resize_float)
    pred = matching({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, confidence = pred['matches0'], pred['matching_scores0']

    # Keep the matching keypoints.
    valid = matches > -1
    # valid = confidence > .96
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = confidence[valid]
    color = cm.jet(confidence[valid])

    # Find homography
    h, status = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC)

    # # Draw top matches
    text = [
        'SuperGlue',
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts0))
    ]
    k_thresh = matching.superpoint.config['keypoint_threshold']
    m_thresh = matching.superglue.config['match_threshold']
    small_text = [
        'Keypoint Threshold: {:.4f}'.format(k_thresh),
        'Match Threshold: {:.2f}'.format(m_thresh)
    ]
    out = make_matching_plot_fast(
        image0, image1, kpts0, kpts1, mkpts0, mkpts1, color, text,
        path=None, show_keypoints=True, small_text=small_text)

    cv2.imshow('matches', out)

    # Use homography
    height, width = image1.shape
    image0 = np.rint(image0).astype(np.uint8)
    aligned = cv2.warpPerspective(image0, h, (width, height))
    # align_result = cv2.hconcat([aligned, template])
    cv2.imshow('aligned', aligned)

    return image, aligned

def orb_bf(image, template, mask):
    MAX_FEATURES = 1050
    GOOD_MATCH_PERCENT = 0.15
    # MAX_FEATURES = cv2.getTrackbarPos('max_feature', winName)
    # GOOD_MATCH_PERCENT = cv2.getTrackbarPos('good_match_percent', winName)/100

    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tem_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(tem_gray, mask=mask)
    print(len(keypoints1))

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2)

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

    # # Match features.
    # matcher = cv2.BFMatcher()
    # matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    # good_matches = [match[0] for match in matches if match[0].distance < 0.8*match[1].distance]
    #
    # # Sort matches by score
    # good_matches.sort(key=lambda x: x.distance, reverse=False)
    #
    # # Remove not so good matches
    # numGoodMatches = int(len(good_matches) * GOOD_MATCH_PERCENT)
    # good_matches = good_matches[:numGoodMatches]
    #
    # # Draw top matches
    # imMatches = cv2.drawMatches(image, keypoints1, template, keypoints2, good_matches, None)
    #
    # # Extract location of good matches
    # points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    # points2 = np.zeros((len(good_matches), 2), dtype=np.float32)
    #
    # for i, match in enumerate(good_matches):
    #     points1[i, :] = keypoints1[match.queryIdx].pt
    #     points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    print(points1)
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = template.shape
    aligned = cv2.warpPerspective(image, h, (width, height))
    align_result = cv2.hconcat([aligned, template])

    return imMatches, aligned


def sift_bf(image, template, mask):
    MAX_FEATURES = 1050
    GOOD_MATCH_PERCENT = .08
    # MAX_FEATURES = cv2.getTrackbarPos('max_feature', winName)
    # GOOD_MATCH_PERCENT = cv2.getTrackbarPos('good_match_percent', winName)/100

    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tem_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(im_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(tem_gray, mask=mask)
    descriptors1 = np.rint(descriptors1).astype(np.uint8)
    descriptors2 = np.rint(descriptors2).astype(np.uint8)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    print(len(matches))

    # Draw top matches
    imMatches = cv2.drawMatches(image, keypoints1, template, keypoints2, matches, None)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # # Match features.
    # matcher = cv2.BFMatcher()
    # matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    # good_matches = [match[0] for match in matches if match[0].distance <= match[1].distance]
    #
    # # Sort matches by score
    # good_matches.sort(key=lambda x: x.distance, reverse=False)
    #
    # # Remove not so good matches
    # numGoodMatches = int(len(good_matches) * GOOD_MATCH_PERCENT)
    # good_matches = good_matches[:numGoodMatches]
    #
    # # Draw top matches
    # imMatches = cv2.drawMatches(image, keypoints1, template, keypoints2, good_matches, None)
    #
    # # Extract location of good matches
    # points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    # points2 = np.zeros((len(good_matches), 2), dtype=np.float32)
    #
    # for i, match in enumerate(good_matches):
    #     points1[i, :] = keypoints1[match.queryIdx].pt
    #     points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = template.shape
    aligned = cv2.warpPerspective(image, h, (width, height))
    align_result = cv2.hconcat([aligned, template])

    return imMatches, aligned


def test_data():
    def callback(x):
        pass

    winName = 'Panel for calibration'
    cv2.namedWindow(winName, 0)
    cv2.resizeWindow(winName, 800, 600)

    cv2.createTrackbar('max_feature', winName, 800, 2000, callback)
    cv2.createTrackbar('good_match_percent', winName, 15, 100, callback)

    folder_template = '../data/Quan/feature_matching'
    template = cv2.imread(folder_template + '/template.JPG')
    h, w = template.shape[:2]

    bbox_dir = join(folder_template, 'template.txt')
    bbox = [float(i) for i in open(bbox_dir, 'r').read().splitlines()[0].split(' ')[1:]]
    top_left = (round((bbox[0] - bbox[2]/2)*w), round((bbox[1] - bbox[3]/2)*h))
    bottom_right = (round((bbox[0] + bbox[2]/2)*w), round((bbox[1] + bbox[3]/2)*h))

    template = template[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]

    mask = cv2.imread(folder_template + '/template_mask.PNG', 0)[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    folder_im = '../data/Quan/test/mr_gauge/origin/image'
    im_names = [filename for filename in listdir(folder_im) if filename.endswith('.JPG') and filename != 'template.JPG']

    while 1:
        for im_name in im_names:
            image = cv2.imread(join(folder_im, im_name))
            h, w = image.shape[:2]

            folder_label = '../data/Quan/test/mr_gauge/origin/label'
            bbox_dir = join(folder_label, im_name.replace('JPG', 'txt'))
            bbox = [float(i) for i in open(bbox_dir, 'r').read().splitlines()[0].split(' ')[1:]]
            top_left_1 = (round((bbox[0] - bbox[2]/2)*w), round((bbox[1] - bbox[3]/2)*h))
            bottom_right_1 = (round((bbox[0] + bbox[2]/2)*w), round((bbox[1] + bbox[3]/2)*h))

            image = image[top_left_1[1]:bottom_right_1[1], top_left_1[0]:bottom_right_1[0], :]

            start = time()
            # im_matches, aligned = orb_bf(image, template, mask)
            im_matches, aligned = superglue(image, template, mask)
            print(time() - start)

            w_result = 1000
            im_matches = cv2.resize(im_matches, (w_result, round(im_matches.shape[0]/im_matches.shape[1]*w_result)))
            cv2.imshow('feature matching', im_matches)

            objs = get_center_and_marks(folder_template + '/center_and_marks.txt')
            template_displace = template.copy()
            for obj in objs:
                if len(obj) == 1:
                    center = (obj[0][0] - top_left[0], obj[0][1] - top_left[1])
                    cv2.circle(template_displace, center, 2, (0, 0, 255), -1)
                    cv2.circle(aligned, center, 2, (0, 0, 255), -1)
                else:
                    initial_point = (obj[0][0] - top_left[0], obj[0][1] - top_left[1])
                    end_point = (obj[1][0] - top_left[0], obj[1][1] - top_left[1])
                    cv2.line(template_displace, initial_point, end_point, (0, 0, 255), 1)
                    cv2.line(aligned, initial_point, end_point, (0, 0, 255), 1)
            # align_compare = cv2.hconcat([aligned, template_displace])
            # align_compare = cv2.resize(align_compare,
            #                            (w_result, round(align_compare.shape[0] / align_compare.shape[1] * w_result)))
            # cv2.imshow('aligned image (left) and template (right)', aligned)

            cv2.waitKey(0)


def draw_center_and_marks():
    global ix, iy, drawing, img, img_list, obj_list, x_, y_

    def draw(event, x, y, flags, params):
        global ix, iy, drawing, mode, img, img_list, obj_list
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

    img_list = []
    obj_list = []
    drawing = False  # true if mouse is pressed
    win_name = 'Draw points and lines'
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, draw)

    dir_template = '../data/Quan/feature_matching/template.JPG'
    img = cv2.imread(dir_template)
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
            with open('../data/Quan/feature_matching/center_and_marks.txt', 'w') as f:
                f.write(str(obj_list))
                f.close()
            break
        elif k == 27:  # press esc
            break


def get_center_and_marks(dir_file):
    with open(dir_file, 'r') as f:
        return literal_eval(f.read())


if __name__ == '__main__':
    test_data()
    # draw_center_and_marks()
    # print(get_center_and_marks(r'data\Quan\feature_matching\center_and_marks.txt')[-1])
    # print("Current Directory", os.getcwd())
    # print("Current Directory", os.pardir)
    # print(sys.path)


