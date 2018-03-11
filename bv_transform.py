import cv2
import numpy as np


def corners_unwarp(undist, src, dst):
    img_size = undist.shape[0:2]
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(undist, M, img_size)
    return warped, M


def bv_transform(img):
    img_size = img.shape[0:2]
    point1 = np.array([276, 673], dtype=np.float32)
    point2 = np.array([584, 455], dtype=np.float32)
    point3 = np.array([698, 455], dtype=np.float32)
    point4 = np.array([1032, 673], dtype=np.float32)

    src = np.array([point1, point2, point3, point4])
    offset = 150

    dst = np.float32([[offset, img_size[1]],
                      [offset, 0],
                      [img_size[0] - offset, 0],
                      [img_size[0] - offset, img_size[1]]])
    warped, M = corners_unwarp(img, src, dst)
    return warped, M
