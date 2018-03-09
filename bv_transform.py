import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

##  Writeupt part
# img_path = 'output/s_binary.jpg'
# img = cv2.imread(img_path, 0)
# img_size = (img.shape[0], img.shape[1])


def corners_unwarp(undist, src, dst):
    img_size = undist.shape[0:2]
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(undist, M, img_size)
    return warped, M


def bv_transform(img):
    img_size = img.shape[0:2]
    point1 = np.array([265, 677], dtype=np.float32)
    # point2 = np.array([556, 477], dtype=np.float32)
    # point3 = np.array([747, 477], dtype=np.float32)
    point2 = np.array([606, 445], dtype=np.float32)
    point3 = np.array([697, 445], dtype=np.float32)
    point4 = np.array([1038, 677], dtype=np.float32)

    src = np.array([point1, point2, point3, point4])
    offset = 150

    dst = np.float32([[offset, img_size[1]],
                      [offset, 0],
                      [img_size[0] - offset, 0],
                      [img_size[0] - offset, img_size[1]]])
    warped, M = corners_unwarp(img, src, dst)
    return warped, M


##  Writeupt part
# warped, M = bv_transform(img)
#
# print(warped.shape)
#
# plt.imshow(warped, cmap='gray')
# plt.show()
# plt.imsave('output/binary_warped.jpg', warped, cmap=cm.gray)
#
# plt.imshow(img)
# plt.show()
