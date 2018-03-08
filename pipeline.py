import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from find_edges import find_edges
from bv_transform import bv_transform
from line_fit import line_fit


def read_distort_params(filename):
    with open(filename, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data = pickle.load(f)
        return data["mtx"], data["dist"]

# def image_pipeline(img):


def draw_lines(image, warped, left_fit, right_fit, M):

    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    Minv = np.linalg.inv(M)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    return result



params_path = 'output/camera_params.pickle'
mtx, dist = read_distort_params(params_path)
print(mtx, dist)

img_path = "test_images/test6.jpg"
# img_path = "camera_cal/calibration1.jpg"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
undist = cv2.undistort(img, mtx, dist, None, mtx)


plt.imshow(undist)
plt.show()

edges = find_edges(undist)
plt.imshow(edges, cmap="gray")
plt.show()

warped, M = bv_transform(edges)
plt.imshow(warped, cmap="gray")
plt.show()

left_fit, right_fit, leftx, lefty, rightx, righty, ploty = line_fit(warped)

print(left_fit, right_fit)

result = draw_lines(img, warped, left_fit, right_fit, M)
plt.imshow(result)
plt.show()
