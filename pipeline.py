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
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    return result


def draw_curv_and_dist(img, left_fit, right_fit, left_curverad, right_curverad):

    xm_per_pix = 3.7/620  # meters per pixel in x dimension

    mid_pos = (right_fit[2] - left_fit[2]) / 2 * xm_per_pix
    mid_str = "Vehicle position: " + str(mid_pos)

    left_curv_str = "Curvature radius left line: " + str(left_curverad) + "m"
    left_curv_str = "Curvature radius right line: " + str(right_curverad) + "m"

    font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(img, mid_str, (10, 20), font, 0.2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, mid_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
    cv2.putText(img, left_curv_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
    cv2.putText(img, left_curv_str, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

def pipeline(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # plt.imshow(undist)
    # plt.show()

    edges = find_edges(undist)
    # plt.imshow(edges, cmap="gray")
    # plt.show()

    warped, M = bv_transform(edges)
    # plt.imshow(warped, cmap="gray")
    # plt.show()

    left_fit, right_fit, left_curverad, right_curverad = line_fit(warped)
    result = draw_lines(img, warped, left_fit, right_fit, M)
    draw_curv_and_dist(result, left_fit, right_fit, left_curverad, right_curverad)

    return result

# Read matrix and distortion parameter of the camera calculated by calibrate.py
params_path = 'output/camera_params.pickle'
mtx, dist = read_distort_params(params_path)

video_path = "project_video.mp4"

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output/output.avi', fourcc, 25.0, (1280, 720), isColor=True)

cap = cv2.VideoCapture(video_path)
cnt = 0
while cap.isOpened():
# for i in range(25):
    ret, frame = cap.read()
    if ret is True:
        result = pipeline(frame, mtx, dist)
        out.write(result)
        cnt += 1
        print(cnt)
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()



