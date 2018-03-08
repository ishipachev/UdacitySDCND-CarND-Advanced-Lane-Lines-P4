import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm

# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
# def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

# Define a function to threshold an image for a given range and Sobel kernel
def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    return color_binary


def get_s_channel(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:, :, 2]
    return s_channel


def color_thresh(s_channel, s_thresh=(0, 255)):
    # Threshold color channel
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    return binary_output


# Main function to get edges from picture
def find_edges(img):
    s_channel = get_s_channel(img)
    s_binary = color_thresh(s_channel, s_thresh=(170, 255))
    return s_binary

fname = "test_images/test2.jpg"
image = cv2.imread(fname)

# Choose a Sobel kernel size
ksize = 3  # Choose a larger odd number to smooth gradient measurements
# Apply each of the thresholding functions
gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
dir_binary = dir_thresh(image, sobel_kernel=15, thresh=(0.7, 1.3))

s_channel = get_s_channel(image)
s_binary = color_thresh(s_channel, s_thresh=(170, 255))

pic_list = [gradx, grady, mag_binary, dir_binary, s_channel, s_binary]
title_list = ["gradx", "grady", "mag_binary", "dir_binary", "s_channel", "s_binary"]
# cv2.imshow('gradx', gradx)
# cv2.imshow('grady', grady)
# cv2.imshow('mag_binary', mag_binary)
# cv2.imshow('dir_binary', dir_binary)

# plt.imshow(gradx)
# plt.show()
# plt.imshow(grady)
# plt.show()
# plt.imshow(mag_binary)
# plt.show()
# plt.imshow(dir_binary)
# plt.show()


# f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 2)
# f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
# # f.tight_layout()
# ax1.imshow(gradx, cmap='gray')
# ax1.set_title('gradx', fontsize=50)
# ax2.imshow(grady, cmap='gray')
# ax2.set_title('grady', fontsize=50)
# ax3.imshow(mag_binary, cmap='gray')
# ax3.set_title('mag_binary', fontsize=50)
# ax4.imshow(dir_binary, cmap='gray')
# ax4.set_title('dir_binary', fontsize=50)

fig, axes = plt.subplots(2, 3)
cnt = 0
for i in range(len(pic_list)):
    axes.flat[i].imshow(pic_list[i], cmap='gray')
    axes.flat[i].set_title(title_list[i])

plt.show()

# Save all pics in output folder
for i in range(len(pic_list)):
    plt.imsave(os.path.join('output', title_list[i] + '.jpg'), pic_list[i], cmap=cm.gray)

