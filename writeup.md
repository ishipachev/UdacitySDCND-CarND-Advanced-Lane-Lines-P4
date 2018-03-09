## Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration1.jpg "Distorted"
[image1b]: ./writeup_pics/calibration1.jpg "Unddistorted"


[image2]: ./writeup_pics/undist_img.jpg "Road Transformed"
[image3]: ./writeup_pics/edges_img.jpg "Binary Example"
[image4]: ./writeup_pics/undist_with_lines.jpg "Warp Example"
[image4b]: ./writeup_pics/bv_with_lines.jpg "Warp Example"
[image5]: ./writeup_pics/birdview_detected.jpg "Warp Example"
[image6]: ./writeup_pics/video_example.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in file `calibrate.py`. Codes invokes one time before any computation to calculate camera params and save them in `camera_params.pickle` file.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function (line 41 in file). I saved camera params in .pickle file to easy load at the moments I need and don't restart calibration procedure every time I need.

In `pipeline.py` file I loaded `mtx` and `dist` cameara parameters in string 81 via function `read_camera_params()`. Loaded parameters I used in core function `pipeline()` to undistort input images.

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]
![alt text][image1b]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. Thresholding steps are in the `find_edges()` function in `find_edges.py`, lines 83-94. I used magnitude threshold with a direction threshold (composed with AND operator, line 91) operator and added results (used OR operator, line 94) to result for single S channel threshold in HLS representation.  Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective was written in `bv_transform.py` module. Transform was made by a function called `bv_transform()`, which appears in lines 13 through 28.  The `bv_transform()` function takes as inputs an image (`img`), source (`src`) build up inside a function, the same true for destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
    point1 = np.array([276, 673], dtype=np.float32)
    point2 = np.array([584, 455], dtype=np.float32)
    point3 = np.array([698, 455], dtype=np.float32)
    point4 = np.array([1032, 673], dtype=np.float32)
```

Point coordinates were calculated in image tool.

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 203, 720      | 100, 1280     | 
| 585, 460      | 100, 0        |
| 695, 460      | 620, 0        |
| 1127, 720     | 620, 1280     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]
![alt text][image4b]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I've found histogram small window of the picture (1/9 of its height) to find distribution of white pixels. Right after this I calculated window covered those section and drop out any outliers. If I found enough points in window I recentered window on next step. So after this procedure I will have points on lines without many outliers to disturb line detection. I found lane lines by usage `polyfit` function provided by `numpy` module. As a result I got coefficients for both lanes.
Picture shows lane lines pixels detected in greed sliding windows and polynomial line to fit these points.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in `line_fit.py` module in function `line_fit()` at the end of the function, where I calculated vairables `left_curverad` and `right_curverad`, from lines 97 with usage of common formula for line curvature.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `pipeline.py`, lines from 88.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](video_result.avi)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further: 
1. HLS channels works good enough to be used as a main source but have some problems with different lighting conditions. All solution doesn't seems like "all conditions robust and stable". Better to use Bayesian networks?
2. Polyfit is nice solution with dropping outliers before apply. But would be much better to use RANSAC algorithm, which allows us to care less about outliers points and use one algorithm to "rule them all"
3. Birdview convertation rely on handwork by getting pixels we need to build up source points. Much better to use reliable information about mounting point of camera: height and angle. So we can calculate birdview without any hardcoded source points, like we did above.
4. I've skipped stabilisation part, but would be nice to try to shift some lines information in next frame and try to stabilize solution in every next frame, which should prevent detection from jumping.
