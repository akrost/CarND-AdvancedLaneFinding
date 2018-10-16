# **Advanced Lane Finding**

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

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./examples/01_original.png "Original Test image"
[image3]: ./examples/02_undistorted.png "Undistorted Test image"
[image4]: ./examples/03_warped.png "Warped Test image"
[image5]: ./examples/04_schannel.png "S-Channel of Test image"
[image6]: ./examples/05_sobelx.png "Sobel in x-direction of Test image"
[image7]: ./examples/06_filtered_s.png "Filtered S-Channel of Test image"
[image8]: ./examples/07_rchannel.png "R-Channel of Test image"
[image9]: ./examples/08_yellow_mask.png "Yellow mask of Test image"
[image10]: ./examples/09_binary.png "Binary Test image"
[image11]: ./examples/lower_half_hist.png "Lower Half Histogram"
[image12]: ./examples/sliding_window_udacity.png "Sliding Window Approach"
[image13]: ./examples/search_around_poly_udacity.png "Search Around Poly Approach"
[result]: ./examples/result.gif "Result GIF"

---
**Requirements**

* [Anaconda 3](https://www.anaconda.com/download/) is installed on your machine.

---
## **Getting started**

1. Clone repository:<br/>
```sh
git clone https://github.com/akrost/CarND-AdvancedLaneFinding.git
cd carnd-advancedlanefinding
```

2. Create and activate Anaconda environment:
```sh
conda create --name carnd-p4 python=3.6
source activate carnd-p4
```
Activating the environment may vary for your OS.

3. Install packages:
```sh
pip install -r requirements.txt
```

4. Run the project
```sh
python AdvancedLaneLines.py project_video.mp4 project_video_output.mp4
```

Optionally you can also add a start and/or end times
```sh
python AdvancedLaneLines.py project_video.mp4 project_video_output.mp4 --t_start '00:00:05' --t_end '00:00:23.234'
```

or just check out the help:

```sh
python AdvancedLaneLines.py -h
usage: AdvancedLaneLines.py [-h] [--t_start T_START] [--t_end T_END]
                            input_video output_video

Script will detect lane lines in a given video.

positional arguments:
  input_video        Path to video clip
  output_video       Path to output video clip

optional arguments:
  -h, --help         show this help message and exit
  --t_start T_START  Start time of video processing. Format '01:03:05.3'
  --t_end T_END      End time of video processing. Format '01:03:05.3'
```


---

## **Project**

### Camera Calibration

The camera calibration is part of the `ColorSpaceExploration.ipynb` jupyter notebook.

#### Camera matrix and distortion coefficients

The code for this step is contained in the first code cell 4 and 5 of the IPython notebook.  

```python
# Convert to grayscale
gray = cv2.cvtColor(cal_img, cv2.COLOR_RGB2GRAY)

# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, cb_size, None)

# Calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
```

First I converted the color image to grayscale. To the grayscale image I applied the `cv2.findChessboardCornerns()` function. With the found corners one can use `cv2.calibrateCamera()` to calculate the camera matrix and the distortion coefficients.

The parameters are also saved in a pickle file for later use in the main script:

```python
# Saving the objects:
with open('camera_calibration.p', 'wb') as file:
    pickle.dump([ret, mtx, dist, rvecs, tvecs], file)
```

#### Undistortion

The camera matrix and the distortion coefficients calculated above can be used to undisort any image with the `cv2.undistort()` function as shown in the 6th cell of the notebook.  

```python
def undistort(img, mtx, dist):
    """
    This function undistorts a given image
    :param img: Input image
    :param mtx: Camera matrix
    :param dist: Distortion coefficients
    :return: Undistorted image
    """
    return cv2.undistort(img, mtx, dist, None, mtx)

# Undistort image
dst = undistort(img, mtx, dist)
```

The image shows an image in its original and its undistorted form.

![alt text][image1]


### Pipeline

The pipeline was build experimenting in the IPython notebook `ColorSpaceExploration.ipynb` and was adapted in the main code `AdvancedLaneLines.py`. 

#### 1. Original image

The original image looks like this:

![alt text][image2]

#### 2. Distortion-correction

After distortion-correction the image looks like this:

![alt text][image3]

To see how the correction was done, refer to section `Undistortion` in this file.

#### 3. Perspective transformation

The code for my perspective transform includes a function called `warp()`, which appears in lines 209 through 249 in the file `AdvanceLaneLines.py`:

```python
def warp(img, pers_margin=425, margin_bottom=50, margin_top=450, margin_sides=150, reverse=False):
```
The `warp()` function takes as inputs an image (`img`), as well as some margins and a reverse-flag. The margins are used to calculate the src points. This way the src-area could later be calculated with parameters extracted from the image itself, rather than setting it to a fixed size. The destination points are a static calculation with respect to the image size:

```python
    img_size = (img.shape[1], img.shape[0])

    # Four source coordinates
    src = np.float32(
            [[img_size[0] - margin_sides - pers_margin, margin_top],
             [img_size[0] - margin_sides, img_size[1] - margin_bottom],
             [margin_sides, img_size[1] - margin_bottom],
             [margin_sides + pers_margin, margin_top]])
    
    # Four destination coordinates
    dst = np.float32(
            [[img_size[0]*3//4, 0],
             [img_size[0]*3//4, img_size[1]],
             [img_size[0]//4, img_size[1]],
             [img_size[0]//4, 0]])
```

The reverse flag swaps the src and dst points so that the `warp()` function would actually **unwarp** an image.

A warped image looks like this:

![alt text][image4]

#### 4. Color transformation

In this step the goal is to extract the lane line pixels from the image in bird's-eye view. 
After comparing different color spaces and their channels it was found that the S-channel works quite good for detecting yellow lines and the R-channel is best for white lane lines.

To extract a certain channel from an image the `channel_thresh()` function was used. It allows you to get a whole channel (with `thresh=(0, 255)`) or you can apply a threshold to that specific channel.

```python
def channel_thresh(img, channel='h', thresh=(90, 255)):
    """
    This function extracts a given channel from an image and applies a threshold to it.
    The output image is a binary image with the same shape of the input image.
    :param img: Input image (BGR)
    :param channel: The channel that the threshold is applied to
    :param thresh: Threshold to apply. Tuple (lower, upper)
    :return: Binary image
    """
    # extract channel
    if channel == 'h':
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        c = hsv[:, :, 0]
    elif channel == 'l':
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        c = hls[:, :, 1]
    elif channel == 's':
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        c = hsv[:, :, 2]
    elif channel == 'b':
        c = img[:, :, 0]
    elif channel == 'g':
        c = img[:, :, 1]
    elif channel == 'r':
        c = img[:, :, 2]
    else:
        raise Exception('Channel {} is not recognized. Possible values are: h, l, s, r, g, b'.format(channel))
        
    # use threshold to generate binary image
    binary = np.zeros_like(c)
    binary[(c > thresh[0]) & (c <= thresh[1])] = 1
    
    return binary
```

##### 4.1 S-Channel and filtering with Sobel

As mentioned above, the **S-Channel** is extracted with the `channel_thresh()` function. A threshold of (90, 255) was chosen:

```python
# S channel
thresh_s = (90, 255)
bin_s = channel_thresh(img, channel='s', thresh=thresh_s)
```
![alt text][image5]
*S-Channel with threshold (90, 255)*

The S-Channel alone detects white lines quite ok, but introduces a high noise for yellow lines. To filter that noise, a **Sobel** edge detection in x-direction was used.

```python
sobel = abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100)
```

The sobel operation detects gradients in certain direction by using a specific kernel. This means, it is good to detect horizontal or in this case vertical lines. But as one can see in the next image, it not only detects lane lines, but also the edges of road dividers or shadows. 

![alt text][image6]
*Sobel edge detection in x-direction*


The solid line on the left side of the image is not a lane line, but a shadows. Having two noisy images it is now possible to combine them, to only get the pixels that are most likely to be valid. 

```python
# Combination of s-channel and sobel
filtered_s = cv2.bitwise_and(bin_s, sobel)
```

![alt text][image7]
*Combination of S-Channel and Sobel*


The combination also removed the yellow line detected by the S-channel almost completely. To overcome this issue, a yellow color mask was used.

##### 4.2 Yellow Mask

A color mask can be extracted by using the `cv2.inRange()` function. 
The HSV-mask was found to work quite robustly with a lower threshold of (0, 100, 100) and an upper threshold of (100, 255, 255).

```python
yellow_low = np.array([0, 100, 100], dtype=np.uint8)
yellow_high = np.array([100, 255, 255], dtype=np.uint8)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
yellow_mask = cv2.inRange(hsv, yellow_low, yellow_high)//255
```

![alt text][image9]
*Yellow Color Mask*

##### 4.3 R-Channel

As mentioned above the R-channel works quite nicely for detecting the white lane lines. 
The threshold is (200, 255):

```python
# R channel
thresh_r = (200, 255)
bin_r = channel_thresh(img, channel='r', thresh=thresh_r)
```

![alt text][image8]
*R-Channel with threshold (200, 255)*

##### 4.4 Combination

As a last step of the color transformation, the three images
*filtered s channel*, *yellow color mask* and *r channel*
are combined using the `cv2.bitwise_or()` operation.

![alt text][image10]
*Combination of filtered-s-channel, yellow-color-mask and r-channel*





#### 5. Line detection
##### 5.1 Sliding Window

The sliding window approach is implemented in lines 471 through 605 in the file `AdvancedLaneLines.py`.

```python
def sliding_window(binary_warped, state, leftx_base=None, rightx_base=None):
```

The leftx_base and the rightx_base parameters are the starting point for the sliding window. If they are not given, they can be determined by using a histogram. 

**Histogram**

The function `get_hist()` returns a histogram of the lower half of a binary image

![alt text][image11]
*Histogram of the Lower Half of a Binary Image*

From the histogram the starting point can be found by searching for the peaks in the right and the left half of the histogram. The function `hist_peaks()` does exactly this. 


Using the histogram to find the starting points works well for rather straight lines. If the lines are heavily bent, the histogram doesn't show clear peaks. For this scenario the `sliding_window()` function allows for starting points to be given. The starting points can then be calculated by using the last detected lines. This is currently not implemented, though.

The found base values for the left and the right lane are then the position for the first bounding box. Within the bounding box all the lane pixels are extracted and the center of the next bounding box is adjusted, if the center of the lane pixels in the current box (using `np.mean()`) doesn't match its current position.

After extracting all possible line pixels, a second order poly line can be fit through those points using the `np.fitpoly()` function.

![alt text][image12]
*Sliding Window Approach [Source: Udacity]* 

If the sliding window function detects two valid lines, they are added to the current state and the function returns `True`.

##### 5.2 Search Around Poly

The search around poly approach is implemented in lines 608 through 672 in the file `AdvancedLaneLines.py`.

```python
def search_around_poly(binary_warped, state):
```

This function can only be used, if the two lane lines where already been found. So for single images this method is not applicable, but for videos it can be used.
The function searches in an area around the last poly lines for new lane line pixels.
After this, the poly line can be created with the same method that is also used in the sliding window approach. 

![alt text][image13]
*Search Around Poly Approach [Source: Udacity]*

If the search around poly function detects two valid lines, they are added to the current state and the function returns `True`.

##### 5.3 Take last Lines

The take last lines function is implemented in lines 678 through 707 in the file `AdvancedLaneLines.py`.

```python
def take_last_lines(state):
```

This approach is not actually a line detection method. It simply takes the last valid line detected (by any of the other detection methods) and uses it as the current line. This should only be a backup method if the other detection methods failed. It is also advisable to only allow copying for so long, especially in safety relevant use-cases.


#### 6. Line() and State()

**Line()**

The class `Line()` is an object to hold all the information needed to describe a line. 
In this implementation this is:

|  variable   |  description                                         |
|:------------|:-----------------------------------------------------|
| side        | Side on which the line is. Can be 'left' or 'right'. |
| poly_params | Paramerters a, b and c of the second order polynomial line described as x = ay²+by+c|
| detection_method | Method how the line was detected (see detection methods below) |
| line_pixels | Pixels that were in the considered area (x and y) |


Detection methods

| detection method | description |
|:-----------------|:------------|
| 1 | Line found in image using "sliding window"-approach |
| 2 | Line found in image using "search around poly"-approach |
| 3 | No valid line found, used last line |
| 4 | No valid line found, no last line available |


**State()**

The class `State()` is an object that holds the current state of the lane line detection. 

| variable         | description |
|:-----------------|:------------|
| IMG_WIDTH        | Width of the input image |
| IMG_HEIGHT       | Height of the input image |
| LANE_WIDTH_METER | Width of the lane in meters |
| num_last_saved   | Number of data points (lane pairs) saved in state. Currently only the last sighting is used, but for smoothing saving more might be useful. |
|copy_limit        | The maximum number of copies of the last line (using `take_last_line()`), if no valid new line was found |
| num_copied_lines | Number of copied lines. Is increased by 1 whenever `take_last_line()` successfully copied the last line. Is reset to 0 if sliding window was successful |
| last_lines       | Array of size <num_last_saved> of last detected lines |
| lane_radius      | Radius of the lane (based on last lines) in meters |
| off_center       | Position of the car with respect to the center of the lane in meters |



#### 7. Calculated the Radius of Curvature of the Lane and the Position of the Vehicle with Respect to Center

Both, the calculation of the radius and the position of the vehicle with respect to the center of the lane are done in the `State()` class. The calculation is triggered by the `add_lines()`  method of the state. Internally, add_lines() triggers the methods `_get_off_center_dist()` and `_get_lane_radius()`.

#### 8. Visualize State

This function is implemented in lines 710 through 791 of the file `AdvancedLaneLines.py`.

```python
def visualize_from_state(img, state, line_color=(255, 0, 0), lane_color=(0, 255, 0), warped=True):
```

The input image (`img`) is the canvas to draw the results on to. The image can be warped or not. If it is warped, the `warped` flag has to be set to True. This will lead to the image being unwarped before returned. The parameter `line_color` and `lane_color` determin the color of the line and the lane. 

The function also puts text (lane radius and off center distance) on the image. All the information like the radius or the lines are stored in the `state`. The visualize_from_state() method always visualizes the last stored state/lines.

---

### Video processing
#### 1. Final Video Output

To use the image preprocessing pipeline as well as the lane line detection and the state visualization in on step, the `process_images()` function was introduce. It is implemented in the lines 794 through 843 in the file `AdvancedLaneLines.py`.

```python
def process_image(img):
```

The only input to this wrapper function is an image (or the current frame of a video). All other necessary variables (state, camera matrix and distortion coefficients) are defined globally. 

In the process_image function the order of line detection methods is set. In this example the order is like follows:
* The detection method for the first frame is always *sliding_window* 
* If the last frame was detected by sliding_window or search around poly (i.e. there was a lane line detected, not copied) *search_around_poly* is used.
* If search_around_poly failed, sliding_window is tried
* If neither search_around_poly nor sliding_window worked, *take_last_lines* is used
* Currently there is no backup in case take_last_lines fails. This is the case, if sliding_window and search_around_poly fail for more than <state.copy_limit> frames.

**Result:**

![alt text][result]


Here is a [link to the video result](./videos/project_video_output.mp4)

---

### Possible Improvements

* Currently lines can only be detected in pairs (left and right line). This is due to the validation method. A future improvement could be to allow detection of individual lines.
* Even for the curent double line detection, the validation can be improved. Currently it filters out a some valid lines as well. 
* There are multiple parameters (like thresholds, margins) that can be fine-tuned.
* Especially for the challenge and the harder challenge video, the pipeline can be improved. Currently for those videos the binary warped images contain quite some noise.
