import cv2
import numpy as np
import pickle
import argparse
from moviepy.editor import VideoFileClip


IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720


class Line:
    """
    Holds all relevant information to describe a line.
    """

    def __init__(self, side, poly_params, detection_method, line_pixels=None):
        # Side on which the line is
        #  'left'
        #  'right'
        self.side = side

        # Parameters of the 2nd order polynomial of the line (a * y² + b * y + c) => [a b c]
        self.poly_params = poly_params

        # Method how the lane was detected
        #  1: Line found in image using "sliding window"-approach
        #  2: Line found in image using "search around poly"-approach
        #  3: No valid line found, used last line
        #  4: No valid line found, no last line available
        self.detection_method = detection_method

        # Pixels that were in the considered area
        # line_pixels = {"x": x_pixels, "y": y_pixels}
        self.line_pixels = line_pixels


class State:
    """
    Holds the current state of the lane line detection.
    """

    def __init__(self, num_last_saved=5, copy_limit=10):
        # CONSTANTS
        self.IMG_WIDTH = 1280
        self.IMG_HEIGHT = 720
        self.LANE_WIDTH_METERS = 3  # Used to calculate the pixel resolution in x direction

        # Number of data points saved in state
        self.num_last_saved = num_last_saved

        # The maximum number of copies of the last line, if no valid new line was found
        self.copy_limit = copy_limit

        # Number of copied lines, is increased if the last line was copied (because no new line was found)
        # Is reset to 0 when a new line is found
        self.num_copied_lines = 0

        # Array to store last lanes as dict {'left':lane_obj, 'right':lane_obj}
        self.last_lines = []

        # Radius of the lane [m]
        self.lane_radius = None

        # Off center [m]
        self.off_center = None

    def _get_lane_width_pixels(self, lines):
        """
        Calculates the lane width in pixels.
        :param lines: Dict of lines {'left': Line(), 'right': Line()}
        :return: Lane with in pixels
        """
        left_line = lines['left']
        right_line = lines['right']

        left_line_x = left_line.poly_params[0] * (self.IMG_HEIGHT - 1) ** 2 \
                      + left_line.poly_params[1] * (self.IMG_HEIGHT - 1) \
                      + left_line.poly_params[2]
        right_line_x = right_line.poly_params[0] * (self.IMG_HEIGHT - 1) ** 2 \
                       + right_line.poly_params[1] * (self.IMG_HEIGHT - 1) \
                       + right_line.poly_params[2]
        return right_line_x - left_line_x

    def _get_lane_radius(self, lines):
        """
        Calculate the average radius of the lines.
        Source: https://www.intmath.com/applications-differentiation/8-radius-curvature.php
        radius = [1 + (dy/dx)^2]^(3/2) / |d^2y/dx^2|
        y = a*x^2 + b*x + c
        dy/dx= 2a*x + b
        d^2y/dx^2 = 2a

        => r = [1 + (2ax+b)^2]^(3/2) / |2a|

        NOTE: Here x and y are flipped
        :param lines: Dict of lines {'left': Line(), 'right': Line()}
        :return: Radius in meters
        """

        y_eval = self.IMG_HEIGHT - 1

        lane_width_pixels = self._get_lane_width_pixels(lines)
        pixel_per_meter_x = lane_width_pixels // self.LANE_WIDTH_METERS
        pixel_per_meter_y = (1 / 10) * pixel_per_meter_x

        # Generate the line in "pixel- or image-space" and transform it to "meter or real-world-space"
        left_line_params = lines['left'].poly_params
        right_line_params = lines['right'].poly_params
        ploty = np.linspace(0, self.IMG_HEIGHT)
        left_fitx = left_line_params[0] * ploty ** 2 + left_line_params[1] * ploty + left_line_params[2]
        right_fitx = right_line_params[0] * ploty ** 2 + right_line_params[1] * ploty + right_line_params[2]

        # Transform to real world space (rws)
        left_fit_rws = np.polyfit(ploty / pixel_per_meter_y, left_fitx / pixel_per_meter_x, 2)
        right_fit_rws = np.polyfit(ploty / pixel_per_meter_y, right_fitx / pixel_per_meter_x, 2)

        left_radius = (1 + ((2 * left_fit_rws[0] * y_eval / pixel_per_meter_y) ** 2) ** (3 / 2)) / np.abs(
            2 * left_fit_rws[0])
        right_radius = (1 + ((2 * right_fit_rws[0] * y_eval / pixel_per_meter_y) ** 2) ** (3 / 2)) / np.abs(
            2 * right_fit_rws[0])

        return np.mean([left_radius, right_radius])

    def _get_off_center_dist(self, lines):
        """
        Calculate the position of the car with respect to the center of the lane. The assumption is, that the camera is
        mounted in the center of the car, so the center of the image is also the center of the car. This means, that the
        center of the two line at the point closest to the car is the center of the lane.
        :param lines: Dict of Lines ['left': Line(), 'right': Line()}
        :return: Off center distance in meters
        """
        # Get the intersection point between the lines and the lower image edge
        left_line = lines['left']

        left_line_x = left_line.poly_params[0] * (self.IMG_HEIGHT - 1) ** 2 \
                      + left_line.poly_params[1] * (self.IMG_HEIGHT - 1) \
                      + left_line.poly_params[2]

        lane_width_pixels = self._get_lane_width_pixels(lines)
        lane_center_pixels = left_line_x + lane_width_pixels // 2

        pixel_per_meter = lane_width_pixels // self.LANE_WIDTH_METERS

        off_center_pixels = self.IMG_WIDTH // 2 - lane_center_pixels
        off_center_meter = off_center_pixels / pixel_per_meter

        return off_center_meter

    def add_lines(self, lines):
        """
        Add lines to state. Only self.num_last_saved lines are stored in state. If limit is exceeded, older lines are
        dropped.
        Method also adds radius and distance of the car to the center of the lane based on the lines.
        :param lines: Dict of line ['left': Line(), 'right': Line()]
        :return: True if successful, False otherwise
        """
        # Remove earliest lines if too many in state
        if len(self.last_lines) >= self.num_last_saved:
            self.last_lines = self.last_lines[1:]
        try:
            # Add new lines
            self.last_lines.append(lines)

            # Reset num_copied_lines if detection_method != 3
            if (lines['left'].detection_method != 3) and (lines['right'].detection_method != 3):
                self.num_copied_lines = 0

            # Add lane radius
            self.lane_radius = self._get_lane_radius(lines)

            # Add distance that the car is off center of the lane
            self.off_center = self._get_off_center_dist(lines)

            return True
        except:
            return False

    def get_last_lines(self):
        """
        Returns last lines saved in state.
        :return: Dict of lines ['left': Line(), 'right': Line()}, None if no lines saved in state.
        """
        if len(self.last_lines) >= 1:
            return self.last_lines[-1]
        else:
            return None

    def get_last_n_lines(self, n):
        """
        Returns last n lines saved in state.
        :param n: Number of last lines to return
        :return: Array of dict of lines [['left': Line(), 'right': Line()}, ...]
        """
        return self.last_lines[-n:]


def undistort(img, mtx, dist):
    """
    This function undistorts a given image
    :param img: Input image
    :param mtx: Camera matrix
    :param dist: Distortion coefficients
    :return: Undistorted image
    """
    return cv2.undistort(img, mtx, dist, None, mtx)


def warp(img, pers_margin=425, margin_bottom=50, margin_top=450, margin_sides=150, reverse=False):
    """
    This function warps an image. For the transformation a src polygon and a destination
    polygon are used. The source polygon is calculated by the image shape and the margins
    given. The destination polygon is calculated solely on the image shape.
    :param img: Input image
    :param pers_margin: This value determines how sharp the polygon is
    :param margin_bottom: This value sets the distance between the polygon and the bottom of
                          the image
    :param margin_top: This value sets the distance between the polygon and the top of the 
                       image
    :param margin_sides: This value sets the distance between the polygon and the sides of the
                        image
    :param reverse: If True, src and dst will be swapped, thus the image will be unwarped
    :return: Warped image
    """
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
    
    # Compute perspective transform matrix
    if not reverse:
        m = cv2.getPerspectiveTransform(src, dst)
    else:
        m = cv2.getPerspectiveTransform(dst, src)
    
    # Warp image
    warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)
    return warped


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


def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255, erosion=False, dilation=False,
                     er_kernel=np.ones((9, 9), np.uint8), dil_kernel=np.ones((15, 15), np.uint8)):
    """
    This function calculates the sobel for an image and applies a
    binary threshold
    :param img: input image
    :param orient: orientation, can be 'x' or 'y'
    :param thresh_min: minimum binary threshold
    :param thresh_max: maximum binary threshold
    :param erosion: If True, erosion will be applied which reduces noise
    :param dilation: If True, dilation will be applied which closes holes in lines
    :param er_kernel: Kernel used for erosion. Shall be numpy array with dtype=np.uint8
    :param dil_kernel: Kernel used for dilation. Shall be numpy array with dtpye=np.uint8
    :return: binary image
    """
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    
    # 5) Create a mask of 1's where the scaled gradient magnitude
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    # 6) [Optional] Apply erosion
    if erosion:
        binary_output = cv2.erode(binary_output, er_kernel)
        
    # 7) [Optional] Apply dilation
    if dilation:
        binary_output = cv2.dilate(binary_output, dil_kernel)        
    
    return binary_output


def cvt_color_space(img):
    """
    This function extracts the R- and S-channel of the image. Since the S-channel is quite noisy
    a sobel in x-direction is used to filter the noise (bitwise_and). Also both, the filtered S- and the R-channel
    are not working that well for detection yellow lines. Therefor a yellow color mask is added (bitwise_or) to the
    filtered S-channel, before being combined with the R-channel (bitwise_or)
    :param img: Input image
    :return: Binary image
    """
    # R channel
    thresh_r = (200, 255)
    bin_r = channel_thresh(img, channel='r', thresh=thresh_r)
        
    # S channel
    thresh_s = (90, 255)
    bin_s = channel_thresh(img, channel='s', thresh=thresh_s)
    
    # S channel contains a lot of noise. The noise can be filtered
    # by using the sobel and bitwise_and the two images  
    sobel = abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100)
    filtered_s = cv2.bitwise_and(bin_s, sobel)

    # convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow_low = np.array([0, 100, 100], dtype=np.uint8)
    yellow_high = np.array([100, 255, 255], dtype=np.uint8)

    yellow_mask = cv2.inRange(hsv, yellow_low, yellow_high)//255

    yellow = cv2.bitwise_or(filtered_s, yellow_mask)

    # combine R and filtered S channel
    bin_img = cv2.bitwise_or(bin_r, yellow)
    return bin_img


def pipeline(img, mtx, dist):
    """
    The pipeline applies all image pre-processing steps to the image (or video frame).
    1) Undistort the image using the given camera matrix and the distortion coefficients
    2) Warp the image to a bird's-eye view
    3) Apply color space conversion
    :param img: Input image (BGR)
    :param mtx: Camera matrix
    :param dist: Distortion coefficients
    :return: Binary image, ideally only of the lane lines (in bird's-eye view)
    """
    # Undistort image
    dst = undistort(img, mtx, dist)
    
    # Apply perpective transfomation
    warped = warp(dst)
    
    # Apply color space operations
    binary = cvt_color_space(warped)
       
    return binary


def hist_peaks(hist):
    """
    This functions finds two peaks in a histogram. For this the image is split
    into a left and a right part. In each part the max histogram value is searched.
    The position of the max histogram value is returned.
    :param hist: Histogram
    :return: left side max position, right side max position (tuple)
    """
    midpoint = np.int(hist.shape[0]//2)
    left_max = np.argmax(hist[:midpoint])
    right_max = np.argmax(hist[midpoint:])+midpoint
    return left_max, right_max


def get_hist(binary_img):
    """
    This function splits the given image into an upper and a lower half.
    The lower half is used to calculate a histogram an.
    :param binary_img: Binary input image
    :return: Histogram of lower half of input image.
    """
    return np.sum(binary_img[binary_img.shape[0]//2:, :], axis=0)


def compare_values(value1, value2, relative, absolute):
    """
    Compare two values with respect to a relative and an absolute deviation.
    :param value1: First value
    :param value2: Second value
    :param relative: Relative deviation (0..1)
    :param absolute: Absolute deviation (e.g. 1, -5.7, 100)
    :return: True is value1 is within valid deviation of value2, False if not
    """
    mi = min(value1, value2)
    ma = max(value1, value2)

    if ((ma * (1 - relative)) - absolute) < mi:
        return True
    else:
        return False


def check_lines_valid(left_line, right_line, last_left_line=None, last_right_line=None):
    """
    Checks validity of two given lines based on there geometry and optionally based on the deviation to the previous
    detected lines. Also calculates the Mean Absolute error and the Mean Squared Error of the x values between the
    two current lines.
    :param left_line: Left line, Line() object
    :param right_line: Right line, Line() object
    :param last_left_line: Previous left line, Line() object, optional
    :param last_right_line: Previous right line, Line() object, optional
    :return: MAE, MSE, validity (True if lines are valid, False if not)
    """
    ploty = np.linspace(0, IMAGE_HEIGHT - 1, IMAGE_WIDTH)
    left_fitx = left_line.poly_params[0] * ploty ** 2 + left_line.poly_params[1] * ploty + left_line.poly_params[2]
    right_fitx = right_line.poly_params[0] * ploty ** 2 + right_line.poly_params[1] * ploty + right_line.poly_params[2]

    mae = (right_fitx - left_fitx).mean()

    mse = ((right_fitx - left_fitx) ** 2).mean()

    valid = True
    # Check if mae is in a certain threshold
    # This ensures that the lines have a minimum (and maximum) distance
    if (mae < 150) or (mae > 750):
        valid = False

    # Check if params have changed too much from the last frame
    if last_left_line is not None:
        last_frame_comparison = compare_values(left_line.poly_params[0], last_left_line.poly_params[0], .2, 2) \
                                and compare_values(right_line.poly_params[0], last_right_line.poly_params[0], .2, 2) \
                                and compare_values(left_line.poly_params[1], last_left_line.poly_params[1], .2, 2) \
                                and compare_values(right_line.poly_params[1], last_right_line.poly_params[1], .2, 2) \
                                and compare_values(left_line.poly_params[2], last_left_line.poly_params[2], .2, 10) \
                                and compare_values(right_line.poly_params[2], last_right_line.poly_params[2], .2, 10)
        valid = valid and last_frame_comparison

    return mae, mse, valid


def sliding_window(binary_warped, state, leftx_base=None, rightx_base=None):
    """
    Lane detection method. Starting from the left and right base the method searched for line pixels using a sliding
    window approach. Left and right base can either be given or are determined by a histogram (see
    "_find_lane_pixels()" for details). If succesful, lines are added to state.
    :param binary_warped: Input image
    :param state: state, type State()
    :param leftx_base: [optional] Left starting point of the sliding window
    :param rightx_base: [optional] Right starting point of the sliding window
    :return: True if lines were found, False otherwise.
    """

    def _find_lane_pixels(binary_warped, leftx_base=None, rightx_base=None):
        """
        Function searches for lane pixels stating from a base point for both line (left and right) using sliding window.
        The base points can be given or are determined by a histogram approach. For this, the histogram of the lower
        half of the image will be used to extract peak points. Those peaks are then used as the starting points for the
        sliding window method.
        :param binary_warped: Input image
        :param leftx_base: [optional] Left starting point of the sliding window
        :param rightx_base: [optional] Right starting point of the sliding window
        :return: x coordinates of left line, y coordinates of left line, x coordinates of right line,
        y coordinates of right line
        """
        if not (leftx_base or rightx_base):
            # Take a histogram of the bottom half of the image
            histogram = get_hist(binary_warped)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        if not leftx_base:
            leftx_base, _ = hist_peaks(histogram)
        if not rightx_base:
            _, rightx_base = hist_peaks(histogram)

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0] // nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            # Find the four below boundaries of the window
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError as e:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty

    # Find our lane pixels first
    leftx, lefty, rightx, righty = _find_lane_pixels(binary_warped, leftx_base=leftx_base, rightx_base=rightx_base)

    # Fit a second order polynomial to each using `np.polyfit`
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
        # Left line with found poly and detection method=sliding window
        left_line = Line('left', left_fit, 1, line_pixels={"x": leftx, "y": lefty})
        
        right_fit = np.polyfit(righty, rightx, 2)
        # Left line with found poly and detection method=sliding window
        right_line = Line('right', right_fit, 1, line_pixels={"x": rightx, "y": righty})

        # Check if lines are valid
        last_lines = state.get_last_lines()
        
        if last_lines is not None:
            last_left_line = last_lines['left']
            last_right_line = last_lines['right']
            _, _, valid = check_lines_valid(left_line, right_line, last_left_line, last_right_line)
        else:
            _, _, valid = check_lines_valid(left_line, right_line)
            
        if valid:
            # Add found lines to state
            state.add_lines({'left': left_line,
                             'right': right_line})
            return True
        else:
            return False
    except:
        # No line found 
        return False


def search_around_poly(binary_warped, state):
    """
    Lane detection methods. Searches in a area arounf the last lane lines for line pixels and fits a new line through
    those pixels. If successful, lines are added to state.
    :param binary_warped: Input image.
    :param state: State, type State()
    :return: True if lines were found, False otherwise
    """
    # PARAMETER
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Set the area of search based on activated x-values
    # within the +/- margin of our polynomial function
    left_lane_inds = ((nonzerox > (state.get_last_lines()['left'].poly_params[0]*(nonzeroy**2)
                                   + state.get_last_lines()['left'].poly_params[1]*nonzeroy 
                                   + state.get_last_lines()['left'].poly_params[2] - margin)) 
                      & (nonzerox < (state.get_last_lines()['left'].poly_params[0]*(nonzeroy**2) 
                                     + state.get_last_lines()['left'].poly_params[1]*nonzeroy 
                                     + state.get_last_lines()['left'].poly_params[2] + margin)))
    right_lane_inds = ((nonzerox > (state.get_last_lines()['right'].poly_params[0]*(nonzeroy**2) 
                                    + state.get_last_lines()['right'].poly_params[1]*nonzeroy 
                                    + state.get_last_lines()['right'].poly_params[2] - margin)) 
                       & (nonzerox < (state.get_last_lines()['right'].poly_params[0]*(nonzeroy**2) 
                                      + state.get_last_lines()['right'].poly_params[1]*nonzeroy 
                                      + state.get_last_lines()['right'].poly_params[2] + margin)))
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    try:
        # Fit a second order polynomial to each using `np.polyfit`
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        left_line = Line('left', left_fit, 2)
        right_line = Line('right', right_fit, 2)
        
        # Check if lines are valid
        last_left_line = state.get_last_lines()['left']
        last_right_line  = state.get_last_lines()['right']
        
        if (last_left_line and last_right_line) is not None:
            _, _, valid = check_lines_valid(left_line, right_line, last_left_line, last_right_line)
        else:
            _, _, valid = check_lines_valid(left_line, right_line)
            
        if valid:
            # Add found lines to state
            state.add_lines({'left': left_line,
                             'right': right_line})
            return True
        else:
            return False
    except:
        # No line found
        return False


def take_last_lines(state):
    """
    Lane detection methods. Return last detected lines if the maximum number of copied lines was not exceeded already.
    Lines will be added to state.
    :param state: State, type State()
    :return: True if copying work, False otherwise.
    """
    # Check if number of maximum copies was not exceeded
    if state.num_copied_lines < state.copy_limit:
        try:
            # Get all saved last lines
            last_lines = state.get_last_lines()

            # Adopt last poly_params
            last_left_line_params = last_lines['left'].poly_params
            last_right_line_params = last_lines['right'].poly_params

            # Set detection method to 3 = copied
            new_left_line = Line('left', last_left_line_params, 3)
            new_right_line = Line('right', last_right_line_params, 3)

            # Add copied lines to state
            state.add_lines({'left': new_left_line,
                             'right': new_right_line})

            # Increase copied line counter
            state.num_copied_lines += state.num_copied_lines
        except:
            return False
        
        return True
    else:
        return False


def visualize_from_state(img, state, line_color=(255, 0, 0), lane_color=(0, 255, 0), warped=True):
    """
    Generate an output image based on an input image and a state (type State()).
    :param img: Input image
    :param state: State, type State()
    :param line_color: Color of the detected lines. Default (255, 0, 0)
    :param lane_color: Color of the detected lane (polygon between two lanes). Default (0, 255, 0)
    :param warped: Can be used to to display bird's-eye view images (e.g. for debugging).
    If True, output image will be unwarped. If False, output image will not be unwarped.
    :return:
    """
    # Generate output image
    warped_out_img = np.zeros_like(img).astype(np.uint8)
        
    # Plot left and right lane lines
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    
    # Get the last lines from the state
    last_lines = state.get_last_lines()
    if last_lines:
        last_left_line = last_lines['left']
        last_right_line = last_lines['right']

        # Take poly parameters form lines
        left_line_params = last_left_line.poly_params
        right_line_params = last_right_line.poly_params

        left_fitx = left_line_params[0] * ploty ** 2 + left_line_params[1] * ploty + left_line_params[2]
        right_fitx = right_line_params[0] * ploty ** 2 + right_line_params[1] * ploty + right_line_params[2]

        for i in range(len(ploty)-1):
            height, width, _ = img.shape

            left_x1 = int(left_fitx[i])
            left_x2 = int(left_fitx[i+1])
            right_x1 = int(right_fitx[i])
            right_x2 = int(right_fitx[i+1])
            y1 = int(ploty[i])
            y2 = int(ploty[i+1])
            if (left_x1 <= width) and (left_x2 <= width) and (y1 <= height) and (y2 <= height):
                warped_out_img = cv2.line(warped_out_img, (left_x1, y1), (left_x2, y2), line_color, 20)
            if (right_x1 <= width) and (right_x2 <= width) and (y1 <= height) and (y2 <= height):
                warped_out_img = cv2.line(warped_out_img, (right_x1, y1), (right_x2, y2), line_color, 20)

        # Draw lane
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(warped_out_img, np.int_([pts]), lane_color)

        # Draw the line pixels
        if last_left_line.line_pixels is not None:
            leftx = last_left_line.line_pixels['x']
            lefty = last_left_line.line_pixels['y']
            warped_out_img[lefty, leftx] = [255, 0, 0]
        if last_left_line.line_pixels is not None:
            rightx = last_right_line.line_pixels['x']
            righty = last_right_line.line_pixels['y']
            warped_out_img[righty, rightx] = [0, 0, 255]

    if warped:
        out_img = warped_out_img
    else:
        # Warp the blank back to original image space
        out_img = warp(warped_out_img, reverse=True)

    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, out_img, 0.3, 0)

    # Add "radius" and "off center" to image if a valid lane was detected
    if last_lines:
        text = 'Lane radius: {0:.3f} m'.format(state.lane_radius)
        cv2.putText(result, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), thickness=2)

        text = 'Off center: {0:.3f} m'.format(state.off_center)
        cv2.putText(result, text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), thickness=2)
    
    return result


def process_image(img):
    """
    Main function to process video frames. Image will be pre-processed with the image pre-processing pipeline. The
    binary output of the pipeline will then be searched for lane lines. To do that, three different approaches are used.
    1) Sliding window
    2) Search around poly
    3) Take last line (if no other was successful)
    :param img: Input image (video frame)
    :return: Output image. Process video frame
    """
    global state, mtx, dist

    # Preprocess image
    warped_binary = pipeline(img, mtx, dist)
    
    # First frame is always sliding window with default starting points
    last_lines = state.get_last_lines()
    if last_lines is None:        
        # First frame "sliding window"
        sliding_window(warped_binary, state, leftx_base=None, rightx_base=None)
    # ALl further frames should be "search around poly" if the last line was detected by
    # sliding window or search around poly
    elif last_lines['left'].detection_method and last_lines['right'].detection_method in (1, 2):
        # Try "search around poly" first
        if not search_around_poly(warped_binary, state):
            # If "search around poly" doesn't work, try "sliding window"
            # TODO: replace leftx_base and rightx_base by last values
            if not sliding_window(warped_binary, state, leftx_base=None, rightx_base=None):
                pass
                # If sliding window doesn't work either, try to take last lines
                if not take_last_lines(state):
                    # If that doesn't work either, no line can be found
                    # TODO: add "None-Line"
                    pass
    # Last lines were copied or not found
    else:
        # If last lines were copied, try to start with sliding window again
        if not sliding_window(warped_binary, state, leftx_base=None, rightx_base=None):
            pass
            # If sliding window doesn't work either, try to take last lines
            if not take_last_lines(state):
                # If that doesn't work either, no line can be found
                # TODO: add "None-Line"
                pass
        
    out = visualize_from_state(img, state, warped=False)
    
    # Debug: Uncomment to see bird's-eye binary image
    #out = visualize_from_state(np.dstack((warped_binary*255, warped_binary*255, warped_binary*255)), state, warped=True)
    return out


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Script will detect lane lines in a given video.')
    parser.add_argument('input_video', help='Path to video clip')
    parser.add_argument('output_video', help='Path to output video clip')
    parser.add_argument('--t_start', help='Start time of video processing. Format \'01:03:05.3\'', default=0)
    parser.add_argument('--t_end', help='End time of video processing. Format \'01:03:05.3\'', default=None)
    args = parser.parse_args()

    # Load camera calibration paramters
    with open('camera_calibration.p', 'rb') as file:
        ret, mtx, dist, rvecs, tvecs = pickle.load(file)

    # Process video
    state = State()
    clip_in = VideoFileClip(args.input_video).subclip(args.t_start, args.t_end)
    clip = clip_in.fl_image(process_image)
    clip.write_videofile(args.output_video, audio=False)
