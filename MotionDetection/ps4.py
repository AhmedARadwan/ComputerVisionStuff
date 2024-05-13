"""Problem Set 4: Motion Detection"""

import cv2
import numpy as np



# Utility function
def read_video(video_file, show=False):
    """Reads a video file and outputs a list of consecuative frames
  Args:
      image (string): Video file path
      show (bool):    Visualize the input video. WARNING doesn't work in
                      notebooks
  Returns:
      list(numpy.ndarray): list of frames
  """
    frames = []
    cap = cv2.VideoCapture(video_file)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

        # Opens a new window and displays the input
        if show:
            cv2.imshow("input", frame)
            # Frames are read by intervals of 1 millisecond. The
            # programs breaks out of the while loop when the
            # user presses the 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # The following frees up resources and
    # closes all windows
    cap.release()
    if show:
        cv2.destroyAllWindows()
    return frames
    
def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to [0, 255].

    Returns:
        numpy.array: output image.
    """
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out


# Assignment code
def gradient_x(image):
    """Computes image gradient in X direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the X direction. Output
                     from cv2.Sobel.
    """
    return cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3, scale=1/8)


def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction.
                     Output from cv2.Sobel.
    """
    return cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3, scale=1.0/8.0)

def median_filter_np(arr, size):
    pad_width = size // 2
    padded_arr = np.pad(arr, pad_width, mode='edge')
    filtered_arr = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            filtered_arr[i, j] = np.median(padded_arr[i:i+size, j:j+size])
    return filtered_arr

def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method.

    Note: Implement this method using the instructions in the lectures
    and the documentation.

    You are not allowed to use any OpenCV functions that are related
    to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. Here we assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use
                      'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1 because the autograder does not
                       use this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """
    I_x = gradient_x(img_a)
    I_y = gradient_y(img_a)
    I_t = img_b - img_a

    kernel = None
    if k_type == 'uniform':
        kernel = np.ones((k_size, k_size), dtype=np.float32) / (k_size**2)
    elif k_type == 'gaussian':
        kernel = cv2.getGaussianKernel(k_size, sigma) @ cv2.getGaussianKernel(k_size, sigma).T
    else:
        kernel = np.ones((k_size, k_size), dtype=np.float32) / (k_size**2)
    I_x2 = cv2.filter2D(I_x**2, -1, kernel)
    I_y2 = cv2.filter2D(I_y**2, -1, kernel)
    I_xy = cv2.filter2D(I_x * I_y, -1, kernel)
    I_xt = cv2.filter2D(I_x * I_t, -1, kernel)
    I_yt = cv2.filter2D(I_y * I_t, -1, kernel)

    u = np.zeros_like(img_a)
    v = np.zeros_like(img_a)

    for y in range(img_a.shape[0]):
        for x in range(img_a.shape[1]):
            M = np.array([[I_x2[y, x],  I_xy[y, x]],
                          [I_xy[y, x],  I_y2[y, x]]])
            b = np.array([-I_xt[y, x], -I_yt[y, x]])
            if np.linalg.matrix_rank(M) == 2:
                uv = np.linalg.inv(M) @ b
                u[y, x] = uv[0]
                v[y, x] = uv[1]
    
    # # removing anomalies using a median filter
    # m_kernel_size = 9

    # u = median_filter_np(u, size=m_kernel_size)
    # v = median_filter_np(v, size=m_kernel_size)
    return u, v


def reduce_image(image):
    """Reduces an image to half its shape.

    The autograder will pass images with even width and height. It is
    up to you to determine values with odd dimensions. For example the
    output image can be the result of rounding up the division by 2:
    (13, 19) -> (7, 10)

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].

    Returns:
        numpy.array: output image with half the shape, same type as the
                     input image.
    """
    kernel = np.array([1, 4, 6, 4, 1]) / 16

    kernel_h = np.reshape(kernel, (1, 5))
    kernel_v = np.reshape(kernel, (5, 1))
    
    conv_h = cv2.filter2D(image, -1, kernel_h, borderType=cv2.BORDER_REFLECT_101)
    conv_v = cv2.filter2D(conv_h, -1, kernel_v, borderType=cv2.BORDER_REFLECT_101)
    
    reduce_image = np.zeros((conv_v.shape[0]//2, conv_v.shape[1]//2), dtype=conv_v.dtype)
    for i in range(0, conv_v.shape[0]-1, 2):
        for j in range(0, conv_v.shape[1]-1, 2):
            reduce_image[i//2, j//2] = conv_v[i, j]
    return reduce_image


def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.

    This method uses reduce_image() at each level. Each image is
    stored in a list of length equal the number of levels.

    The first element in the list ([0]) should contain the input
    image. All other levels contain a reduced version of the previous
    level.

    All images in the pyramid should floating-point with values in 
    [0.0, 1.0].

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """

    pyramid = []
    for i in range(0, levels):
        if i == 0:
            pyramid.append(image.astype(np.float32))
        else:
            reduced_image = reduce_image(pyramid[-1]).astype(np.float32)
            pyramid.append(reduced_image)
    return pyramid


def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side.

    Ordering should be large to small from left to right.

    See the problem set instructions for a reference on how the output
    should look like.

    Make sure you call normalize_and_scale() for each image in the
    pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked
                     from left to right.
    """
    normalized_imgs = []
    for img in img_list:
        normalized_imgs.append(normalize_and_scale(img.astype(np.float64)))

    heights = []
    widths = []
    for i in range(len(normalized_imgs)):
        heights.append(normalized_imgs[i].shape[0])
        widths.append(normalized_imgs[i].shape[1])

    output_height = max(heights)
    output_width = sum(widths)

    output_img = np.zeros((output_height, output_width), dtype=np.uint8)

    current_x = 0
    for img in normalized_imgs:
        height, width = img.shape[:2]
        output_img[:height, current_x:current_x+width] = img
        current_x += width
    return output_img


def expand_image(image):
    """Expands an image doubling its width and height.

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code

    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].

    Returns:
        numpy.array: same type as 'image' with the doubled height and
                     width.
    """
    
    # based on what I understood here -> https://edstem.org/us/courses/51358/discussion/4363017?comment=10167969
    kernel = np.array([1, 4, 6, 4, 1]) / 8
    output_height, output_width = image.shape[0]*2, image.shape[1]*2

    expanded_image = np.zeros((output_height, output_width), dtype=image.dtype)
    
    for i in range(0, expanded_image.shape[0], 2):
        for j in range(0, expanded_image.shape[1], 2):
            expanded_image[i, j] = image[i//2, j//2]
    
    expanded_image = cv2.sepFilter2D(expanded_image, -1, kernel, kernel)
    return expanded_image


def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """
    l_pyr = []
    for i in range(1, len(g_pyr)):
        expanded = expand_image(g_pyr[i])
        l_pyr.append(g_pyr[i-1][:expanded.shape[0], :expanded.shape[1]] - expanded)
    l_pyr.append(g_pyr[-1])
    return l_pyr


def warp(image, U, V, interpolation, border_mode):
    """Warps image using X and Y displacements (U and V).

    This function uses cv2.remap. The autograder will use cubic
    interpolation and the BORDER_REFLECT101 border mode. You may
    change this to work with the problem set images.

    See the cv2.remap documentation to read more about border and
    interpolation methods.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        U (numpy.array): displacement (in pixels) along X-axis.
        V (numpy.array): displacement (in pixels) along Y-axis.
        interpolation (Inter): interpolation method used in cv2.remap.
        border_mode (BorderType): pixel extrapolation method used in
                                  cv2.remap.

    Returns:
        numpy.array: warped image, such that
                     warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """

    height, width = image.shape[:2]
    U_resized = cv2.resize(U, (width, height))
    V_resized = cv2.resize(V, (width, height))
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    
    remap_x = np.clip(x_coords + U_resized, 0, width - 1)
    remap_y = np.clip(y_coords + V_resized, 0, height - 1)
    warped_image = cv2.remap(image, remap_x.astype(np.float32), remap_y.astype(np.float32), interpolation, borderMode=border_mode)

    return warped_image


def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation,
                    border_mode):
    """Computes the optic flow using Hierarchical Lucas-Kanade.

    This method should use reduce_image(), expand_image(), warp(),
    and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis,
                             same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along Y-axis,
                             same size and type as U.
    """
    pyramid_a = [img_a]
    pyramid_b = [img_b]
    for i in range(levels - 1):
        img_a = reduce_image(img_a)
        img_b = reduce_image(img_b)
        pyramid_a.append(img_a)
        pyramid_b.append(img_b)
    
    # init U_k+1 and V_k+1 with zeros
    U, V = np.zeros_like(pyramid_a[-1]), np.zeros_like(pyramid_a[-1])

    for level in range(levels - 1, -1, -1):
        
        # upsample u_i+1 and v_i+1 to create u_i and v_i
        U = expand_image(U)
        V = expand_image(V)
        
        # multiply u_i and v_i by 2
        U *= 2
        V *= 2

        # warp level i gaussian version of I2 according to predicted flow to create I2_warped
        U = cv2.resize(U, (pyramid_a[level].shape[1], pyramid_a[level].shape[0]))
        V = cv2.resize(V, (pyramid_a[level].shape[1], pyramid_a[level].shape[0]))

        I2_warped = warp(pyramid_b[level], U, V, interpolation, border_mode)
        u, v = optic_flow_lk(pyramid_a[level], I2_warped, k_size, k_type, sigma)
        
        U += u
        V += v
    
    return U, V



def classify_video(images):
    """Classifies a set of frames as either
        - int(1) == "Running"
        - int(2) == "Walking"
        - int(3) == "Clapping"
    Args:
        images list(numpy.array): greyscale floating-point frames of a video
    Returns:
        int:  Class of video
    """

    u_avg_list = []
    v_avg_list = []
    for frame in range(50, len(images)-50, 1):
        levels = 3
        k_size = 15
        k_type = "uniform"
        sigma = 0
        interpolation = cv2.INTER_CUBIC
        border_mode = cv2.BORDER_REFLECT101

        prev_frame = images[frame-4]
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        current_frame = images[frame]
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        u, v = hierarchical_lk(prev_frame, current_frame, levels, k_size,
                               k_type, sigma, interpolation, border_mode)
        
        u_avg = abs(np.average(u[np.nonzero(u)]))
        v_avg = abs(np.average(v[np.nonzero(v)]))
        u_avg_list.append(u_avg)
        v_avg_list.append(v_avg)
        print("frame: {}, u_avg: {}, v_avg: {}".format(frame, u_avg, v_avg))
    
    u_avg = np.average(u_avg_list)
    v_avg = np.average(v_avg_list)
    print("u_avg: {}, v_avg: {}".format(u_avg, v_avg))

    predicted_labels = None

    # I don't know another way to load the model other than using pickle
    # kindly uncomment if you want to test the code

    # import pickle
    # with open('decision_tree_classifier.pkl', 'rb') as f:
    #     clf = pickle.load(f)
    # predicted_labels = clf.predict([[u_avg, v_avg]])

    if predicted_labels[0] == 'running':
        return 1
    elif predicted_labels[0] == 'walking':
        return 2
    else:
        return 3