import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import fftpack


def traffic_light_detection(img_in, radii_range):
    """Finds the coordinates of a traffic light image given a radii
    range.
    Use the radii range to find the circles in the traffic light and
    identify which of them represents the yellow light.
    Analyze the states of all three lights and determine whether the
    traffic light is red, yellow, or green. This will be referred to
    as the 'state'.
    It is recommended you use Hough tools to find these circles in
    the image.
    The input image may be just the traffic light with a white
    background or a larger image of a scene containing a traffic
    light.
    Args:
        img_in (numpy.array): image containing a traffic light.
        radii_range (list): range of radii values to search for.
    Returns:
        tuple: 2-element tuple containing:
        coordinates (tuple): traffic light center using the (x, y)
                             convention.
        state (str): traffic light state. A value in {'red', 'yellow',
                     'green'}
    """
    gray_img = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 2)

    circles = cv2.HoughCircles(
        blurred_img,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=min(radii_range),
        maxRadius=max(radii_range)
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            x, y, r = i[0], i[1], i[2]
            roi = img_in[y - r:y + r, x - r:x + r]

            mean_color = np.mean(roi, axis=(0, 1))

            red_threshold = 100
            green_threshold = 100

            if mean_color[0] > red_threshold:
                state = 'red'
            elif mean_color[1] > green_threshold:
                state = 'green'
            else:
                state = 'yellow'
        
            return (x, y), state

    return None


def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.
    Args:
        img_in (numpy.array): image containing a traffic light.
    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    hsv_img = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds of the color range for the construction sign
    lower_color = np.array([0, 100, 100])
    upper_color = np.array([20, 255, 255])

    # Create a mask using the inRange function to extract the construction sign
    mask = cv2.inRange(hsv_img, lower_color, upper_color)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour (assuming it's the construction sign)
        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate the centroid of the contour
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy

    # Return None if no construction sign is found
    return None


def template_match(img_orig, img_template, method):
    """Returns the location corresponding to match between original image and provided template.
    Args:
        img_orig (np.array) : numpy array representing 2-D image on which we need to find the template
        img_template: numpy array representing template image which needs to be matched within the original image
        method: corresponds to one of the four metrics used to measure similarity between template and image window
    Returns:
        Co-ordinates of the topmost and leftmost pixel in the result matrix with maximum match
    """
    """Each method is calls for a different metric to determine
       the degree to which the template matches the original image
       We are required to implement each technique using the
       sliding window approach.
       Suggestion : For loops in python are notoriously slow
       Can we find a vectorized solution to make it faster?
    """
    result = np.zeros(
        (
            (img_orig.shape[0] - img_template.shape[0] + 1),
            (img_orig.shape[1] - img_template.shape[1] + 1),
        ),
        dtype=float,
    )
    top_left = []
    print("Matching with Method: ", method)
    # Sum of squared differences
    if method == "tm_ssd":
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = np.sum((img_orig[i:i+img_template.shape[0], j:j+img_template.shape[1]] - img_template)**2)
        top_left.append(np.unravel_index(np.argmin(result), result.shape)[1])
        top_left.append(np.unravel_index(np.argmin(result), result.shape)[0])

    # Normalized sum of squared differences
    elif method == "tm_nssd":
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = np.sum((img_orig[i:i+img_template.shape[0], j:j+img_template.shape[1]] - img_template)**2)
        result = result / (np.linalg.norm(img_orig[i:i+img_template.shape[0], j:j+img_template.shape[1]]) * np.linalg.norm(img_template))
        top_left.append(np.unravel_index(np.argmin(result), result.shape)[1])
        top_left.append(np.unravel_index(np.argmin(result), result.shape)[0])

    # Cross Correlation
    elif method == "tm_ccor":
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = np.sum((img_orig[i:i+img_template.shape[0], j:j+img_template.shape[1]] * img_template))
        top_left.append(np.unravel_index(np.argmax(result), result.shape)[1])
        top_left.append(np.unravel_index(np.argmax(result), result.shape)[0])

    # Normalized Cross Correlation
    elif method == "tm_nccor":
        # result = cv2.matchTemplate(img_orig, img_template, cv2.TM_CCORR_NORMED)
        # max_loc = cv2.minMaxLoc(result)
        # top_left = max_loc[3]
        img_orig_mean = np.mean(img_orig)
        img_template_mean = np.mean(img_template)
        
        # Calculate the normalized cross-correlation
        img_orig_norm = img_orig - img_orig_mean
        img_template_norm = img_template - img_template_mean
        result = np.zeros_like(result)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                window = img_orig[i:i+img_template.shape[0], j:j+img_template.shape[1]]
                window_norm = window - np.mean(window)
                result[i, j] = np.sum(window_norm * img_template_norm) / (np.linalg.norm(window_norm) * np.linalg.norm(img_template_norm))
        top_left.append(np.unravel_index(np.argmax(result), result.shape)[1])
        top_left.append(np.unravel_index(np.argmax(result), result.shape)[0])
        

    else:
        raise ValueError("Invalid method")
    
    return top_left


'''Below is the helper code to print images for the report'''
#     cv2.rectangle(img_orig,top_left, bottom_right, 255, 2)
#     plt.subplot(121),plt.imshow(result,cmap = 'gray')
#     plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#     plt.subplot(122),plt.imshow(img_orig,cmap = 'gray')
#     plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
#     plt.suptitle(method)
#     plt.show()


def dft(x):
    """Discrete Fourier Transform for 1D signal
    Args:
        x (np.array): 1-dimensional numpy array of shape (n,) representing signal
    Returns:
        y (np.array): 1-dimensional numpy array of shape (n,) representing Fourier Transformed Signal

    """
    return np.fft.fft(x)


def idft(x):
    """Inverse Discrete Fourier Transform for 1D signal
    Args:
        x (np.array): 1-dimensional numpy array of shape (n,) representing Fourier-Transformed signal
    Returns:
        y (np.array): 1-dimensional numpy array of shape (n,) representing signal

    """
    return np.fft.ifft(x)


def dft2(img):
    """Discrete Fourier Transform for 2D signal
    Args:
        img (np.array): 2-dimensional numpy array of shape (n,m) representing image
    Returns:
        y (np.array): 2-dimensional numpy array of shape (n,m) representing Fourier-Transformed image

    """
    return np.fft.fft2(img)


def idft2(img):
    """Inverse Discrete Fourier Transform for 2D signal
    Args:
        img (np.array): 2-dimensional numpy array of shape (n,m) representing Fourier-Transformed image
    Returns:
        y (np.array): 2-dimensional numpy array of shape (n,m) representing image

    """
    return np.fft.ifft2(img)

def dft2_(img):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))


def compress_image_fft(img_bgr, threshold_percentage):
    """Return compressed image by converting to fourier domain, thresholding based on threshold percentage, and converting back to fourier domain
    Args:
        img_bgr (np.array): numpy array of shape (n,m,3) representing bgr image
        threshold_percentage (float): between 0 and 1 representing what percentage of Fourier image to keep
    Returns:
        img_compressed (np.array): numpy array of shape (n,m,3) representing compressed image. (Make sure the data type of the np array is float64)
        compressed_frequency_img (np.array): numpy array of shape (n,m,3) representing the compressed image in the frequency domain

    """
    f_transform_r = dft2(img_bgr[:, :, 2])
    f_transform_g = dft2(img_bgr[:, :, 1])
    f_transform_b = dft2(img_bgr[:, :, 0])

    magnitude_r = np.abs(f_transform_r)
    magnitude_g = np.abs(f_transform_g)
    magnitude_b = np.abs(f_transform_b)

    total_pixels = img_bgr.shape[0] * img_bgr.shape[1]

    sorted_magnitude_r = np.sort(magnitude_r, axis=None)[::-1]
    sorted_magnitude_g = np.sort(magnitude_g, axis=None)[::-1]
    sorted_magnitude_b = np.sort(magnitude_b, axis=None)[::-1]


    threshold_index_r = int(total_pixels * threshold_percentage)
    threshold_index_g = int(total_pixels * threshold_percentage)
    threshold_index_b = int(total_pixels * threshold_percentage)

    threshold_value_r = sorted_magnitude_r[threshold_index_r]
    threshold_value_g = sorted_magnitude_g[threshold_index_g]
    threshold_value_b = sorted_magnitude_b[threshold_index_b]

    mask_r = magnitude_r > threshold_value_r
    mask_g = magnitude_g > threshold_value_g
    mask_b = magnitude_b > threshold_value_b

    compressed_image_r = f_transform_r * mask_r
    compressed_image_g = f_transform_g * mask_g
    compressed_image_b = f_transform_b * mask_b

    reconstructed_image_r = idft2(compressed_image_r).real + 1j * idft2(compressed_image_r).imag
    reconstructed_image_g = idft2(compressed_image_g).real + 1j * idft2(compressed_image_g).imag
    reconstructed_image_b = idft2(compressed_image_b).real + 1j * idft2(compressed_image_b).imag

    img_compressed = np.stack((reconstructed_image_b,
                               reconstructed_image_g,
                               reconstructed_image_r), axis=-1).astype(np.float64)
    compressed_frequency_img = np.stack((compressed_image_b, compressed_image_g, compressed_image_r), axis=-1).astype(np.float64)
    
    return img_compressed, compressed_frequency_img

def low_pass_filter(img_bgr, r):
    """Return low pass filtered image by keeping a circle of radius r centered on the frequency domain image
    Args:
        img_bgr (np.array): numpy array of shape (n,m,3) representing bgr image
        r (float): radius of low pass circle
    Returns:
        img_low_pass (np.array): numpy array of shape (n,m,3) representing low pass filtered image. (Make sure the data type of the np array is float64)
        low_pass_frequency_img (np.array): numpy array of shape (n,m,3) representing the low pass filtered image in the frequency domain

    """

    r_channel, g_channel, b_channel = img_bgr[:,:,2], img_bgr[:,:,1], img_bgr[:,:,0]

    img_low_pass_r, low_pass_frequency_r = low_pass_filter_single_channel(r_channel, r)
    img_low_pass_g, low_pass_frequency_g = low_pass_filter_single_channel(g_channel, r)
    img_low_pass_b, low_pass_frequency_b = low_pass_filter_single_channel(b_channel, r)

    img_low_pass = np.stack((img_low_pass_b, img_low_pass_g, img_low_pass_r), axis=-1).astype(np.float64)
    low_pass_frequency_img = np.stack((low_pass_frequency_b, low_pass_frequency_g, low_pass_frequency_r), axis=-1).astype(np.float64)

    return img_low_pass, low_pass_frequency_img

def low_pass_filter_single_channel(channel, r):
    fft_channel = fftpack.fftshift(fftpack.fft2(channel))

    x, y = channel.shape[0], channel.shape[1]
    xx, yy = np.meshgrid(np.arange(x), np.arange(y))
    center_x, center_y = x // 2, y // 2
    distance = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)

    low_pass = np.where(distance <= r, 1, 0)

    filtered_channel = fft_channel * low_pass.T
    ifft_channel = np.real(fftpack.ifft2(fftpack.ifftshift(filtered_channel)))

    img_low_pass_channel = np.maximum(0, np.minimum(ifft_channel, 255)).astype(np.float64)

    return img_low_pass_channel, filtered_channel
    

