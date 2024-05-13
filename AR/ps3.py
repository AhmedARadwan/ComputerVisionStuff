"""
CS6476 Assignment 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np

import cv2
import numpy as np
from typing import Tuple
from scipy import ndimage
from scipy.signal import convolve2d


class Mouse_Click_Correspondence(object):

    def __init__(self,path1='',path2='',img1='',img2=''):
        self.sx1 = []
        self.sy1 = []
        self.sx2 = []
        self.sy2 = []
        self.img = img1
        self.img2 = img2
        self.path1 = path1
        self.path2 = path2


    def click_event(self,event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print('x y', x, ' ', y)

            sx1=self.sx1
            sy1=self.sy1

            sx1.append(x)
            sy1.append(y)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.img, str(x) + ',' +
                        str(y), (x, y), font,
                        1, (255, 0, 0), 2)
            cv2.imshow('image 1', self.img)

            # checking for right mouse clicks
        if event == cv2.EVENT_RBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print(x, ' ', y)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            b = self.img[y, x, 0]
            g = self.img[y, x, 1]
            r = self.img[y, x, 2]
            cv2.putText(self.img, str(b) + ',' +
                        str(g) + ',' + str(r),
                        (x2, y2), font, 1,
                        (255, 255, 0), 2)
            cv2.imshow('image 1', self.img)

        # driver function

    def click_event2(self,event2, x2, y2, flags, params):
        # checking for left mouse clicks
        if event2 == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print('x2 y2', x2, ' ', y2)

            sx2= self.sx2
            sy2 = self.sy2

            sx2.append(x2)
            sy2.append(y2)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.img2, str(x2) + ',' +
                        str(y2), (x2, y2), font,
                        1, (0, 255, 255), 2)
            cv2.imshow('image 2', self.img2)

            # checking for right mouse clicks
        if event2 == cv2.EVENT_RBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print(x2, ' ', y2)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            b = self.img2[y2, x2, 0]
            g = self.img2[y2, x2, 1]
            r = self.img2[y2, x2, 2]
            cv2.putText(self.img2, str(b) + ',' +
                        str(g) + ',' + str(r),
                        (x2, y2), font, 1,
                        (255, 255, 0), 2)
            cv2.imshow('image 2', self.img2)

    # driver function
    def driver(self,path1,path2):
        # reading the image
        # path = r'D:\GaTech\TA - CV\ps05\ps05\ps5-1-b-1.png'
        #path1 = r'1a_notredame.jpg'
        #path2 = r'1b_notredame.jpg'


        #path1 = self.path1
        #path2 = self.path2

        # path1 = r'crop1.jpg'
        # path2 = r'crop2.jpg'

        self.img = cv2.imread(path1, 1)
        self.img2 = cv2.imread(path2, 2)

        # displaying the image
        cv2.namedWindow("image 1", cv2.WINDOW_NORMAL)
        cv2.imshow('image 1', self.img)
        cv2.namedWindow("image 2", cv2.WINDOW_NORMAL)
        cv2.imshow('image 2', self.img2)

        # setting mouse hadler for the image
        # and calling the click_event() function
        cv2.setMouseCallback('image 1', self.click_event)
        cv2.setMouseCallback('image 2', self.click_event2)

        # wait for a key to be pressed to exit

        cv2.waitKey(0)
        # close the window
        cv2.destroyAllWindows()

        print('sx1 sy1', self.sx1, self.sy1)
        print('sx2 sy2', self.sx2, self.sy2)

        points1, points2 = [], []
        for x, y in zip(self.sx1, self.sy1):
            points1.append((x, y))

        points_1 = np.array(points1)

        for x, y in zip(self.sx2, self.sy2):
            points2.append((x, y))

        points_2 = np.array(points2)

        np.save('p1-Manual.npy', points_1)
        np.save('p2-Manual.npy', points_2)



def euclidean_distance(p0, p1):
    """Get the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1
        p1 (tuple): Point 2
    Return:
        float: The distance between points
    """

    x0, y0 = p0
    x1, y1 = p1
    return np.sqrt((x1 - x0)**2 + (y1 - y0)**2)


def get_corners_list(image):
    """List of image corner coordinates used in warping.

    Args:
        image (numpy.array of float64): image array.
    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right]
    """
    height, width = image.shape[:2]
    corners = [(0, 0), (0, height-1), (width-1, 0), (width-1, height-1)]
    return corners

# # Function to rotate image by a specified angle
# def rotate_image(image, angle):
#     img = image.copy()
#     rotated_image = ndimage.rotate(img, angle, reshape=False, mode='nearest')
#     return rotated_image


# def find_markers(image, template=None):
#     """Finds four corner markers.

#     Use a combination of circle finding and convolution to find the
#     four markers in the image.

#     Reference: 
#     https://docs.opencv.org/3.4/d4/dc6/tutorial_py_template_matching.html

#     Args:
#         image (numpy.array of uint8): image array.
#         template (numpy.array of unint8): template of the markers
#     Returns:
#         list: List of four (x, y) tuples
#             in the order [top-left, bottom-left, top-right, bottom-right]
#     """
#     # method:
#     # for each angle from -90 to 90 with a step of 5
#     #   getting matches using cv2.matchTemplate
#     #   get max score with each angle from -90 to 90
#     #   append to scores list
#     # get angle with max score
#     # rotate template by max score angle
#     # getting locations of possible matches that exceed a threshold (tunable)
#     # sorting by score
#     # removing duplicates using IOU
#     # get the top four locations
#     # get the centroid of the top four locations
#     # get the top left, bottom left, top right, bottom right points
    
#     scores = []
    
#     # get max score with each angle from -90 to 90
#     template_ = template.copy()
#     for i in range(-90, 91, 5):
#         template_ = rotate_image(template, i)
#         result = cv2.matchTemplate(image, template_, cv2.TM_CCOEFF_NORMED)
#         scores.append([np.max(result), i])

#     max_score = float('-inf')
#     max_i = None
#     for score, index in scores:
#         if score > max_score:
#             max_score = score
#             max_i = index
#     print("max_score: ", max_score, "max_i: ", max_i)


#     template = rotate_image(template, max_i)

#     result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
#     loc = np.where(result >= 0.3)
#     locations = list(zip(*loc[::-1]))
#     locations_with_scores = [(x[0], x[1], result[x[1], x[0]]) for x in locations]
#     locations_with_scores.sort(key=lambda x: x[2], reverse=True)

#     top_locations, top_votes = remove_duplicate_matches(locations_with_scores[:100], template, iou_threshold=0.80)

#     top_four_locations = top_locations[:4]
#     top_four_locations = (np.array(top_four_locations)[:, :2]).astype(int)
#     top_four_locations += int(template.shape[0] / 2)

#     centroid_x = sum(point[0] for point in top_four_locations) / len(top_four_locations)
    
#     left_points = [point for point in top_four_locations if point[0] < centroid_x]
#     print("centroid_x: ", centroid_x)
#     print("left_points: ", left_points)
#     right_points = [point for point in top_four_locations if point[0] > centroid_x]
#     top_left = left_points[0] if left_points[0][1] < left_points[1][1] else left_points[1]
#     bottom_left = left_points[0] if left_points[0][1] > left_points[1][1] else left_points[1]
#     top_right = right_points[0] if right_points[0][1] < right_points[1][1] else right_points[1]
#     bottom_right = right_points[0] if right_points[0][1] > right_points[1][1] else right_points[1]

#     return (top_left, bottom_left, top_right, bottom_right)

def find_markers(image, template=None):
    """Finds four corner markers.

    Use a combination of circle finding and convolution to find the
    four markers in the image.

    Reference: 
    https://docs.opencv.org/3.4/d4/dc6/tutorial_py_template_matching.html

    Args:
        image (numpy.array of uint8): image array.
        template (numpy.array of unint8): template of the markers
    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right]
    """
    
    scores = []
    template_ = template.copy()
    for i in range(0, 2):
        if i == 1:
            template_ = cv2.rotate(template, cv2.ROTATE_90_COUNTERCLOCKWISE)
        result = cv2.matchTemplate(image, template_, cv2.TM_CCOEFF_NORMED)
        scores.append([np.max(result), i])

    max_score = float('-inf')
    max_i = None
    for score, index in scores:
        if score > max_score:
            max_score = score
            max_i = index

    # rotate the template if the max score is at 90 degrees
    if max_i == 1:
        template = cv2.rotate(template, cv2.ROTATE_90_COUNTERCLOCKWISE)

    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= 0.2)
    locations = list(zip(*loc[::-1]))
    locations.sort(key=lambda x: result[x[1], x[0]], reverse=True)

    top_locations = remove_duplicate_matches(locations[:1000], template)

    top_four_locations = top_locations[:4]
    top_four_locations = np.array(top_four_locations)
    top_four_locations += int(template.shape[0] / 2)

    centroid = np.mean(top_four_locations, axis=0)

    top_four_locations_sorted = [None] * 4

    top_four_locations_sorted = top_four_locations.copy()

    for point in top_four_locations:
        if point[0] < centroid[0] and point[1] < centroid[1]:
            top_four_locations_sorted[0] = point
        elif point[0] < centroid[0] and point[1] > centroid[1]:
            top_four_locations_sorted[1] = point
        elif point[0] > centroid[0] and point[1] < centroid[1]:
            top_four_locations_sorted[2] = point
        else:
            top_four_locations_sorted[3] = point
    

    return tuple(top_four_locations_sorted)

def calculate_iou(box1, box2):
    x1_int = max(box1[0], box2[0])
    y1_int = max(box1[1], box2[1])
    x2_int = min(box1[2], box2[2])
    y2_int = min(box1[3], box2[3])

    intersection_area = max(0, x2_int - x1_int + 1) * max(0, y2_int - y1_int + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou

# def remove_duplicate_matches(matches_list, template, iou_threshold=0.75):
#     unique_matches = []
#     votes = []
#     for match in matches_list:
#         is_unique = True
#         for i, unique_match in enumerate(unique_matches):
#             iou_score = calculate_iou(
#                 (match[0], match[1], match[0] + template.shape[1], match[1] + template.shape[0]),
#                 (unique_match[0], unique_match[1], unique_match[0] + template.shape[1], unique_match[1] + template.shape[0])
#             )
#             if iou_score > iou_threshold:
#                 is_unique = False
#                 votes[i] += 1
#                 break
#         if is_unique:
#             unique_matches.append(match)
#             votes.append(1)

#     unique_matches_sorted = [x for _, x in sorted(zip(votes, unique_matches), reverse=True)]
#     votes_sorted = sorted(votes, reverse=True)

#     return unique_matches_sorted, votes_sorted

def remove_duplicate_matches(matches_list, template):
    unique_matches = []
    for match in matches_list:
        is_unique = True
        for unique_match in unique_matches:
            iou_score = calculate_iou((match[0], match[1], match[0] + template.shape[1], match[1] + template.shape[0]),
                                      (unique_match[0], unique_match[1], unique_match[0] + template.shape[1], unique_match[1] + template.shape[0]))
            if iou_score > 0.1:
                is_unique = False
                break
        if is_unique:
            unique_matches.append(match)

    return unique_matches
    


def draw_box(image, markers, thickness=1):
    """Draw 1-pixel width lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line and leave the default "thickness" and "lineType".

    Args:
        image (numpy.array of uint8): image array
        markers(list of tuple): the points where the markers were located
        thickness(int): thickness of line used to draw the boxes edges
    Returns:
        numpy.array: image with lines drawn.
    """

    out_image = image.copy()

    for i in range(len(markers)):
        cv2.line(out_image, markers[i], markers[(i + 1) % len(markers)], (0, 255, 0), thickness)

    return out_image


def project_imageA_onto_imageB(imageA, imageB, homography):
    """Using the four markers in imageB, project imageA into the marked area.

    You should have used your find_markers method to find the corners and then
    compute the homography matrix prior to using this function.

    Args:
        image (numpy.array of uint8): image array
        image (numpy.array of uint8): image array
        homography (numpy.array): Perspective transformation matrix, 3 x 3
    Returns:
        numpy.array: combined image
    """

    out_image = imageB.copy()
    rows, cols = imageA.shape[:2]

    for y in range(rows):
        for x in range(cols):
            
            transformed = np.dot(homography, [x, y, 1])
            u, v, w = transformed / transformed[2]

            u, v = int(round(u)), int(round(v))

            if 0 <= u < out_image.shape[1] and 0 <= v < out_image.shape[0]:
                out_image[v, u] = imageA[y, x]

    return out_image


def find_four_point_transform(srcPoints, dstPoints):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform
    Hint: You will probably need to use least squares to solve this.
    Args:
        srcPoints (list): List of four (x,y) source points
        dstPoints (list): List of four (x,y) destination points
    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values
    """

    num_points = len(srcPoints)
    A = np.zeros((2*num_points, 8))
    b = np.zeros((2*num_points, 1))

    for i in range(num_points):
        x_src, y_src = srcPoints[i]
        x_dst, y_dst = dstPoints[i]

        A[2*i] = [x_src, y_src, 1, 0, 0, 0, -x_src*x_dst, -y_src*x_dst]
        A[2*i+1] = [0, 0, 0, x_src, y_src, 1, -x_src*y_dst, -y_src*y_dst]

        b[2*i] = x_dst
        b[2*i+1] = y_dst

    # Solve using least squares
    h = np.linalg.lstsq(A, b, rcond=None)[0]
    homography = np.array([[h[0][0], h[1][0], h[2][0]], [h[3][0], h[4][0], h[5][0]], [h[6][0], h[7][0], 1]])

    return homography


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename
    """

    video = cv2.VideoCapture(filename)

    while True:
        ret, frame = video.read()

        if not ret:
            break

        yield frame

    video.release()
    yield None



class Automatic_Corner_Detection(object):

    def __init__(self):

        self.SOBEL_X = np.array(
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ]).astype(np.float32)
        self.SOBEL_Y = np.array(
            [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ]).astype(np.float32)



    def gradients(self, image_bw):
        '''Use convolution with Sobel filters to calculate the image gradient at each
            pixel location
            Input -
            :param image_bw: A numpy array of shape (M,N) containing the grayscale image
            Output -
            :return Ix: Array of shape (M,N) representing partial derivatives of image
                    in x-direction
            :return Iy: Array of shape (M,N) representing partial derivative of image
                    in y-direction
        '''
        image_bw /= image_bw.max()
        Ix = cv2.filter2D(image_bw, -1, self.SOBEL_X, borderType=cv2.BORDER_CONSTANT).astype(np.float32)
        Iy = cv2.filter2D(image_bw, -1, self.SOBEL_Y, borderType=cv2.BORDER_CONSTANT).astype(np.float32)

        return Ix, Iy



    def second_moments(self, image_bw, ksize=7, sigma=10):
        """ Compute second moments from image.
            Compute image gradients, Ix and Iy at each pixel, the mixed derivatives and then the
            second moments (sx2, sxsy, sy2) at each pixel,using convolution with a Gaussian filter. You may call the
            previously written function for obtaining the gradients here.
            Input -
            :param image_bw: array of shape (M,N) containing the grayscale image
            :param ksize: size of 2d Gaussian filter
            :param sigma: standard deviation of Gaussian filter
            Output -
            :return sx2: np array of shape (M,N) containing the second moment in x direction
            :return sy2: np array of shape (M,N) containing the second moment in y direction
            :return sxsy: np array of shape (M,N) containing the second moment in the x then the
                    y direction
        """
        Ix, Iy = self.gradients(image_bw)
        size = ksize // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]

        kernel = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
        kernel /= kernel.sum()

        sx2 = cv2.filter2D(Ix*Ix, -1, kernel=kernel, borderType=cv2.BORDER_CONSTANT).astype(np.float32)
        sy2 = cv2.filter2D(Iy*Iy, -1, kernel=kernel, borderType=cv2.BORDER_CONSTANT).astype(np.float32)
        sxsy = cv2.filter2D(Ix*Iy, -1, kernel=kernel, borderType=cv2.BORDER_CONSTANT).astype(np.float32)

        return sx2, sy2, sxsy


    def harris_response_map(self, image_bw, ksize=7, sigma=5, alpha=0.05):
        """Compute the Harris cornerness score at each pixel (See Szeliski 7.1.1)
            R = det(M) - alpha * (trace(M))^2
            where M = [S_xx S_xy;
                       S_xy  S_yy],
                  S_xx = Gk * I_xx
                  S_yy = Gk * I_yy
                  S_xy  = Gk * I_xy,
            and * is a convolutional operation over a Gaussian kernel of size (k, k).
            (You can verify that this is equivalent to taking a (Gaussian) weighted sum
            over the window of size (k, k), see how convolutional operation works here:
                http://cs231n.github.io/convolutional-networks/)
            Ix, Iy are simply image derivatives in x and y directions, respectively.
            Input-
            :param image_bw: array of shape (M,N) containing the grayscale image
            :param ksize: size of 2d Gaussian filter
            :param sigma: standard deviation of gaussian filter
            :param alpha: scalar term in Harris response score
            Output-
            :return R: np array of shape (M,N), indicating the corner score of each pixel.
            """
        Sxx, Syy, Sxy = self.second_moments(image_bw, ksize, sigma)
        det_M = Sxx * Syy - Sxy ** 2
        trace_M = Sxx + Syy
        R = det_M - alpha * (trace_M ** 2)
        R_normalized = (R - np.min(R)) / (np.max(R) - np.min(R))
        print("type: ", type(R_normalized[0][0]))
        return R_normalized


    def nms_maxpool(self, R, k, ksize):
        """ Get top k interest points that are local maxima over (ksize,ksize)
        neighborhood.
        One simple way to do non-maximum suppression is to simply pick a
        local maximum over some window size (u, v). Note that this would give us all local maxima even when they
        have a really low score compare to other local maxima. It might be useful
        to threshold out low value score before doing the pooling.
        Threshold globally everything below the median to zero, and then
        MaxPool over a 7x7 kernel. This will fill every entry in the subgrids
        with the maximum nearby value. Binarize the image according to
        locations that are equal to their maximum. Multiply this binary
        image, multiplied with the cornerness response values.
        Args:
            R: np array of shape (M,N) with score response map
            k: number of interest points (take top k by confidence)
            ksize: kernel size of max-pooling operator
        Returns:
            x: np array of shape (k,) containing x-coordinates of interest points
            y: np array of shape (k,) containing y-coordinates of interest points
        """
        threshold = np.median(R)
        R[R < threshold] = 0
        kh, kw = ksize, ksize
        output = np.zeros_like(R)
        for y in range(R.shape[0]):
            for x in range(R.shape[1]):
                roi = R[max(0, y - kh // 2):min(R.shape[0], y + kh // 2 + 1),
                        max(0, x - kw // 2):min(R.shape[1], x + kw // 2 + 1)]
                output[y, x] = np.max(roi)

        binary_image = (R == output)
        R *= binary_image
        indices_flat = np.argpartition(R.flatten(), -k)[-k:]
        indices = np.unravel_index(indices_flat, R.shape)

        sorted_indices = np.argsort(R[indices])
        x = indices[1][sorted_indices]
        y = indices[0][sorted_indices]

        return x, y


    def harris_corner(self, image_bw, k=100):
        """
            Implement the Harris Corner detector. You can call harris_response_map(), nms_maxpool() functions here.
            Input-
            :param image_bw: array of shape (M,N) containing the grayscale image
            :param k: maximum number of interest points to retrieve
            Output-
            :return x: np array of shape (p,) containing x-coordinates of interest points
            :return y: np array of shape (p,) containing y-coordinates of interest points
            """
        R = self.harris_response_map(image_bw).astype(np.float32)
        x, y = self.nms_maxpool(R, k, ksize=7)
        # print(type(x[0]))

        return x, y

class Image_Mosaic(object):

    def __int__(self):
        pass

    def image_warp_inv(self, im_src, im_dst, H):
        '''
        Input -
        :param im_src: Image 1
        :param im_dst: Image 2
        :param H: numpy ndarray - 3x3 homography matrix
        Output -
        :return: Inverse Warped Resulting Image
        '''
        height_dst, width_dst = im_dst.shape[:2]
        corners_dst = np.array([[0, 0], [width_dst, 0], [0, height_dst], [width_dst, height_dst]], dtype=np.float32)
        transformed_corners = cv2.perspectiveTransform(corners_dst.reshape(1, -1, 2), H).reshape(-1, 2)
        min_x = np.min(transformed_corners[:, 0])
        max_x = np.max(transformed_corners[:, 0])
        min_y = np.min(transformed_corners[:, 1])
        max_y = np.max(transformed_corners[:, 1])
        result_width = int(np.ceil(max_x - min_x))
        result_height = int(np.ceil(max_y - min_y))
        
        print("Resultant image dimensions:", result_width, "x", result_height)
        H_inv = np.linalg.inv(H)
        
        warped_img = np.zeros([result_height, result_width, 3], dtype=np.uint8)

        for y in range(result_height):
            for x in range(result_width):
                pt_src = np.array([[x], [y], [1]])
                pt_dst = np.dot(H_inv, pt_src)
                pt_dst = pt_dst / pt_dst[2]
                x_dst, y_dst, _ = pt_dst.flatten()

                if x_dst >= 0 and y_dst >= 0 and x_dst < im_src.shape[1] - 1  and y_dst < im_src.shape[0] - 1:
                    x_dst_int, y_dst_int = int(round(x_dst)), int(round(y_dst))
                    warped_img[y, x] = im_dst[y_dst_int, x_dst_int]

        return warped_img

    def output_mosaic(self, img_src, img_warped):
        '''
        Input -
        :param img_src: Image 1
        :param img_warped: Warped Image
        Output -
        :return: Output Image Mosiac
        '''
        
        height_src, width_src, _ = img_src.shape

        i1_mask = np.zeros(img_warped.shape, np.uint8)
        i1_mask[:height_src, :width_src] = img_src
        
        im_mos_out = np.zeros(img_warped.shape, np.uint8)

        h = im_mos_out.shape[0]
        w = im_mos_out.shape[1]
        for ch in range(3):
            for x in range(0, h):
                for y in range(0, w):
                    im_mos_out[x, y, ch] = max(img_warped[x,y, ch], i1_mask[x,y, ch])
        return im_mos_out






