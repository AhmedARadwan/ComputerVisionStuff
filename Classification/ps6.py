"""Problem Set 6: PCA, Boosting, Haar Features, Viola-Jones."""
import numpy as np
import cv2
import os

from helper_classes import WeakClassifier, VJ_Classifier


# assignment code
def load_images(folder, size=(32, 32)):
    """Load images to workspace.

    Args:
        folder (String): path to folder with images.
        size   (tuple): new image sizes

    Returns:
        tuple: two-element tuple containing:
            X (numpy.array): data matrix of flatten images
                             (row:observations, col:features) (float).
            y (numpy.array): 1D array of labels (int).
    """

    X = []
    y = []

    images_files = [f for f in os.listdir(folder)]
    for image_file in images_files:
        image = cv2.imread(os.path.join(folder, image_file), 0)
        image = cv2.resize(image, size)
        X.append(image.flatten())
        y.append(int(image_file.split(".")[0].split("subject")[1]))

    return (np.array(X), np.array(y))



def split_dataset(X, y, p):
    """Split dataset into training and test sets.

    Let M be the number of images in X, select N random images that will
    compose the training data (see np.random.permutation). The images that
    were not selected (M - N) will be part of the test data. Record the labels
    accordingly.

    Args:
        X (numpy.array): 2D dataset.
        y (numpy.array): 1D array of labels (int).
        p (float): Decimal value that determines the percentage of the data
                   that will be the training data.

    Returns:
        tuple: Four-element tuple containing:
            Xtrain (numpy.array): Training data 2D array.
            ytrain (numpy.array): Training data labels.
            Xtest (numpy.array): Test data test 2D array.
            ytest (numpy.array): Test data labels.
    """

    num_train_samples = int(len(X) * p)
    indices = np.random.permutation(len(X))

    train_indices = indices[:num_train_samples]
    test_indices = indices[num_train_samples:]

    Xtrain, Xtest = X[train_indices], X[test_indices]
    ytrain, ytest = y[train_indices], y[test_indices]

    return (Xtrain, ytrain, Xtest, ytest)


def get_mean_face(x):
    """Return the mean face.

    Calculate the mean for each column.

    Args:
        x (numpy.array): array of flattened images.

    Returns:
        numpy.array: Mean face.
    """

    return np.mean(x, axis=0)


def pca(X, k):
    """PCA Reduction method.

    Return the top k eigenvectors and eigenvalues using the covariance array
    obtained from X.


    Args:
        X (numpy.array): 2D data array of flatten images (row:observations,
                         col:features) (float).
        k (int): new dimension space

    Returns:
        tuple: two-element tuple containing
            eigenvectors (numpy.array): 2D array with the top k eigenvectors.
            eigenvalues (numpy.array): array with the top k eigenvalues.
    """
    X_mean = np.mean(X, axis=0)
    delta = X - X_mean
    sigma = np.dot(delta.T, delta)
    eigenvalues, eigenvectors = np.linalg.eigh(sigma)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    eigenvalues = eigenvalues[:k]
    eigenvectors = eigenvectors[:, :k]
    return eigenvectors, eigenvalues


class Boosting:
    """Boosting classifier.

    Args:
        X (numpy.array): Data array of flattened images
                         (row:observations, col:features) (float).
        y (numpy.array): Labels array of shape (observations, ).
        num_iterations (int): number of iterations
                              (ie number of weak classifiers).

    Attributes:
        Xtrain (numpy.array): Array of flattened images (float32).
        ytrain (numpy.array): Labels array (float32).
        num_iterations (int): Number of iterations for the boosting loop.
        weakClassifiers (list): List of weak classifiers appended in each
                               iteration.
        alphas (list): List of alpha values, one for each classifier.
        num_obs (int): Number of observations.
        weights (numpy.array): Array of normalized weights, one for each
                               observation.
        eps (float): Error threshold value to indicate whether to update
                     the current weights or stop training.
    """

    def __init__(self, X, y, num_iterations):
        self.Xtrain = np.float32(X)
        self.ytrain = np.float32(y)
        self.num_iterations = num_iterations
        self.weakClassifiers = []
        self.alphas = []
        self.num_obs = X.shape[0]
        self.weights = np.array([1.0 / self.num_obs] * self.num_obs)  # uniform weights
        self.eps = 0.0001

    def train(self):
        """Implement the for loop shown in the problem set instructions."""
        for _ in range(self.num_iterations):
            weak_classifier = WeakClassifier(self.Xtrain, self.ytrain, self.weights)
            weak_classifier.train()
            
            error = 0
            for i, (sample_features, true_label) in enumerate(zip(self.Xtrain, self.ytrain)):
                predicted_label = weak_classifier.predict(sample_features)
                if predicted_label != true_label:
                    error += self.weights[i]
            
            alpha = 0.5 * np.log((1 - error) / max(error, 1e-10))  # Avoid division by zero
            self.weakClassifiers.append(weak_classifier)
            self.alphas.append(alpha)

            predictions = []
            for x in self.Xtrain:
                prediction = weak_classifier.predict(x)
                predictions.append(prediction)

            exponential_values = []
            for i in range(len(predictions)):
                exponential_value = np.exp(-alpha * self.ytrain[i] * predictions[i])
                exponential_values.append(exponential_value)

            self.weights *= np.array(exponential_values)
            self.weights /= np.sum(self.weights)

    def evaluate(self):
        """Return the number of correct and incorrect predictions.

        Use the training data (self.Xtrain) to obtain predictions. Compare
        them with the training labels (self.ytrain) and return how many
        where correct and incorrect.

        Returns:
            tuple: two-element tuple containing:
                correct (int): Number of correct predictions.
                incorrect (int): Number of incorrect predictions.
        """
        correct = 0
        for i in range(self.num_obs):
            if self.predict(self.Xtrain[i]) == self.ytrain[i]:
                correct += 1

        incorrect = self.num_obs - correct
        return correct, incorrect

    def predict(self, X):
        """Return predictions for a given array of observations.

        Use the alpha values stored in self.aphas and the weak classifiers
        stored in self.weakClassifiers.

        Args:
            X (numpy.array): Array of flattened images (observations).

        Returns:
            numpy.array: Predictions, one for each row in X.
        """
        y_pred = []
        for weak_classifier in self.weakClassifiers:
            y_pred.append([weak_classifier.predict(np.transpose(X))])
        
        for i in range(0, len(self.alphas)):
            y_pred[i] = np.array(y_pred[i]) * self.alphas[i]
        
        y_pred = np.sum(y_pred, axis=0)
        y_pred = y_pred[0]
        return np.sign(y_pred)


class HaarFeature:
    """Haar-like features.

    Args:
        feat_type (tuple): Feature type {(2, 1), (1, 2), (3, 1), (2, 2)}.
        position (tuple): (row, col) position of the feature's top left corner.
        size (tuple): Feature's (height, width)

    Attributes:
        feat_type (tuple): Feature type.
        position (tuple): Feature's top left corner.
        size (tuple): Feature's width and height.
    """

    def __init__(self, feat_type, position, size):
        self.feat_type = feat_type
        self.position = position
        self.size = size

    def _create_two_horizontal_feature(self, shape):
        """Create a feature of type (2, 1).

        Use int division to obtain half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).
            # gray_area_value = ii[row+height-1, col+width-1] \
            #                 + ii[row-1, col+(width//2)-1] \
            #                 - ii[row-1, col+width-1] \
            #                 - ii[row+height-1, col+(width//2)-1]

            # print("score: ", white_area_value - gray_area_value)
        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        img = np.zeros(shape, dtype=np.uint8)
        row, col = self.position
        height, width = self.size
        half_height = height // 2
        img[row:row+half_height, col:col+width] = 255
        img[row+half_height:row+height, col:col+width] = 126
        return img

    def _create_two_vertical_feature(self, shape):
        """Create a feature of type (1, 2).

        Use int division to obtain half the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        img = np.zeros(shape, dtype=np.uint8)
        row, col = self.position
        height, width = self.size
        half_width = width // 2
        img[row:row+height, col:col+half_width] = 255
        img[row:row+height, col+half_width:col+width] = 126
        return img

    def _create_three_horizontal_feature(self, shape):
        """Create a feature of type (3, 1).

        Use int division to obtain a third of the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        img = np.zeros(shape, dtype=np.uint8)
        row, col = self.position
        height, width = self.size
        third_height = height // 3
        img[row:row+third_height, col:col+width] = 255
        img[row+third_height:row+2*third_height, col:col+width] = 126
        img[row+2*third_height:row+height, col:col+width] = 255
        return img

    def _create_three_vertical_feature(self, shape):
        """Create a feature of type (1, 3).

        Use int division to obtain a third of the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        img = np.zeros(shape, dtype=np.uint8)
        row, col = self.position
        height, width = self.size
        third_width = width // 3
        img[row:row+height, col:col+third_width] = 255
        img[row:row+height, col+third_width:col+2*third_width] = 126
        img[row:row+height, col+2*third_width:col+width] = 255
        return img

    def _create_four_square_feature(self, shape):
        """Create a feature of type (2, 2).

        Use int division to obtain half the width and half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        img = np.zeros(shape, dtype=np.uint8)
        row, col = self.position
        height, width = self.size
        half_height = height // 2
        half_width = width // 2
        img[row:row+half_height, col:col+half_width] = 126
        img[row:row+half_height, col+half_width:col+width] = 255
        img[row+half_height:row+height, col:col+half_width] = 255
        img[row+half_height:row+height, col+half_width:col+width] = 126
        return img

    def preview(self, shape=(24, 24), filename=None):
        """Return an image with a Haar-like feature of a given type.

        Function that calls feature drawing methods. Each method should
        create an 2D zeros array. Each feature is made of a white area (255)
        and a gray area (126).

        The drawing methods use the class attributes position and size.
        Keep in mind these are in (row, col) and (height, width) format.

        Args:
            shape (tuple): Array numpy-style shape (rows, cols).
                           Defaults to (24, 24).

        Returns:
            numpy.array: Array containing a Haar feature (float or uint8).
        """

        if self.feat_type == (2, 1):  # two_horizontal
            X = self._create_two_horizontal_feature(shape)

        if self.feat_type == (1, 2):  # two_vertical
            X = self._create_two_vertical_feature(shape)

        if self.feat_type == (3, 1):  # three_horizontal
            X = self._create_three_horizontal_feature(shape)

        if self.feat_type == (1, 3):  # three_vertical
            X = self._create_three_vertical_feature(shape)

        if self.feat_type == (2, 2):  # four_square
            X = self._create_four_square_feature(shape)

        if filename is None:
            cv2.imwrite("{}_feature.png".format(self.feat_type), X)

        else:
            cv2.imwrite("{}.png".format(filename), X)

        return X

    def evaluate(self, ii):
        """Evaluates a feature's score on a given integral image.

        Calculate the score of a feature defined by the self.feat_type.
        Using the integral image and the sum / subtraction of rectangles to
        obtain a feature's value. Add the feature's white area value and
        subtract the gray area.

        For example, on a feature of type (2, 1):
        score = sum of pixels in the white area - sum of pixels in the gray area

        Keep in mind you will need to use the rectangle sum / subtraction
        method and not numpy.sum(). This will make this process faster and
        will be useful in the ViolaJones algorithm.

        Args:
            ii (numpy.array): Integral Image.

        Returns:
            float: Score value.
        """
        
        row, col = self.position
        height, width = self.size
        white_area_value = 0
        gray_area_value = 0

        if self.feat_type == (2, 1):
            ii = ii.astype(np.float64)
            white_area_value = ii[row+(height//2)-1, col+width-1] \
                             + ii[row-1, col-1] \
                             - ii[row-1, col+width-1] \
                             - ii[row+(height//2)-1, col-1] 

            gray_area_value = ii[row+height-1, col+width-1] \
                            + ii[row+(height//2)-1, col-1] \
                            - ii[row+(height//2)-1, col+width-1] \
                            - ii[row+height-1, col-1]

        elif self.feat_type == (1, 2):
            ii = ii.astype(np.float64)
            white_area_value = ii[row+height-1, col+(width//2)-1] \
                             + ii[row-1, col-1] \
                             - ii[row-1, col+(width//2)-1] \
                             - ii[row+height-1, col-1]
            
            gray_area_value = ii[row+height-1, col+width-1] \
                            + ii[row-1, col+(width//2)-1] \
                            - ii[row-1, col+width-1] \
                            - ii[row+height-1, col+(width//2)-1]

        elif self.feat_type == (3, 1):
            ii = ii.astype(np.float64)
            white_area_value = ii[row+(height//3)-1, col+width-1] \
                             + ii[row-1, col-1] \
                             - ii[row-1, col+width-1] \
                             - ii[row+(height//3)-1, col-1]
            
            gray_area_value = ii[row+2*(height//3)-1, col+width-1] \
                            + ii[row+(height//3)-1, col-1] \
                            - ii[row+(height//3)-1, col+width-1] \
                            - ii[row+2*(height//3)-1, col-1]
            
            white_area_value += ii[row+height-1, col+width-1] \
                              + ii[row+2*(height//3)-1, col-1] \
                              - ii[row+2*(height//3)-1, col+width-1] \
                              - ii[row+height-1, col-1]

        elif self.feat_type == (1, 3):
            ii = ii.astype(np.float64)
            white_area_value = ii[row+height-1, col+(width//3)-1] \
                             + ii[row-1, col-1] \
                             - ii[row-1, col+(width//3)-1] \
                             - ii[row+height-1, col-1]
            
            gray_area_value = ii[row+height-1, col+2*(width//3)-1] \
                            + ii[row-1, col+(width//3)-1] \
                            - ii[row-1, col+2*(width//3)-1] \
                            - ii[row+height-1, col+(width//3)-1]
            
            white_area_value += ii[row+height-1, col+width-1] \
                              + ii[row-1, col+2*(width//3)-1] \
                              - ii[row-1, col+width-1] \
                              - ii[row+height-1, col+2*(width//3)-1]

        elif self.feat_type == (2, 2):
            ii = ii.astype(np.float64)
            gray_area_value = ii[row+(height//2)-1, col+(width//2)-1] \
                            + ii[row-1, col-1] \
                            - ii[row-1, col+(width//2)-1] \
                            - ii[row+(height//2)-1, col-1]

            white_area_value = ii[row+(height//2)-1, col+width-1] \
                             + ii[row-1, col+(width//2)-1] \
                             - ii[row-1, col+width-1] \
                             - ii[row+(height//2)-1, col+(width//2)-1]
            
            white_area_value += ii[row+height-1, col+(width//2)-1] \
                              + ii[row+(height//2)-1, col-1] \
                              - ii[row+(height//2)-1, col+(width//2)-1] \
                              - ii[row+height-1, col-1]

            gray_area_value += ii[row+height-1, col+width-1] \
                             + ii[row+(height//2)-1, col+(width//2)-1] \
                             - ii[row+(height//2)-1, col+width-1] \
                             - ii[row+height-1, col+(width//2)-1]

        return white_area_value - gray_area_value


def convert_images_to_integral_images(images):
    """Convert a list of grayscale images to integral images.

    Args:
        images (list): List of grayscale images (uint8 or float).

    Returns:
        (list): List of integral images.
    """

    integral_images = []
    for img in images:
        integral_img = np.cumsum(np.cumsum(img, axis=0), axis=1)
        integral_images.append(integral_img)
    return integral_images

#############################
class ViolaJones:
    """Viola Jones face detection method

    Args:
        pos (list): List of positive images.
        neg (list): List of negative images.
        integral_images (list): List of integral images.

    Attributes:
        haarFeatures (list): List of haarFeature objects.
        integralImages (list): List of integral images.
        classifiers (list): List of weak classifiers (VJ_Classifier).
        alphas (list): Alpha values, one for each weak classifier.
        posImages (list): List of positive images.
        negImages (list): List of negative images.
        labels (numpy.array): Positive and negative labels.
    """
    def __init__(self, pos, neg, integral_images):
        self.haarFeatures = []
        self.integralImages = integral_images
        self.classifiers = []
        self.alphas = []
        self.posImages = pos
        self.negImages = neg
        self.labels = np.hstack((np.ones(len(pos)), -1*np.ones(len(neg))))
        self.threshold = 1.0

    def createHaarFeatures(self):
        # Let's take detector resolution of 24x24 like in the paper
        FeatureTypes = {"two_horizontal": (2, 1),
                        "two_vertical": (1, 2),
                        "three_horizontal": (3, 1),
                        "three_vertical": (1, 3),
                        "four_square": (2, 2)}

        haarFeatures = []
        for _, feat_type in FeatureTypes.items():
            for sizei in range(feat_type[0], 24 + 1, feat_type[0]):
                for sizej in range(feat_type[1], 24 + 1, feat_type[1]):
                    for posi in range(0, 24 - sizei + 1, 4):
                        for posj in range(0, 24 - sizej + 1, 4):
                            haarFeatures.append(
                                HaarFeature(feat_type, [posi, posj],
                                            [sizei, sizej]))
        self.haarFeatures = haarFeatures

    def set_threshold(self, threshold):
        self.threshold = threshold

    def init_train(self):
        """ This function initializes self.scores, self.weights

        Args:
            None

        Returns:
            None
        """
    
        # Use this scores array to train a weak classifier using VJ_Classifier
        # in the for loop below.
        if not self.integralImages or not self.haarFeatures:
            print("No images provided. run convertImagesToIntegralImages() first")
            print("       Or no features provided. run creatHaarFeatures() first")
            return

        self.scores = np.zeros((len(self.integralImages), len(self.haarFeatures)))
        print(" -- compute all scores --")
        for i, im in enumerate(self.integralImages):
            self.scores[i, :] = [hf.evaluate(im) for hf in self.haarFeatures]

        weights_pos = np.ones(len(self.posImages), dtype='float') * 1.0 / (
                           2*len(self.posImages))
        weights_neg = np.ones(len(self.negImages), dtype='float') * 1.0 / (
                           2*len(self.negImages))
        self.weights = np.hstack((weights_pos, weights_neg))

    def train(self, num_classifiers):
        """ Initialize and train Viola Jones face detector

        The function should modify self.weights, self.classifiers, self.alphas, and self.threshold

        Args:
            None

        Returns:
            None
        """
        self.init_train()
        print(" -- select classifiers --")
        for i in range(num_classifiers):

            self.weights = self.weights/sum(self.weights)
            self.scores = [hf.evaluate(ii) for ii in self.integralImages for hf in self.haarFeatures]
            n_feats = len(self.haarFeatures)
            n_iis = len(self.integralImages)
            X_ = np.reshape(self.scores, (n_iis, n_feats))
            y_ = self.labels
            vj = VJ_Classifier(X_, y_, self.weights)
            vj.train()
            preds = vj.predict(X_.T)
            eps = 1*(preds != self.labels)
            beta = vj.error/(1-vj.error)
            print(vj.error)
            self.weights = self.weights*(beta**(1-eps))
            alpha = np.log(1/beta)
            
            self.classifiers.append(vj)
            self.alphas.append(alpha)

    def predict(self, images):
        """Return predictions for a given list of images.

        Args:
            images (list of element of type numpy.array): list of images (observations).

        Returns:
            list: Predictions, one for each element in images.
        """

        ii = convert_images_to_integral_images(images)

        scores = np.zeros((len(ii), len(self.haarFeatures)))

        for i, img in enumerate(ii):
            for j, hf in enumerate(self.haarFeatures):
                scores[i, j] = hf.evaluate(img)

        predictions = np.zeros(len(images))

        for i, img_scores in enumerate(scores):
            score = sum(alpha * clf.predict(img_scores) for alpha, clf in zip(self.alphas, self.classifiers))
            predictions[i] = 1 if score >= self.threshold else -1

        return predictions

    def faceDetection(self, image, filename):
        """Scans for faces in a given image.

        Complete this function following the instructions in the problem set
        document.

        Use this function to also save the output image.

        Args:
            image (numpy.array): Input image.
            filename (str): Output image file name.

        Returns:
            None.
        """

        out = image.copy()
        image =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape
        face = [self.predict([image[i:i+24, j:j+24]]) for i in range(h-24) for j in range(w-24)]
        face = np.reshape(face, (h-24, w-24))
        ind_face = np.where(face==1)
        y, x = np.mean(ind_face, axis = 1).astype(int)
        
        out = cv2.rectangle(out, (x, y), (x+24, y+24), color=(0, 255, 0), thickness=1)
                
        cv2.imwrite(filename, out)


class CascadeClassifier:
    """Viola Jones Cascade Classifier Face Detection Method

    Lesson: 8C-L2, Boosting and face detection

    Args:
        f_max (float): maximum acceptable false positive rate per layer
        d_min (float): minimum acceptable detection rate per layer
        f_target (float): overall target false positive rate
        pos (list): List of positive images.
        neg (list): List of negative images.

    Attributes:
        f_target: overall false positive rate
        classifiers (list): Adaboost classifiers
        train_pos (list of numpy arrays):  
        train_neg (list of numpy arrays): 

    """
    def __init__(self, pos, neg, f_max_rate=0.30, d_min_rate=0.70, f_target = 0.07):
        
        train_percentage = 0.85

        pos_indices = np.random.permutation(len(pos)).tolist()
        neg_indices = np.random.permutation(len(neg)).tolist()

        train_pos_num = int(train_percentage * len(pos))
        train_neg_num = int(train_percentage * len(neg))

        pos_train_indices = pos_indices[:train_pos_num]
        pos_validate_indices = pos_indices[train_pos_num:]

        neg_train_indices = neg_indices[:train_neg_num]
        neg_validate_indices = neg_indices[train_neg_num:]

        self.train_pos = [pos[i] for i in pos_train_indices]
        self.train_neg = [neg[i] for i in neg_train_indices]

        self.validate_pos = [pos[i] for i in pos_validate_indices]
        self.validate_neg = [neg[i] for i in neg_validate_indices]

        self.f_max_rate = f_max_rate
        self.d_min_rate = d_min_rate
        self.f_target = f_target
        self.classifiers = []

    def predict(self, classifiers, img):
        """Predict face in a single image given a list of cascaded classifiers

        Args:
            classifiers (list of element type ViolaJones): list of ViolaJones classifiers to predict 
                where index i is the i'th consecutive ViolaJones classifier
            img (numpy.array): Input image

        Returns:
            Return 1 (face detected) or -1 (no face detected) 
        """

        # TODO
        raise NotImplementedError

    def evaluate_classifiers(self, pos, neg, classifiers):
        """ 
        Given a set of classifiers and positive and negative set
        return false positive rate and detection rate 

        Args:
            pos (list): Input image.
            neg (list): Output image file name.
            classifiers (list):  

        Returns:
            f (float): false positive rate
            d (float): detection rate
            false_positives (list): list of false positive images
        """

        # TODO
        raise NotImplementedError

    def train(self):
        """ 
        Trains a cascaded face detector

        Sets self.classifiers (list): List of ViolaJones classifiers where index i is the i'th consecutive ViolaJones classifier

        Args:
            None

        Returns:
            None
             
        """
        # TODO
        raise NotImplementedError


    def faceDetection(self, image, filename="ps6-5-b-1.jpg"):
        """Scans for faces in a given image using the Cascaded Classifier.

        Complete this function following the instructions in the problem set
        document.

        Use this function to also save the output image.

        Args:
            image (numpy.array): Input image.
            filename (str): Output image file name.

        Returns:
            None.
        """
        raise NotImplementedError
