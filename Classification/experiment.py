"""Problem Set 6: PCA, Boosting, Haar Features, Viola-Jones."""
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import math

import ps6

# I/O directories
INPUT_DIR = "input_images"
OUTPUT_DIR = "./"

YALE_FACES_DIR = os.path.join(INPUT_DIR, 'Yalefaces')
FRUITS_DIR = os.path.join(INPUT_DIR, 'fruits')
POS_DIR = os.path.join(INPUT_DIR, "pos")
POS2_DIR = os.path.join(INPUT_DIR, "pos2")
NEG_DIR = os.path.join(INPUT_DIR, "neg")
NEG2_DIR = os.path.join(INPUT_DIR, "neg2")


def load_images_from_dir(data_dir, size=(24, 24), ext=".png"):
    imagesFiles = [f for f in os.listdir(data_dir) if f.endswith(ext)]
    imgs = [np.array(cv2.imread(os.path.join(data_dir, f), 0)) for f in imagesFiles]
    imgs = [cv2.resize(x, size) for x in imgs]

    return imgs

# Utility function
def plot_eigen_faces(eig_vecs, fig_name="", visualize=False):
    r = np.ceil(np.sqrt(len(eig_vecs)))
    c = int(np.ceil(len(eig_vecs)/r))
    r = int(r)
    fig = plt.figure()

    for i,v in enumerate(eig_vecs):
        sp = fig.add_subplot(r,c,i+1)

        plt.imshow(v.reshape(32,32).real, cmap='gray')
        sp.set_title('eigenface_%i'%i)
        sp.axis('off')

    fig.subplots_adjust(hspace=.5)

    if visualize:
        plt.show()

    if not fig_name == "":
        plt.savefig("{}".format(fig_name))


# Functions you need to complete
def visualize_mean_face(x_mean, size, new_dims):
    """Rearrange the data in the mean face to a 2D array

    - Organize the contents in the mean face vector to a 2D array.
    - Normalize this image.
    - Resize it to match the new dimensions parameter

    Args:
        x_mean (numpy.array): Mean face values.
        size (tuple): x_mean 2D dimensions
        new_dims (tuple): Output array dimensions

    Returns:
        numpy.array: Mean face uint8 2D array.
    """
    mean_face_2d = x_mean.reshape(size)
    mean_face_normalized = cv2.normalize(mean_face_2d, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    resized_mean_face = cv2.resize(mean_face_normalized, new_dims, interpolation=cv2.INTER_LINEAR)
    return resized_mean_face


def part_1a_1b():

    orig_size = (192, 231)
    small_size = (32, 32)
    X, y = ps6.load_images(YALE_FACES_DIR, small_size)

    # Get the mean face
    x_mean = ps6.get_mean_face(X)

    x_mean_image = visualize_mean_face(x_mean, small_size, orig_size)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "ps6-1-a-1.png"), x_mean_image)

    # PCA dimension reduction
    k = 10
    eig_vecs, eig_vals = ps6.pca(X, k)

    plot_eigen_faces(eig_vecs.T, "ps6-1-b-1.png")


def part_1c():
    p = 0.8  # Select a split percentage value
    k = 50  # Select a value for k
    
    for k_ in range(2, k, 2):
        size = (32, 32)
        X, y = ps6.load_images(YALE_FACES_DIR, size)
        Xtrain, ytrain, Xtest, ytest = ps6.split_dataset(X, y, p)

        # training
        mu = ps6.get_mean_face(Xtrain)
        eig_vecs, eig_vals = ps6.pca(Xtrain, k_)
        Xtrain_proj = np.dot(Xtrain - mu, eig_vecs)

        # testing
        mu = ps6.get_mean_face(Xtest)
        Xtest_proj = np.dot(Xtest - mu, eig_vecs)

        good = 0
        bad = 0

        for i, obs in enumerate(Xtest_proj):

            dist = [np.linalg.norm(obs - x) for x in Xtrain_proj]

            idx = np.argmin(dist)
            y_pred = ytrain[idx]

            if y_pred == ytest[i]:
                good += 1

            else:
                bad += 1
        print("(PCA) using k = ", k_)
        print('(PCA) Good predictions = ', good, 'Bad predictions = ', bad)
        print('(PCA) {0:.2f}% accuracy'.format(100 * float(good) / (good + bad)))


def part_2a():

    for num_iter in [5, 10, 15, 20, 25]:
        y0 = 1
        y1 = 2

        X, y = ps6.load_images(FRUITS_DIR)

        # Select only the y0 and y1 classes
        idx = y == y0
        idx |= y == y1

        X = X[idx,:]
        y = y[idx]

        # Label them 1 and -1
        y0_ids = y == y0
        y1_ids = y == y1
        y[y0_ids] = 1
        y[y1_ids] = -1

        p = 0.8
        Xtrain, ytrain, Xtest, ytest = ps6.split_dataset(X, y, p)

        # Picking random numbers
        rand_y = np.random.choice([-1, 1], (len(ytrain)))
        # TODO: find which of these labels match ytrain and report its accuracy
        correct_rand = 0
        for i in range(len(ytrain)):
            if rand_y[i] == ytrain[i]:
                correct_rand += 1
        rand_accuracy = 100 * float(correct_rand) / len(ytrain)
        print('(Random) Training accuracy: {0:.2f}%'.format(rand_accuracy))

        # Using Weak Classifier
        uniform_weights = np.ones((Xtrain.shape[0],)) / Xtrain.shape[0]
        wk_clf = ps6.WeakClassifier(Xtrain, ytrain, uniform_weights)
        wk_clf.train()
        wk_results = [wk_clf.predict(x) for x in Xtrain]

        # TODO: find which of these labels match ytrain and report its accuracy
        correct_weak = 0
        for i in range(len(ytrain)):
            if wk_results[i] == ytrain[i]:
                correct_weak += 1
        wk_accuracy = 100 * float(correct_weak) / len(ytrain)

        print('(Weak) Training accuracy {0:.2f}%'.format(wk_accuracy))

        

        boost = ps6.Boosting(Xtrain, ytrain, num_iter)
        boost.train()
        good, bad = boost.evaluate()
        boost_accuracy = 100 * float(good) / (good + bad)
        print('(Boosting) Training accuracy {0:.2f}%, using num_iter: {1}'.format(boost_accuracy, num_iter))

        # Picking random numbers
        rand_y = np.random.choice([-1, 1], (len(ytest)))
        # TODO: find which of these labels match ytest and report its accuracy
        correct_rand = 0
        for i in range(len(ytest)):
            if rand_y[i] == ytest[i]:
                correct_rand += 1
        rand_accuracy = 100 * float(correct_rand) / len(ytest)
        print('(Random) Testing accuracy: {0:.2f}%'.format(rand_accuracy))

        # Using Weak Classifier
        wk_results = [wk_clf.predict(x) for x in Xtest]
        # TODO: find which of these labels match ytest and report its accuracy
        correct_weak = 0
        for i in range(len(ytest)):
            if rand_y[i] == ytest[i]:
                correct_weak += 1
        wk_accuracy = 100 * float(correct_weak) / len(ytest)
        print('(Weak) Testing accuracy {0:.2f}%'.format(wk_accuracy))

        y_pred = boost.predict(Xtest)
        # TODO: find which of these labels match ytest and report its accuracy
        correct_boost = 0
        for i in range(len(ytest)):
            if y_pred[i] == ytest[i]:
                correct_boost += 1
        boost_accuracy = 100 * float(correct_boost) / len(ytest)
        
        print('(Boosting) Testing accuracy {0:.2f}%, using num_iter: {1}'.format(boost_accuracy, num_iter))


def part_3a():
    """Complete the remaining parts of this section as instructed in the
    instructions document."""

    feature1 = ps6.HaarFeature((2, 1), (25, 30), (50, 100))
    feature1.preview((200, 200), filename="ps6-3-a-1.png")

    feature2 = ps6.HaarFeature((1, 2), (10, 25), (50, 150))
    feature2.preview((200, 200), filename="ps6-3-a-2.png")
    
    feature3 = ps6.HaarFeature((3, 1), (50, 50), (100, 50))
    feature3.preview((200, 200), filename="ps6-3-a-3.png")
    
    feature4 = ps6.HaarFeature((1, 3), (50, 125), (100, 50))
    feature4.preview((200, 200), filename="ps6-3-a-4.png")
    
    feature5 = ps6.HaarFeature((2, 2), (50, 25), (100, 150))
    feature5.preview((200, 200), filename="ps6-3-a-5.png")


def part_4_a_b():

    pos = load_images_from_dir(POS_DIR)
    neg = load_images_from_dir(NEG_DIR)

    train_pos = pos
    train_neg = neg
    images = train_pos + train_neg
    labels = np.array(len(train_pos) * [1] + len(train_neg) * [-1])

    integral_images = ps6.convert_images_to_integral_images(images)
    VJ = ps6.ViolaJones(train_pos, train_neg, integral_images)
    VJ.createHaarFeatures()

    VJ.train(5)

    VJ.haarFeatures[VJ.classifiers[0].feature].preview(filename="ps6-4-b-1")
    VJ.haarFeatures[VJ.classifiers[1].feature].preview(filename="ps6-4-b-2")

    predictions = VJ.predict(images)
    vj_accuracy = (predictions == labels).mean() * 100
    print("Prediction accuracy on training: {0:.2f}%".format(vj_accuracy))

    pos2 = load_images_from_dir(POS2_DIR)
    neg2 = load_images_from_dir(NEG2_DIR)

    test_pos = pos2
    test_neg = neg2
    test_images = test_pos + test_neg
    real_labels = np.array(len(test_pos) * [1] + len(test_neg) * [-1])
    predictions = VJ.predict(test_images)

    vj_accuracy = (predictions == real_labels).mean() * 100
    print("Prediction accuracy on testing: {0:.2f}%".format(vj_accuracy))


def part_4_c():
    pos = load_images_from_dir(POS_DIR)[:20]
    neg = load_images_from_dir(NEG_DIR)

    images = pos + neg

    integral_images = ps6.convert_images_to_integral_images(images)
    VJ = ps6.ViolaJones(pos, neg, integral_images)
    VJ.createHaarFeatures()

    VJ.train(4)

    image = cv2.imread(os.path.join(INPUT_DIR, "man.jpeg"), -1)
    image = cv2.resize(image, (120, 60))
    VJ.faceDetection(image, filename="ps4-4-c-1.png")

def part_5_b():
    pos2 = load_images_from_dir(POS2_DIR)
    neg2 = load_images_from_dir(NEG2_DIR)

    CC = ps6.CascadeClassifier(pos2, neg2)
    CC.train()

    # - you can test and debug your algorithm with the man.jpeg image but for
    # the final submission you need to choose another image to test on - ie
    # change the "man.jpeg" in the following line:
    image = cv2.imread(os.path.join(INPUT_DIR, "man.jpeg"), -1)
    image = cv2.resize(image, (120, 60))
    CC.faceDetection(image, filename="cascade")


if __name__ == "__main__":
    # part_1a_1b()
    # part_1c()
    # part_2a()
    # part_3a()
    part_4_a_b()
    # part_4_c()
    # part_5_b()
