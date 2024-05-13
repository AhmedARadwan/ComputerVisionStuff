# according to the instructions PDF, we can use sklearn.tree.DecisionTreeClassifier to train a classifier
# kindly uncomment the code below to train the classifier and save it to a file
# from sklearn.tree import DecisionTreeClassifier
# import numpy as np
# import pickle

# X = np.array([
#     [128.9657223672473, 115.6948384318657],  # person08_running_d4_uncomp
#     [129.33535741415702, 111.40481579674928], # person01_running_d2_uncomp
#     [128.88860103379952, 104.60221745633692], # person04_walking_d4_uncomp
#     [127.66012551707459, 120.52364957380148], # person06_walking_d1_uncomp
#     [119.2897415301331, 120.22118009845369],  # person10_handclapping_d4_uncomp
#     [109.18598964944525, 83.61364406281133]    # person14_handclapping_d3_uncomp
# ])

# y = np.array(['running', 'running', 'walking', 'walking', 'clapping', 'clapping'])

# clf = DecisionTreeClassifier()
# clf.fit(X, y)

# quick test
# x = clf.predict([[128.88860103379952, 104.60221745633692]])[0]
# print("pred: ", x)

# with open('decision_tree_classifier.pkl', 'wb') as f:
#     pickle.dump(clf, f)