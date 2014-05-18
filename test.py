__author__ = 'lukas'
__author__ = 'lukas'
# Code source: Gael Varoqueux
#              Andreas Mueller
# Modified for Documentation merge by Jaques Grobler
# License: BSD 3 clause
import sys
import numpy as np
import pylab as pl
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
import vigra
from random_forest_training_data_prep import random_forest_training_data_prep
from muell import connectedCompFeat


threshHold = 160
firstBinClosing = 6
secondDilatation = 3
thirdErosion = 2
max_size = 5000


h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest1", "Random Forest2", "Random Forest3", "Random Forest4",
         "Random Forestbes1","Random Forest6", "Random Forest7",
         "AdaBoost", "Naive Bayes", "LDA", "QDA"]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=1),
    RandomForestClassifier(max_depth=2, n_estimators=1, max_features=1),
    RandomForestClassifier(max_depth=2, n_estimators=1, max_features=1),
    RandomForestClassifier(max_depth=2, n_estimators=1, max_features=1),
    RandomForestClassifier(max_depth=2, n_estimators=1, max_features=1),
    RandomForestClassifier(max_depth=2, n_estimators=1, max_features=1),
    RandomForestClassifier(max_depth=2, n_estimators=1, max_features=1),
    RandomForestClassifier(max_depth=2, n_estimators=1, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LDA(),
    QDA()]


classifiers[3] = RandomForestClassifier(max_depth=5, n_estimators=1, max_features=1)
print "RF: ", classifiers[3]
print "len cf: ", len(classifiers)
print "len nm: ", len(names)



# get two labels grey value and disc erosion
groundtruth = vigra.impex.readHDF5('/home/lukas/data/groundtruthRFMAG1.h5', 'groundtruthMAG1L')
rawUint8 = vigra.impex.readHDF5('/home/lukas/data/RAWtrainingMAG1.h5', 'raw')



discEr = connectedCompFeat(firstBinClosing, secondDilatation, threshHold, max_size, True, rawUint8)
# discEr = vigra.filters.discErosion(rawUint8, 1)
discErRF = np.ravel(discEr)
discErRF = discErRF.reshape(discErRF.shape[0], 1)
rawUint8RF = np.ravel(rawUint8)
rawUint8RF = rawUint8RF.reshape(rawUint8RF.shape[0] ,1)




print "disc ErosionRF: ", discErRF.shape

print "rawUint8 RF: ", rawUint8RF.shape

trainingData = np.concatenate((rawUint8RF.astype(np.float32), discErRF.astype(np.float32)), 1)

print "trainingData: ", trainingData.shape
print "groundtruth: ", groundtruth.shape

X, y = random_forest_training_data_prep(trainingData, groundtruth, 10000)
print "done shuffeling..."
# X[:,[0,1]]=X[:,[1,0]]

print X
y = y.reshape(y.shape[0])

y = y.astype(np.int64)

print "Old X, y: ", X.shape, "  ", y.shape
print X.dtype, y.dtype


linearly_separable = (X, y)

datasets = [
            linearly_separable
            ]

figure = pl.figure(figsize=(40, 9))
i = 1
# iterate over datasets
for ds in datasets:
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = pl.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = pl.subplot(len(datasets), len(classifiers) + 1, i)
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = pl.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

figure.subplots_adjust(left=.02, right=.98)
pl.show()