# ----------------
# BACKGROUND INFORMATION
# ----------------

'''Learning the parameters of a prediction function and testing it on the same data set is a methodological
mistake: a model would just repeat the labels of the samples that it has just seen would have a perfect
score but would fail to predict anything useful on yet-unseen data. This situation is called overfitting.'''

'''To avoid it, it is common practice when performing a (supervised) machine learning experiment to hold out
part of the available data as a test set (X_test, y_test).'''

# ----------------
# IMPORT PACKAGES
# ----------------

import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
from sklearn.cross_validation import KFold

# ----------------
# OBTAIN DATA
# ----------------

iris = datasets.load_iris()

# ----------------
# PROFILE DATA
# ----------------

# Sample a training set while holding out 40% of the data for testing (evaluating) our classifier.
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
	iris.data, iris.target, test_size=0.4, random_state=0)

clf = svm.SVC(kernel="linear", C=1).fit(X_train, y_train)
clf.score(X_test, y_test)

# ----------------
# COMPUTING CROSS-VALIDATED METRICS
# ----------------

# Estimate the accuracy of a linear kernel support vector machine on the iris dataset by splitting the data.
# Fit a model and compute the score 5 consectuvie times (with different splits each time)
clf = svm.SVC(kernel="linear", C=1)
scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# ----------------
# CROSS-VALIDATION ITERATORS
# ----------------

# KFold divides the samples in k groups of sampes, called folds of equal sizes.
# The prediction function is learned used k-1 folds and the fold left out is used for test.
kf = KFold(4, n_folds=2)
for train, test in kf:
	print("%s %s" % (train, test))