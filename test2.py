import scipy.io
import numpy as np
import matplotlib
#import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# this may be redundant
matplotlib.rcParams['backend'] = "Qt4Agg"

# load the data
X = scipy.io.loadmat('./matlabFiles/data28.mat')['matImages']
y = scipy.io.loadmat('./matlabFiles/labels28.mat')['matLabels'].ravel() - 1
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# print the shapes
print("size of X: " + str(X.shape))
print("size of y: " + str(y.shape))

#image processing
img = X[0, :]
img = np.reshape(img, (16, 16)).T
print(img)
plt.imshow(img, cmap='gray')
#plt.savefig('test.png')
