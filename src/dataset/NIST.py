import scipy.io
import numpy as np
import Augmentor

X_full = scipy.io.loadmat('./matlabFiles/data28.mat')['matImages'][0]
X_full = np.array([x.reshape((784, )) for x in X_full])
y_full = scipy.io.loadmat('./matlabFiles/labels28.mat')['matLabels'].ravel() - 1

# p = Augmentor.Pipeline(X_full)
# p.rotate(probability=0.7, max_left=10, max_right=10)
# p.sample(5)
# image = np.array_str(img)
# image = image.replace('0', ' ').replace('1', '#')
