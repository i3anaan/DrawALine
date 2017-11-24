import scipy.io
import numpy as np
import matplotlib
#import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# this may be redundant
matplotlib.rcParams['backend'] = "Qt4Agg"

# load the data
X = scipy.io.loadmat('./matlabFiles/data.mat')['data']
y = scipy.io.loadmat('./matlabFiles/label.mat')['labels'].ravel() - 1

# print the shapes
print("size of X: " + str(X.shape))
print("size of y: " + str(y.shape))

#image processing
#img = X[0, :]
#img = np.reshape(img, (16, 16)).T
#print(img)
#plt.imshow(img, cmap='gray')
#plt.savefig('test.png')

# build the model
logistic = LogisticRegression()
logistic.fit(X, y)

# show model predictions on some instances
for i in range(0, 1999, 200):
    img = X[i]
    print(img.reshape((16, 16)).T)
    prediction = logistic.predict(np.array([img]))
    print("Predicted value: " + str(prediction[0]))
    print("True value:      " + str(y[i]))

# overall accuracy of the model
print("Overall accuracy: " + str(logistic.score(X, y)))
