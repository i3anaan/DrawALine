import scipy.io
import numpy as np
import matplotlib
import sys
#import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# this may be redundant
matplotlib.rcParams['backend'] = "Qt4Agg"

# load the data
X = scipy.io.loadmat('./matlabFiles/data.mat')['data']
y = scipy.io.loadmat('./matlabFiles/label.mat')['labels'].ravel() - 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# print the shapes
print("size of X: " + str(X.shape))
print("size of y: " + str(y.shape))

#image processing
#img = X[0, :]
#img = np.reshape(img, (16, 16)).T
#print(img)
#plt.imshow(img, cmap='gray')
#plt.savefig('test.png')

def visualizeImage( img ):
   image = np.array_str(img)
   image = image.replace('0', ' ').replace('1', '#')
   return image

# build the model
logistic = LogisticRegression(max_iter=1000, C=0.1)
logistic.fit(X_train, y_train)
svm_model = SVC()
svm_model.fit(X_train, y_train)
clf = MLPClassifier(solver='lbfgs', alpha=3, hidden_layer_sizes=(25, 10), random_state=1)
clf.fit(X_train, y_train)


# overall accuracy of the model
print("Accuracy Logistic Regression: " + str(logistic.score(X_train, y_train)) + " - " + str(logistic.score(X_test, y_test)))
print("Accuracy Support Vector Machine: " + str(svm_model.score(X_train, y_train)) + " - " + str(svm_model.score(X_test, y_test)))
print("Accuracy Neural Network: " + str(clf.score(X_train, y_train)) + " - " + str(clf.score(X_test, y_test)))

if (len(sys.argv) <= 1 or sys.argv[1] != "--no-examples"):
    print("some examples from the NN:")
    # show model predictions on some instances
    for i in range(0, 399):
        img = X_test[i]
        print(visualizeImage(img.reshape((16, 16)).T))
        prediction = clf.predict(np.array([img]))
        print("Predicted value: " + str(prediction[0]))
        print("True value:      " + str(y_test[i]))
        userin = input("Continue? (Y/n):")
        if userin == "n":
            break
