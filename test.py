import sys
import scipy.io
import numpy as np
import matplotlib
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# this may be redundant
matplotlib.rcParams['backend'] = "Qt4Agg"

# load the data
X = scipy.io.loadmat('./matlabFiles/data28.mat')['matImages'][0]
X = np.array([x.reshape((784,)) for x in X])
y = scipy.io.loadmat('./matlabFiles/labels28.mat')['matLabels'].ravel() - 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

# print the shapes
print("Training set size: " + str(X_train.shape))
print("Test set size:     " + str(X_test.shape))

#image processing
#img = X[0, :]
#img = np.reshape(img, (16, 16)).T
#print(img)
#plt.imshow(img, cmap='gray')
#plt.savefig('test.png')

# build the model
logistic = LogisticRegression(max_iter=1000, C=0.01)
logistic.fit(X_train, y_train)
#svm_model = SVC(C=3) #10
#svm_model.fit(X_train, y_train)
#alpha_test = 0.03
#print("alpha: " + str(alpha_test))
#clf = MLPClassifier(solver='adam', alpha=alpha_test, hidden_layer_sizes=(800, 10), random_state=1, max_iter=10000)
#clf.fit(X_train, y_train)

# overall accuracy of the model
print("Accuracy Logistic Regression: " + str(logistic.score(X_train, y_train)) + " - " + str(logistic.score(X_test, y_test)))
#print("Accuracy Support Vector Machine: " + str(svm_model.score(X_train, y_train)) + " - " + str(svm_model.score(X_test, y_test)))
#print("Accuracy Neural Network: " + str(clf.score(X_train, y_train)) + " - " + str(clf.score(X_cv, y_cv)))

def visualize_img (img):
    image = np.array_str(img)
    image = image.replace('0', ' ').replace('1', '#')
    return image

def classification_error_training_size (model, sizes, training_set, test_set):
    training_errors = []
    test_errors = []
    (X_tr, y_tr) = training_set
    (X_te, y_te) = test_set

    for size in sizes:
        model.fit(X_tr[1:size], y_tr[1:size])
        training_errors.append(model.score(X_tr[1:size], y_tr[1:size]))
        test_errors.append(model.score(X_te, y_te))

    return [(sizes, training_errors), (sizes, test_errors)]

# show model predictions on some instances
def print_examples(model, test_set, test_set_answers):
    print("some examples: ")
    for i in range(0, len(test_set)):
        userin = input("Continue? (Y/n):")
        if userin == "n":
            break
        img = test_set[i]
        print(visualize_img(img.reshape((28, 28))))
        prediction = model.predict(np.array([img]))
        print("Predicted value: " + str(prediction[0]))
        print("True value:      " + str(test_set_answers[i]))

if (len(sys.argv) <= 1 or sys.argv[1] != "--no-examples"):
    print_examples(logistic, X_test, y_test)

set_sizes = [10,50,100,200,300,400,500,600,800,1000,1200,1400,1600,1800]
print(classification_error_training_size(logistic, set_sizes, (X_train, y_train), (X_test, y_test)))
