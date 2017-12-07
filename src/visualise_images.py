"""
Functions to visualise the images using ASCII art.
"""

import numpy as np

def visualize_img(img):
    """
    Turns the array into a string and makes it more readable.

    :param img - A 2D array containing a single image
    :return The original array as a string with all 0 -> " " and 1 -> "#"
    """

    image = np.array_str(img)
    image = image.replace('0', ' ').replace('1', '#')
    return image

def print_examples(model, test_set, test_set_answers):
    """
    Visualize model predictions on some instances.

    :param model - fully trained model
    :param test_set - the test set X values
    :param test_set_answers - the test set y values
    :return null
    """

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
