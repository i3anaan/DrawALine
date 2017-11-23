import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import toimage

labels = scipy.io.loadmat('./matlabFiles/label.mat')
imgs = scipy.io.loadmat('./matlabFiles/data.mat')

#image processing
img = imgs['data'][0,:]
img = np.reshape(img,(16,16)).T
plt.imshow(img, cmap='gray')
plt.show()