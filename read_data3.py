import scipy.io
import numpy as np


mat = scipy.io.loadmat('dataset/collision_detection/mini_dataset.mat')
print(mat['dataset'][0,1][0][0][3])