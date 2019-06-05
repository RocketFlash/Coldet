import scipy.io
import numpy as np
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import generate_binary_structure
import cv2


def extract_data_from_file(filename, for_classification=True):
    inputs = []
    torques = []
    labels = []

    print('Read {} ...'.format(filename))
    mat = scipy.io.loadmat(
        'dataset/collision_detection/{}.mat'.format(filename))

    struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    for i in range(mat['dataset'].shape[1]):
        inputs.append(mat['dataset'][0, i][0][0][1][:, :, 7:])
        torques.append(mat['dataset'][0, i][0][0][3])
        label = mat['dataset'][0, i][0][0][2]

        if for_classification:
            label_modified = label
        else:
            label_modified = binary_dilation(
                label, structure=struct, iterations=1).astype(label.dtype)

        labels.append(label_modified)
        # cv2.imwrite('dataset/images/label{}.png'.format(i), label*255)
        # cv2.imwrite('dataset/images/label_modified{}.png'.format(i),
        #             label_modified*255)

    print('Reading done!')
    return np.array(inputs), np.array(torques), np.array(labels)


for j in range(10):
    X_jac, X_tor, y_col = extract_data_from_file(
        filename='dataset_unet{}'.format(j+1))

    data_collision = {'X1': X_jac, 'X2': X_tor, 'y': y_col}
    np.save("dataset/collision_dataset_classification{}.npy".format(j+1), data_collision)
    print('Done: collision_dataset_classification{}.npy file was created!'.format(j+1))

# X_jac, X_tor, y_col = extract_data_from_file(
#     filename='mini_dataset')

# data_collision = {'X1': X_jac, 'X2': X_tor, 'y': y_col}
# np.save("dataset/mini_datase.npy", data_collision)
# print('Done: mini_datase.npy file was created!')
