import scipy.io
import numpy as np


def extract_data_from_file(filename,collision_number=None,min_torque=None):
    if filename=='KUKA_HS_dataset':
        name_joints_positions = 'joints_pos'
        name_joints_torques = 'joints_torque'
        name_labels = 'gt_hs'
    else:
        name_joints_positions = 'pos'
        name_joints_torques = 'torque_ext'
    
    print('Read {} ...'.format(filename))
    mat = scipy.io.loadmat('dataset/{}.mat'.format(filename))
    joints_torques = mat[name_joints_torques]
    if filename=='KUKA_HS_dataset':
        joints_torques_con = joints_torques
        collisions_labels = mat[name_labels]
    else:
        joints_torques1 = joints_torques[0::2]
        joints_torques2 = joints_torques[1::2]
        labels1 = np.zeros((joints_torques1.shape[0],1))
        labels2 = np.zeros((joints_torques2.shape[0],1))
        indexes1 = np.logical_not(np.all(abs(joints_torques1)<=min_torque, axis=1))
        indexes2 = np.logical_not(np.all(abs(joints_torques2)<=min_torque, axis=1))
        labels1[indexes1] = collision_number 
        labels2[indexes2] = collision_number 
        joints_torques_con = np.concatenate((joints_torques1, joints_torques2), axis=0)
        collisions_labels = np.concatenate((labels1, labels2), axis=0)


    print('Torques array shape: {}'.format(joints_torques_con.shape))
    print('Labels array shape: {}'.format(collisions_labels.shape))
    return joints_torques_con, collisions_labels




joints_torques_hard, lebels_hard = extract_data_from_file(filename='hard_only',collision_number=1,min_torque=2)
joints_torques_soft, lebels_soft = extract_data_from_file(filename='soft_only',collision_number=2,min_torque=2)
joints_torques_mixed, lebels_mixed = extract_data_from_file(filename='KUKA_HS_dataset')

data_hard = {'X':joints_torques_hard, 'y':lebels_hard}
data_soft = {'X':joints_torques_soft, 'y':lebels_soft}
data_mixed = {'X':joints_torques_mixed, 'y':lebels_mixed}

np.save("dataset/data_hard.npy", data_hard)
np.save("dataset/data_soft.npy", data_soft)
np.save("dataset/data_mixed.npy", data_mixed)