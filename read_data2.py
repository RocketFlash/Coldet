import scipy.io
import matplotlib.pyplot as plt

filename = 'hard_only'
name_joints_positions = 'joints_pos'
name_joints_torques = 'joints_torque'
name_labels = 'gt_hs'

# print('Reading .mat file')
mat = scipy.io.loadmat('dataset/{}.mat'.format(filename))
print(mat)
# joints_positions = mat[name_joints_positions]
# joints_torques = mat[name_joints_torques]
# collisions_labels = mat[name_labels]