# Copyright (c) 2020 Carnegie Mellon University, Wenshan Wang <wenshanw@andrew.cmu.edu>
# For License information please see the LICENSE file in the root directory.
# Cridit: Xiangwei Wang https://github.com/TimingSpace

import numpy as np
from scipy.spatial.transform import Rotation as R

def line2mat(line_data):
    mat = np.eye(4)
    mat[0:3,:] = line_data.reshape(3,4)
    return np.matrix(mat)

def motion2pose(data):
    data_size = len(data)
    all_pose = [] # np.zeros((data_size+1, 4, 4))
    all_pose.append(np.eye(4,4)) #[0,:] = np.eye(4,4)
    pose = np.eye(4,4)
    for i in range(0,data_size):
        pose = pose.dot(data[i])
        all_pose.append(pose)
    return all_pose

def pose2motion(data):
    data_size = len(data)
    all_motion = []
    for i in range(0,data_size-1):
        motion = np.linalg.inv(data[i]).dot(data[i+1])
        all_motion.append(motion)

    return np.array(all_motion) # N x 4 x 4

def SE2se(SE_data):
    result = np.zeros((6))
    result[0:3] = np.array(SE_data[0:3,3].T)
    result[3:6] = SO2so(SE_data[0:3,0:3]).T
    return result
    
def SO2so(SO_data):
    return R.from_matrix(SO_data).as_rotvec()

def so2SO(so_data):
    return R.from_rotvec(so_data).as_matrix()

def se2SE(se_data):
    result_mat = np.matrix(np.eye(4))
    result_mat[0:3,0:3] = so2SO(se_data[3:6])
    result_mat[0:3,3]   = np.matrix(se_data[0:3]).T
    return result_mat
### can get wrong result
def se_mean(se_datas):
    all_SE = np.matrix(np.eye(4))
    for i in range(se_datas.shape[0]):
        se = se_datas[i,:]
        SE = se2SE(se)
        all_SE = all_SE*SE
    all_se = SE2se(all_SE)
    mean_se = all_se/se_datas.shape[0]
    return mean_se

def ses_mean(se_datas):
    se_datas = np.array(se_datas)
    se_datas = np.transpose(se_datas.reshape(se_datas.shape[0],se_datas.shape[1],se_datas.shape[2]*se_datas.shape[3]),(0,2,1))
    se_result = np.zeros((se_datas.shape[0],se_datas.shape[2]))
    for i in range(0,se_datas.shape[0]):
        mean_se = se_mean(se_datas[i,:,:])
        se_result[i,:] = mean_se
    return se_result

def ses2poses(data):
    data_size = data.shape[0]
    all_pose = np.zeros((data_size+1,12))
    temp = np.eye(4,4).reshape(1,16)
    all_pose[0,:] = temp[0,0:12]
    pose = np.matrix(np.eye(4,4))
    for i in range(0,data_size):
        data_mat = se2SE(data[i,:])
        pose = pose*data_mat
        pose_line = np.array(pose[0:3,:]).reshape(1,12)
        all_pose[i+1,:] = pose_line
    return all_pose

def SEs2ses(motion_data):
    data_size = motion_data.shape[0]
    ses = np.zeros((data_size,6))
    for i in range(0,data_size):
        SE = np.matrix(np.eye(4))
        SE[0:3,:] = motion_data[i,:].reshape(3,4)
        ses[i,:] = SE2se(SE)
    return ses

def so2quat(so_data):
    so_data = np.array(so_data)
    theta = np.sqrt(np.sum(so_data*so_data))
    axis = so_data/theta
    quat=np.zeros(4)
    quat[0:3] = np.sin(theta/2)*axis
    quat[3] = np.cos(theta/2)
    return quat

def quat2so(quat_data):
    quat_data = np.array(quat_data)
    sin_half_theta = np.sqrt(np.sum(quat_data[0:3]*quat_data[0:3]))
    axis = quat_data[0:3]/sin_half_theta
    cos_half_theta = quat_data[3]
    theta = 2*np.arctan2(sin_half_theta,cos_half_theta)
    so = theta*axis
    return so

# input so_datas batch*channel*height*width
# return quat_datas batch*numner*channel
def sos2quats(so_datas,mean_std=[[1],[1]]):
    so_datas = np.array(so_datas)
    so_datas = so_datas.reshape(so_datas.shape[0],so_datas.shape[1],so_datas.shape[2]*so_datas.shape[3])
    so_datas = np.transpose(so_datas,(0,2,1))
    quat_datas = np.zeros((so_datas.shape[0],so_datas.shape[1],4))
    for i_b in range(0,so_datas.shape[0]):
        for i_p in range(0,so_datas.shape[1]):
            so_data = so_datas[i_b,i_p,:]
            quat_data = so2quat(so_data)
            quat_datas[i_b,i_p,:] = quat_data
    return quat_datas

def SO2quat(SO_data):
    rr = R.from_matrix(SO_data)
    return rr.as_quat()

def quat2SO(quat_data):
    return R.from_quat(quat_data).as_matrix()


def pos_quat2SE(quat_data):
    SO = R.from_quat(quat_data[3:7]).as_matrix()
    SE = np.matrix(np.eye(4))
    SE[0:3,0:3] = np.matrix(SO)
    SE[0:3,3]   = np.matrix(quat_data[0:3]).T
    SE = np.array(SE[0:3,:]).reshape(1,12)
    return SE


def pos_quats2SEs(quat_datas):
    data_len = quat_datas.shape[0]
    SEs = np.zeros((data_len,12))
    for i_data in range(0,data_len):
        SE = pos_quat2SE(quat_datas[i_data,:])
        SEs[i_data,:] = SE
    return SEs


def pos_quats2SE_matrices(quat_datas):
    data_len = quat_datas.shape[0]
    SEs = []
    for quat in quat_datas:
        SO = R.from_quat(quat[3:7]).as_matrix()
        SE = np.eye(4)
        SE[0:3,0:3] = SO
        SE[0:3,3]   = quat[0:3]
        SEs.append(SE)
    return SEs

def SE2pos_quat(SE_data):
    pos_quat = np.zeros(7)
    pos_quat[3:] = SO2quat(SE_data[0:3,0:3])
    pos_quat[:3] = SE_data[0:3,3].T
    return pos_quat