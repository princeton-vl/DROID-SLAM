# Copyright (c) 2020 Carnegie Mellon University, Wenshan Wang <wenshanw@andrew.cmu.edu>
# For License information please see the LICENSE file in the root directory.

import numpy as np
from . import transformation as tf

def shift0(traj): 
    '''
    Traj: a list of [t + quat]
    Return: translate and rotate the traj
    '''
    traj_ses = tf.pos_quats2SE_matrices(np.array(traj))
    traj_init = traj_ses[0]
    traj_init_inv = np.linalg.inv(traj_init)
    new_traj = []
    for tt in traj_ses:
        ttt=traj_init_inv.dot(tt)
        new_traj.append(tf.SE2pos_quat(ttt))
    return np.array(new_traj)

def ned2cam(traj):
    '''
    transfer a ned traj to camera frame traj
    '''
    T = np.array([[0,1,0,0],
                  [0,0,1,0],
                  [1,0,0,0],
                  [0,0,0,1]], dtype=np.float32) 
    T_inv = np.linalg.inv(T)
    new_traj = []
    traj_ses = tf.pos_quats2SE_matrices(np.array(traj))

    for tt in traj_ses:
        ttt=T.dot(tt).dot(T_inv)
        new_traj.append(tf.SE2pos_quat(ttt))
        
    return np.array(new_traj)

def cam2ned(traj):
    '''
    transfer a camera traj to ned frame traj
    '''
    T = np.array([[0,0,1,0],
                  [1,0,0,0],
                  [0,1,0,0],
                  [0,0,0,1]], dtype=np.float32) 
    T_inv = np.linalg.inv(T)
    new_traj = []
    traj_ses = tf.pos_quats2SE_matrices(np.array(traj))

    for tt in traj_ses:
        ttt=T.dot(tt).dot(T_inv)
        new_traj.append(tf.SE2pos_quat(ttt))
        
    return np.array(new_traj)


def trajectory_transform(gt_traj, est_traj):
    '''
    1. center the start frame to the axis origin
    2. align the GT frame (NED) with estimation frame (camera)
    '''
    gt_traj_trans = shift0(gt_traj)
    est_traj_trans = shift0(est_traj)

    # gt_traj_trans = ned2cam(gt_traj_trans)
    # est_traj_trans = cam2ned(est_traj_trans)

    return gt_traj_trans, est_traj_trans

def rescale_bk(poses_gt, poses):
    motion_gt = tf.pose2motion(poses_gt)
    motion    = tf.pose2motion(poses)
    
    speed_square_gt = np.sum(motion_gt[:,0:3,3]*motion_gt[:,0:3,3],1)
    speed_gt = np.sqrt(speed_square_gt)
    speed_square    = np.sum(motion[:,0:3,3]*motion[:,0:3,3],1)
    speed = np.sqrt(speed_square)
    # when the speed is small, the scale could become very large
    # import ipdb;ipdb.set_trace()
    mask = (speed_gt>0.0001) # * (speed>0.00001)
    scale = np.mean((speed[mask])/speed_gt[mask])
    scale = 1.0/scale
    motion[:,0:3,3] = motion[:,0:3,3]*scale
    pose_update = tf.motion2pose(motion)
    return  pose_update, scale

def pose2trans(pose_data):
    data_size = len(pose_data)
    trans = []
    for i in range(0,data_size-1):
        tran = np.array(pose_data[i+1][:3]) - np.array(pose_data[i][:3]) # np.linalg.inv(data[i]).dot(data[i+1])
        trans.append(tran)

    return np.array(trans) # N x 3


def rescale(poses_gt, poses):
    '''
    similar to rescale
    poses_gt/poses: N x 7 poselist in quaternion format
    '''
    trans_gt = pose2trans(poses_gt)
    trans    = pose2trans(poses)
    
    speed_square_gt = np.sum(trans_gt*trans_gt,1)
    speed_gt = np.sqrt(speed_square_gt)
    speed_square    = np.sum(trans*trans,1)
    speed = np.sqrt(speed_square)
    # when the speed is small, the scale could become very large
    # import ipdb;ipdb.set_trace()
    mask = (speed_gt>0.0001) # * (speed>0.00001)
    scale = np.mean((speed[mask])/speed_gt[mask])
    scale = 1.0/scale
    poses[:,0:3] = poses[:,0:3]*scale
    return  poses, scale

def trajectory_scale(traj, scale):
    for ttt in traj:
        ttt[0:3,3] = ttt[0:3,3]*scale
    return traj
 
def timestamp_associate(first_list, second_list, max_difference):
    """
    Associate two trajectory of [stamp,data]. As the time stamps never match exactly, we aim 
    to find the closest match for every input tuple.
    
    Input:
    first_list -- first list of (stamp,data)
    second_list -- second list of (stamp,data)
    max_difference -- search radius for candidate generation

    Output:
    first_res: matched data from the first list
    second_res: matched data from the second list
    
    """
    first_dict = dict([(l[0],l[1:]) for l in first_list if len(l)>1])
    second_dict = dict([(l[0],l[1:]) for l in second_list if len(l)>1])

    first_keys = first_dict.keys()
    second_keys = second_dict.keys()
    potential_matches = [(abs(a - b ), a, b) 
                         for a in first_keys 
                         for b in second_keys 
                         if abs(a - b) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))
    
    matches.sort()

    first_res = []
    second_res = []
    for t1, t2 in matches:
        first_res.append(first_dict[t1])
        second_res.append(second_dict[t2])
    return np.array(first_res), np.array(second_res)
