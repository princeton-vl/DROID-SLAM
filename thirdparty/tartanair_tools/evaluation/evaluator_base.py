# Copyright (c) 2020 Carnegie Mellon University, Wenshan Wang <wenshanw@andrew.cmu.edu>
# For License information please see the LICENSE file in the root directory.

import numpy as np
from .trajectory_transform import trajectory_transform, rescale
from .transformation import pos_quats2SE_matrices, SE2pos_quat


np.set_printoptions(suppress=True, precision=2, threshold=100000)

def transform_trajs(gt_traj, est_traj, cal_scale):
    gt_traj, est_traj = trajectory_transform(gt_traj, est_traj)
    if cal_scale :
        est_traj, s = rescale(gt_traj, est_traj)
        print('  Scale, {}'.format(s))
    else:
        s = 1.0
    return gt_traj, est_traj, s

def quats2SEs(gt_traj, est_traj):
    gt_SEs = pos_quats2SE_matrices(gt_traj)
    est_SEs = pos_quats2SE_matrices(est_traj)
    return gt_SEs, est_SEs

from .evaluate_ate_scale import align, plot_traj


class ATEEvaluator(object):
    def __init__(self):
        super(ATEEvaluator, self).__init__()


    def evaluate(self, gt_traj, est_traj, scale):
        gt_xyz = np.matrix(gt_traj[:,0:3].transpose())
        est_xyz = np.matrix(est_traj[:, 0:3].transpose())

        rot, trans, trans_error, s = align(gt_xyz, est_xyz, scale)
        print('  ATE scale: {}'.format(s))
        error = np.sqrt(np.dot(trans_error,trans_error) / len(trans_error))

        # align two trajs 
        est_SEs = pos_quats2SE_matrices(est_traj)
        T = np.eye(4) 
        T[:3,:3] = rot
        T[:3,3:] = trans 
        T = np.linalg.inv(T)
        est_traj_aligned = []
        for se in est_SEs:
            se[:3,3] = se[:3,3] * s
            se_new = T.dot(se)
            se_new = SE2pos_quat(se_new)
            est_traj_aligned.append(se_new)


        return error, gt_traj, est_traj_aligned

# =======================

from .evaluate_rpe import evaluate_trajectory

class RPEEvaluator(object):
    def __init__(self):
        super(RPEEvaluator, self).__init__()


    def evaluate(self, gt_SEs, est_SEs):
        result = evaluate_trajectory(gt_SEs, est_SEs)
        
        trans_error = np.array(result)[:,2]
        rot_error = np.array(result)[:,3]

        trans_error_mean = np.mean(trans_error)
        rot_error_mean = np.mean(rot_error)

        # import ipdb;ipdb.set_trace()

        return (rot_error_mean, trans_error_mean)

# =======================

from .evaluate_kitti import evaluate as kittievaluate

class KittiEvaluator(object):
    def __init__(self):
        super(KittiEvaluator, self).__init__()

    # return rot_error, tra_error
    def evaluate(self, gt_SEs, est_SEs):
        # trajectory_scale(est_SEs, 0.831984631412)
        error = kittievaluate(gt_SEs, est_SEs)
        return error
