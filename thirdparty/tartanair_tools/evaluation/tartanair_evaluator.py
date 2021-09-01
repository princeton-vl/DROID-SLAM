# Copyright (c) 2020 Carnegie Mellon University, Wenshan Wang <wenshanw@andrew.cmu.edu>
# For License information please see the LICENSE file in the root directory.

import numpy as np
from os.path import isdir, isfile

from .evaluator_base import ATEEvaluator, RPEEvaluator, KittiEvaluator, transform_trajs, quats2SEs

# from trajectory_transform import timestamp_associate


def plot_traj(gtposes, estposes, vis=False, savefigname=None, title=''):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(4,4))


    cm = plt.cm.get_cmap('Spectral')

    plt.subplot(111)
    plt.plot(gtposes[:,2],gtposes[:,0], linestyle='dashed',c='k')
    plt.plot(estposes[:, 2], estposes[:, 0],c='#ff7f0e')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend(['Ground Truth','Ours'])
    plt.title(title)

    plt.axis('equal')    

    if savefigname is not None:
        plt.savefig(savefigname)
    
    if vis:
        plt.show()
    
    plt.close(fig)


# 

class TartanAirEvaluator:
    def __init__(self, scale = False, round=1):
        self.ate_eval = ATEEvaluator()
        self.rpe_eval = RPEEvaluator()
        self.kitti_eval = KittiEvaluator()
        
    def evaluate_one_trajectory(self, gt_traj, est_traj, scale=False, title=''):
        """
        scale = True: calculate a global scale
        """

        if gt_traj.shape[0] != est_traj.shape[0]:
            raise Exception("POSEFILE_LENGTH_ILLEGAL")
            
        if gt_traj.shape[1] != 7 or est_traj.shape[1] != 7:
            raise Exception("POSEFILE_FORMAT_ILLEGAL")

        gt_traj = gt_traj.astype(np.float64)
        est_traj = est_traj.astype(np.float64)

        ate_score, gt_ate_aligned, est_ate_aligned = self.ate_eval.evaluate(gt_traj, est_traj, scale)

        plot_traj(np.matrix(gt_ate_aligned), np.matrix(est_ate_aligned), vis=False, savefigname="figures/%s.pdf"%title, title=title)

        est_ate_aligned = np.array(est_ate_aligned)
        gt_SEs, est_SEs = quats2SEs(gt_ate_aligned, est_ate_aligned)



        rpe_score = self.rpe_eval.evaluate(gt_SEs, est_SEs)
        kitti_score = self.kitti_eval.evaluate(gt_SEs, est_SEs)

        return {'ate_score': ate_score, 'rpe_score': rpe_score, 'kitti_score': kitti_score}


if __name__ == "__main__":
    
    # scale = True for monocular track, scale = False for stereo track
    aicrowd_evaluator = TartanAirEvaluator()
    result = aicrowd_evaluator.evaluate_one_trajectory('pose_gt.txt', 'pose_est.txt', scale=True)
    print(result)
