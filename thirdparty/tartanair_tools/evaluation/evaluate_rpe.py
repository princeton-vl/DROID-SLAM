#!/usr/bin/python
# Software License Agreement (BSD License)
#
# Modified by Wenshan Wang
# Copyright (c) 2013, Juergen Sturm, TUM
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of TUM nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
This script computes the relative pose error from the ground truth trajectory
and the estimated trajectory.
"""

import random
import numpy as np
import sys

def ominus(a,b):
    """
    Compute the relative 3D transformation between a and b.
    
    Input:
    a -- first pose (homogeneous 4x4 matrix)
    b -- second pose (homogeneous 4x4 matrix)
    
    Output:
    Relative 3D transformation from a to b.
    """
    return np.dot(np.linalg.inv(a),b)

def compute_distance(transform):
    """
    Compute the distance of the translational component of a 4x4 homogeneous matrix.
    """
    return np.linalg.norm(transform[0:3,3])

def compute_angle(transform):
    """
    Compute the rotation angle from a 4x4 homogeneous matrix.
    """
    # an invitation to 3-d vision, p 27
    return np.arccos( min(1,max(-1, (np.trace(transform[0:3,0:3]) - 1)/2) ))

def distances_along_trajectory(traj):
    """
    Compute the translational distances along a trajectory. 
    """
    motion = [ominus(traj[i+1],traj[i]) for i in range(len(traj)-1)]
    distances = [0]
    sum = 0
    for t in motion:
        sum += compute_distance(t)
        distances.append(sum)
    return distances
    

def evaluate_trajectory(traj_gt, traj_est, param_max_pairs=10000, param_fixed_delta=False,
                        param_delta=1.00):
    """
    Compute the relative pose error between two trajectories.
    
    Input:
    traj_gt -- the first trajectory (ground truth)
    traj_est -- the second trajectory (estimated trajectory)
    param_max_pairs -- number of relative poses to be evaluated
    param_fixed_delta -- false: evaluate over all possible pairs
                         true: only evaluate over pairs with a given distance (delta)
    param_delta -- distance between the evaluated pairs
    param_delta_unit -- unit for comparison:
                        "s": seconds
                        "m": meters
                        "rad": radians
                        "deg": degrees
                        "f": frames
    param_offset -- time offset between two trajectories (to model the delay)
    param_scale -- scale to be applied to the second trajectory
    
    Output:
    list of compared poses and the resulting translation and rotation error
    """
    
    if not param_fixed_delta:
        if(param_max_pairs==0 or len(traj_est)<np.sqrt(param_max_pairs)):
            pairs = [(i,j) for i in range(len(traj_est)) for j in range(len(traj_est))]
        else:
            pairs = [(random.randint(0,len(traj_est)-1),random.randint(0,len(traj_est)-1)) for i in range(param_max_pairs)]
    else:
        pairs = []
        for i in range(len(traj_est)):
            j = i + param_delta
            if j < len(traj_est): 
                pairs.append((i,j))
        if(param_max_pairs!=0 and len(pairs)>param_max_pairs):
            pairs = random.sample(pairs,param_max_pairs)
        
    result = []
    for i,j in pairs:
        
        error44 = ominus(  ominus( traj_est[j], traj_est[i] ),
                           ominus( traj_gt[j], traj_gt[i] ) )
        
        trans = compute_distance(error44)
        rot = compute_angle(error44)
        
        result.append([i,j,trans,rot])
        
    if len(result)<2:
        raise Exception("Couldn't find pairs between groundtruth and estimated trajectory!")
        
    return result

# import argparse
# if __name__ == '__main__':
#     random.seed(0)

#     parser = argparse.ArgumentParser(description='''
#     This script computes the relative pose error from the ground truth trajectory and the estimated trajectory. 
#     ''')
#     parser.add_argument('groundtruth_file', help='ground-truth trajectory file (format: "timestamp tx ty tz qx qy qz qw")')
#     parser.add_argument('estimated_file', help='estimated trajectory file (format: "timestamp tx ty tz qx qy qz qw")')
#     parser.add_argument('--max_pairs', help='maximum number of pose comparisons (default: 10000, set to zero to disable downsampling)', default=10000)
#     parser.add_argument('--fixed_delta', help='only consider pose pairs that have a distance of delta delta_unit (e.g., for evaluating the drift per second/meter/radian)', action='store_true')
#     parser.add_argument('--delta', help='delta for evaluation (default: 1.0)',default=1.0)
#     parser.add_argument('--delta_unit', help='unit of delta (options: \'s\' for seconds, \'m\' for meters, \'rad\' for radians, \'f\' for frames; default: \'s\')',default='s')
#     parser.add_argument('--offset', help='time offset between ground-truth and estimated trajectory (default: 0.0)',default=0.0)
#     parser.add_argument('--scale', help='scaling factor for the estimated trajectory (default: 1.0)',default=1.0)
#     parser.add_argument('--save', help='text file to which the evaluation will be saved (format: stamp_est0 stamp_est1 stamp_gt0 stamp_gt1 trans_error rot_error)')
#     parser.add_argument('--plot', help='plot the result to a file (requires --fixed_delta, output format: png)')
#     parser.add_argument('--verbose', help='print all evaluation data (otherwise, only the mean translational error measured in meters will be printed)', action='store_true')
#     args = parser.parse_args()
    
#     if args.plot and not args.fixed_delta:
#         sys.exit("The '--plot' option can only be used in combination with '--fixed_delta'")
    
#     traj_gt = np.loadtxt(args.groundtruth_file)
#     traj_est = np.loadtxt(args.estimated_file)
    
#     from trajectory_transform import trajectory_transform
#     traj_gt, traj_est = trajectory_transform(traj_gt, traj_est)

#     traj_gt = tf.pos_quats2SE_matrices(traj_gt)
#     traj_est = tf.pos_quats2SE_matrices(traj_est)

#     result = evaluate_trajectory(traj_gt,
#                                  traj_est,
#                                  int(args.max_pairs),
#                                  args.fixed_delta,
#                                  float(args.delta),
#                                  args.delta_unit)
    
#     trans_error = np.array(result)[:,2]
#     rot_error = np.array(result)[:,3]
    
#     if args.save:
#         f = open(args.save,"w")
#         f.write("\n".join([" ".join(["%f"%v for v in line]) for line in result]))
#         f.close()
    
#     if args.verbose:
#         print "compared_pose_pairs %d pairs"%(len(trans_error))

#         print "translational_error.rmse %f m"%np.sqrt(np.dot(trans_error,trans_error) / len(trans_error))
#         print "translational_error.mean %f m"%np.mean(trans_error)
#         print "translational_error.median %f m"%np.median(trans_error)
#         print "translational_error.std %f m"%np.std(trans_error)
#         print "translational_error.min %f m"%np.min(trans_error)
#         print "translational_error.max %f m"%np.max(trans_error)

#         print "rotational_error.rmse %f deg"%(np.sqrt(np.dot(rot_error,rot_error) / len(rot_error)) * 180.0 / np.pi)
#         print "rotational_error.mean %f deg"%(np.mean(rot_error) * 180.0 / np.pi)
#         print "rotational_error.median %f deg"%(np.median(rot_error) * 180.0 / np.pi)
#         print "rotational_error.std %f deg"%(np.std(rot_error) * 180.0 / np.pi)
#         print "rotational_error.min %f deg"%(np.min(rot_error) * 180.0 / np.pi)
#         print "rotational_error.max %f deg"%(np.max(rot_error) * 180.0 / np.pi)
#     else:
#         print np.mean(trans_error)

#     import ipdb;ipdb.set_trace()

#     if args.plot:    
#         import matplotlib
#         matplotlib.use('Agg')
#         import matplotlib.pyplot as plt
#         import matplotlib.pylab as pylab
#         fig = plt.figure()
#         ax = fig.add_subplot(111)        
#         ax.plot(stamps - stamps[0],trans_error,'-',color="blue")
#         #ax.plot([t for t,e in err_rot],[e for t,e in err_rot],'-',color="red")
#         ax.set_xlabel('time [s]')
#         ax.set_ylabel('translational error [m]')
#         plt.savefig(args.plot,dpi=300)
        

