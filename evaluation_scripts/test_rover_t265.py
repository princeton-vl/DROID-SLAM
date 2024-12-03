import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse

from torch.multiprocessing import Process
from droid import Droid

import torch.nn.functional as F


def show_image(image, title):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow(title, image / 255.0)
    cv2.waitKey(1)

def image_stream(datapath, image_size=[400, 424], stereo=False, stride=1):
    """ image generator """

    K_l = np.array([[280.4362476646957, 0, 434.5911290024899],
                [0, 279.5757903173993, 395.3741210501516],
                [0, 0, 1]])
    d_l = np.array([-0.011532772136434897, 0.0501515488043061, -0.05041450901368907, 0.012741893876582578])
    R_l = np.array([
         0.99986924, -0.01333354,  0.00915   ,
         0.01334595,  0.9999101 , -0.00129711,
        -0.00913188,  0.00141906,  0.9999573,
    ]).reshape(3,3)
    
    P_l = np.array([
        544.53435553,    0.          ,    432.19955826,    0.,
        0.          ,    544.53435553,    341.53060913,    0.,
        0.          ,    0.          ,    1.          ,    0.,
    ]).reshape(3,4)
    map_l = cv2.initUndistortRectifyMap(K_l, d_l, R_l, P_l[:3,:3], (848, 800), cv2.CV_32F)
    
    K_r = np.array([[280.311263999059, 0, 431.35302371548494],
                [0, 279.5434630904508, 388.5071222043099],
                [0, 0, 1]])
    d_r = np.array([-0.011950967309164085, 0.0530642563172375, -0.049469178559530994, 0.011573768486635416])
    R_r = np.array([
         0.99987795, -0.00637892,  0.01426177,
         0.00635955,  0.99997879,  0.00140352,
        -0.01427042, -0.00131265,  0.99989731,
    ]).reshape(3,3)
    
    P_r = np.array([
        544.53435553,    0.          ,    432.19955826,    -34.56813177,
        0.          ,    544.53435553,    341.53060913,    0.,
        0.          ,    0.          ,    1.          ,    0.,
    ]).reshape(3,4)
    map_r = cv2.initUndistortRectifyMap(K_r, d_r, R_r, P_r[:3,:3], (848, 800), cv2.CV_32F)

    intrinsics_vec = [544.53435553, 544.53435553, 432.19955826, 341.53060913]
    ht0, wd0 = 800, 848

    mask_left = cv2.imread("../calib/mask_t265_cam0.png", cv2.IMREAD_GRAYSCALE)
    mask_left = cv2.bitwise_not(mask_left)
    if stereo:
        mask_right = cv2.imread("../calib/mask_t265_cam1.png", cv2.IMREAD_GRAYSCALE)
        mask_right = cv2.bitwise_not(mask_right) 

    # read all png images in folder
    images_left = sorted(glob.glob(os.path.join(datapath, 'cam_left/*.png')))[::stride]
    images_right = [x.replace('cam_left', 'cam_right') for x in images_left]

    for t, (imgL, imgR) in enumerate(zip(images_left, images_right)):
        if stereo and not os.path.isfile(imgR):
            continue
        tstamp = float(imgL.split('/')[-1][:-4])
        imgL = cv2.imread(imgL)
        # imgL = cv2.bitwise_and(imgL, imgL, mask=mask_left)
        imgL = cv2.remap(imgL, map_l[0], map_l[1], interpolation=cv2.INTER_LINEAR)
        
        if stereo:
            imgR = cv2.imread(imgR)
            # imgR = cv2.bitwise_and(imgR, imgR, mask=mask_right)
            imgR = cv2.remap(imgR, map_r[0], map_r[1], interpolation=cv2.INTER_LINEAR)
            images = [imgL, imgR]
        else:
            images = [imgL]

        images = [cv2.resize(image, (image_size[1], image_size[0])) for image in images]
        
        images = torch.from_numpy(np.stack(images, 0))
        images = images.permute(0, 3, 1, 2).to("cuda:0", dtype=torch.float32)
        
        ht1, wd1 = image_size
        scale_x = wd1 / wd0
        scale_y = ht1 / ht0
        
        intrinsics = torch.as_tensor(intrinsics_vec).cuda()
        intrinsics[0] *= scale_x
        intrinsics[1] *= scale_y
        intrinsics[2] *= scale_x
        intrinsics[3] *= scale_y

        yield stride*t, images, None, intrinsics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath")
    parser.add_argument("--outputpath")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--image_size", default=[400, 424])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--stereo", action="store_true")

    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--filter_thresh", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--keyframe_thresh", type=float, default=3.5)
    parser.add_argument("--frontend_thresh", type=float, default=16.0)
    parser.add_argument("--frontend_window", type=int, default=16)
    parser.add_argument("--frontend_radius", type=int, default=1)
    parser.add_argument("--frontend_nms", type=int, default=0)

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    print("Running evaluation on {}".format(args.datapath))
    print(args)

    droid = Droid(args)
    time.sleep(5)

    for (t, image, depth, intrinsics) in tqdm(image_stream(args.datapath, image_size=args.image_size,stereo=args.stereo, stride=args.stride)):
        if not args.disable_vis:
            show_image(image[0], "left_image")
            if args.stereo:
                show_image(image[1], "right_image")
            
        droid.track(t, image, intrinsics=intrinsics)

    traj_est = droid.terminate(image_stream(args.datapath, stride=args.stride))

    ### run evaluation ###

    import evo
    from evo.core.trajectory import PoseTrajectory3D
    from evo.tools import file_interface
    from evo.core import sync
    import evo.main_ape as main_ape
    import evo.main_rpe as main_rpe
    from evo.core.metrics import PoseRelation
    from pathlib import Path
    import json

    image_path = os.path.join(args.datapath, 'cam_left')
    images_list = sorted(glob.glob(os.path.join(image_path, '*.png')))[::stride]
    tstamps = [float(x.split('/')[-1][:-4]) for x in images_list]

    traj_est_fused = PoseTrajectory3D(
        positions_xyz=traj_est[:,:3],
        orientations_quat_wxyz=traj_est[:,3:],
        timestamps=np.array(tstamps))

    result_path = os.path.join(args.outputpath)
    Path(result_path).mkdir(parents=True, exist_ok=True)
    result_file_name = "t265_stereo_slam_trajectory.txt" if args.stereo else "t265_mono_slam_trajectory.txt"
    file_interface.write_tum_trajectory_file(os.path.join(result_path, result_file_name), traj_est_fused)
    
    gt_file = os.path.join(args.datapath, 'groundtruth.txt')
    traj_ref = file_interface.read_tum_trajectory_file(gt_file)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est_fused)

    result_ape = main_ape.ape(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=False if args.depth else True)

    with open(os.path.join(result_path, result_file_name.replace("slam_trajectory", "ape_results")), "w") as file:
        file.write(json.dumps(result_ape.stats))

    result_rpe = main_rpe.rpe(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=False if args.depth else True)

    with open(os.path.join(result_path, result_file_name.replace("slam_trajectory", "rpe_results")), "w") as file:
        file.write(json.dumps(result_rpe.stats))


