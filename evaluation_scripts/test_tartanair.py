import sys
sys.path.append('droid_slam')
sys.path.append('thirdparty/tartanair_tools')

from evaluation.tartanair_evaluator import TartanAirEvaluator

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob
import time
import yaml
import argparse

from droid import Droid
from droid_async import DroidAsync

# camera baseline hardcoded to 0.1m
STEREO_SCALE_FACTOR = 2.5

MONO_TEST_SCENES = [f"M{s}{i:03d}" for s in ["E", "H"] for i in range(8)]
STEREO_TEST_SCENES = [f"S{s}{i:03d}" for s in ["E", "H"] for i in range(8)]


def image_stream(datapath, image_size=[384, 512], intrinsics_vec=[320.0, 320.0, 320.0, 240.0], stereo=False):
    """ image generator """

    # read all png images in folder
    ht0, wd0 = [480, 640]

    if stereo:
        images_left = sorted(glob.glob(os.path.join(datapath, 'image_left/*.png')))
        images_right = sorted(glob.glob(os.path.join(datapath, 'image_right/*.png')))

    else:
        if os.path.exists(os.path.join(datapath, "image_left")):
            images_left = sorted(glob.glob(os.path.join(datapath, 'image_left/*.png')))
        else:
            images_left = sorted(glob.glob(os.path.join(datapath, '*.png')))

    data = []
    for t in range(len(images_left)):
        images = [ cv2.resize(cv2.imread(images_left[t]), (image_size[1], image_size[0])) ]
        if stereo:
            images += [ cv2.resize(cv2.imread(images_right[t]), (image_size[1], image_size[0])) ]

        images = torch.from_numpy(np.stack(images, 0)).permute(0,3,1,2)
        intrinsics = .8 * torch.as_tensor(intrinsics_vec)

        data.append((t, images, intrinsics))

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath")
    parser.add_argument("--gt_path")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=1000)
    parser.add_argument("--image_size", default=[384,512])
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--plot_curve", action="store_true")
    parser.add_argument("--scene", type=str)

    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.5)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--keyframe_thresh", type=float, default=3.0)
    parser.add_argument("--frontend_thresh", type=float, default=15)
    parser.add_argument("--frontend_window", type=int, default=20)
    parser.add_argument("--frontend_radius", type=int, default=1)
    parser.add_argument("--frontend_nms", type=int, default=1)

    parser.add_argument("--backend_thresh", type=float, default=20.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)

    # damped linear velocity model
    parser.add_argument("--motion_damping", type=int, default=0.5)

    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--asynchronous", action="store_true")
    parser.add_argument("--frontend_device", type=str, default="cuda")
    parser.add_argument("--backend_device", type=str, default="cuda")

    args = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn')


    if not os.path.isdir("figures"):
        os.mkdir("figures")

    test_scenes = STEREO_TEST_SCENES if args.stereo else MONO_TEST_SCENES

    # evaluate on a specific scene
    if args.scene is not None:
        test_scenes = [args.scene]

    ate_list = []
    for scene in test_scenes:
        print("Performing evaluation on {}".format(scene))
        torch.cuda.empty_cache()

        droid = DroidAsync(args) if args.asynchronous else Droid(args)

        scenedir = os.path.join(args.datapath, scene)
        gt_file = os.path.join(args.gt_path, f"{scene}.txt")

        for (tstamp, image, intrinsics) in tqdm(image_stream(scenedir, stereo=args.stereo), desc=scene):
            droid.track(tstamp, image, intrinsics=intrinsics)

        # fill in non-keyframe poses + global BA
        traj_est = droid.terminate(image_stream(scenedir))

        if args.stereo:
            traj_est[:, :3] *= STEREO_SCALE_FACTOR

        ### do evaluation ###
        evaluator = TartanAirEvaluator()
        traj_ref = np.loadtxt(gt_file, delimiter=' ')[:, [1, 2, 0, 4, 5, 3, 6]] # ned -> xyz

        # usually stereo should not be scale corrected, but we are comparing monocular and stereo here
        results = evaluator.evaluate_one_trajectory(
            traj_ref, traj_est, scale=not args.stereo, title=scenedir[-20:].replace('/', '_'))
        
        print(results)
        ate_list.append(results["ate_score"])

    print("Results")
    for (scene, ate) in zip(test_scenes, ate_list):
        print(f"{scene}: {ate}")

    print(ate_list)
    print("Mean ATE", np.mean(ate_list))

    if args.plot_curve:
        import matplotlib.pyplot as plt
        ate = np.array(ate_list)
        xs = np.linspace(0.0, 1.0, 512)
        ys = [np.count_nonzero(ate < t) / ate.shape[0] for t in xs]

        plt.plot(xs, ys)
        plt.xlabel("ATE [m]")
        plt.ylabel("% runs")
        plt.show()

