import torch
import lietorch
import numpy as np

import time
from lietorch import SE3
from droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from droid_frontend import DroidFrontend
from droid_backend import DroidAsyncBackend
from trajectory_filler import PoseTrajectoryFiller
from align import align_pose_fragements

from collections import OrderedDict
from torch.multiprocessing import Process


def load_network(weights, device="cuda:0"):
    net = DroidNet()
    state_dict = OrderedDict(
        [(k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()]
    )

    state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
    state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
    state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
    state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

    net.load_state_dict(state_dict)
    net.to(device=device)
    net.eval()

    return net


def backend_process(args, depth_video1, depth_video2, device="cuda"):
    torch.set_num_threads(8)

    seperate_device = False
    if device != "cuda":
        seperate_device = True
        torch.cuda.set_device(device)

    with torch.no_grad():
        net = load_network(args.weights, device=device)

        # use more compute if running backend on seperate device
        sleep_time = 10
        num_iters = 12 if seperate_device else 8

        backend = DroidAsyncBackend(net, depth_video2, args)

        while 1:
            if depth_video1.counter.value > 32 or depth_video2.ready.value:
                # check if the end of the video has been reached
                is_last_iteration = depth_video2.ready.value

                # don't align scale for RGB-D or stereo videos
                align_scale = not depth_video2.stereo and not torch.any(
                    depth_video1.disps_sens
                )

                if is_last_iteration:
                    t0 = max(depth_video2.counter.value - 2, 0)
                    t1 = depth_video1.counter.value

                else:
                    t0 = max(depth_video2.counter.value - 2, 0)
                    # avoid keyframing area
                    t1 = depth_video1.counter.value - 5

                with depth_video1.get_lock():
                    pose1_copy = depth_video1.poses.clone().to(device=device)
                    disps1_copy = depth_video1.disps.clone().to(device=device)

                if t0 > 0:
                    # align pose and scale
                    dP, s = align_pose_fragements(
                        pose1_copy[t0 - 10 : t0 - 1],
                        depth_video2.poses[t0 - 10 : t0 - 1],
                    )

                    if not align_scale:
                        s = 1.0

                    pose1_copy.data[..., :3] *= s

                else:
                    s = 1.0
                    dP = SE3.IdentityLike(SE3(depth_video2.poses[[t0]]))

                with depth_video1.get_lock():
                    depth_video2.poses[t0:t1] = (dP * SE3(pose1_copy[t0:t1])).data.to(
                        device=device
                    )
                    depth_video2.disps[t0:t1] = disps1_copy[t0:t1].to(device=device) / s

                    depth_video2.disps_sens[t0:t1] = depth_video1.disps_sens[t0:t1].to(
                        device=device
                    )
                    depth_video2.images[t0:t1] = depth_video1.images[t0:t1].to(
                        device=device
                    )
                    depth_video2.tstamp[t0:t1] = depth_video1.tstamp[t0:t1].to(
                        device=device
                    )
                    depth_video2.intrinsics[t0:t1] = depth_video1.intrinsics[t0:t1].to(
                        device=device
                    )
                    depth_video2.fmaps[t0:t1] = depth_video1.fmaps[t0:t1].to(
                        device=device
                    )
                    depth_video2.nets[t0:t1] = depth_video1.nets[t0:t1].to(
                        device=device
                    )
                    depth_video2.inps[t0:t1] = depth_video1.inps[t0:t1].to(
                        device=device
                    )

                depth_video2.counter.value = t1
                backend(num_iters, normalize=False)

                if is_last_iteration:
                    break

                if not depth_video2.ready.value > 0:
                    time.sleep(sleep_time)

    del depth_video2


class DroidAsync:
    def __init__(self, args):
        super(DroidAsync, self).__init__()
        net = load_network(args.weights)
        self.args = args
        self.disable_vis = args.disable_vis

        torch.set_num_threads(8)

        self.frontend_device = (
            "cuda" if not hasattr(args, "frontend_device") else args.frontend_device
        )
        self.backend_device = (
            "cuda" if not hasattr(args, "backend_device") else args.backend_device
        )

        net = load_network(args.weights, device=self.frontend_device)

        self.video1 = DepthVideo(
            args.image_size,
            args.buffer,
            stereo=args.stereo,
            device=self.frontend_device,
        )
        self.video2 = DepthVideo(
            args.image_size, args.buffer, stereo=args.stereo, device=self.backend_device
        )

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(net, self.video1, thresh=args.filter_thresh)

        # frontend process
        self.frontend = DroidFrontend(net, self.video1, self.args)

        # backend process
        self.backend_proc = Process(
            target=backend_process,
            args=(args, self.video1, self.video2, self.backend_device),
        )
        self.backend_proc.start()

        # visualizer
        if not self.disable_vis:
            from visualizer.droid_visualizer import visualization_fn

            self.visualizer = Process(
                target=visualization_fn,
                args=(
                    self.video1,
                    self.video2,
                ),
            )
            self.visualizer.start()

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(net, self.video2)

    def track(self, tstamp, image, depth=None, intrinsics=None):
        """main thread - update map"""

        with torch.no_grad():
            # check there is enough motion
            self.filterx.track(tstamp, image, depth, intrinsics)

            # local bundle adjustment
            self.frontend()

    def terminate(self, stream=None):
        """terminate the visualization process, return poses [t, q]"""

        self.video2.ready.value = 1
        self.backend_proc.join()

        del self.frontend
        del self.video1

        torch.cuda.empty_cache()

        # fill in missing non-keyframe poses
        self.traj_filler.video = self.traj_filler.video.to(self.frontend_device)
        camera_trajectory = self.traj_filler(stream)

        return camera_trajectory.inv().data.cpu().numpy()
