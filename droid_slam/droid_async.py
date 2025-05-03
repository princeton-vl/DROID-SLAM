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


def load_network(weights):
    net = DroidNet()
    state_dict = OrderedDict(
        [(k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()]
    )

    state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
    state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
    state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
    state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

    net.load_state_dict(state_dict)
    net.to("cuda:0")
    net.eval()

    return net


def backend_process(args, depth_video1, depth_video2, device="cuda:0", sleep_time=10):
    torch.cuda.set_device(device)

    with torch.no_grad():
        net = load_network(args.weights)
        backend = DroidAsyncBackend(net, depth_video2, args)

        while 1:
            if depth_video1.counter.value > 32 or depth_video2.ready.value:
                # check if the end of the video has been reached
                is_last_iteration = depth_video2.ready.value

                if is_last_iteration:
                    t0 = max(depth_video2.counter.value - 2, 0)
                    t1 = depth_video1.counter.value

                else:
                    t0 = max(depth_video2.counter.value - 2, 0)
                    t1 = depth_video1.counter.value - 3

                with depth_video1.get_lock():
                    pose1_copy = depth_video1.poses.clone()
                    disps1_copy = depth_video1.disps.clone()

                if t0 > 0:
                    # align pose and scale
                    dP, s = align_pose_fragements(
                        pose1_copy[t0 - 10 : t0 - 1],
                        depth_video2.poses[t0 - 10 : t0 - 1],
                    )

                    pose1_copy.data[..., :3] *= s

                else:
                    s = 1.0
                    dP = SE3.IdentityLike(SE3(depth_video2.poses[[t0]]))

                depth_video2.poses[t0:t1] = (dP * SE3(pose1_copy[t0:t1])).data.to(device=device)

                with depth_video1.get_lock():
                    for t in range(t0, t1):
                        depth_video2.images[t] = depth_video1.images[t].to(device=device)
                        depth_video2.tstamp[t] = depth_video1.tstamp[t].to(device=device)
                        depth_video2.disps[t] = disps1_copy[t].to(device=device) / s
                        depth_video2.intrinsics[t] = depth_video1.intrinsics[t].to(device=device)
                        depth_video2.fmaps[t] = depth_video1.fmaps[t].to(device=device)
                        depth_video2.nets[t] = depth_video1.nets[t].to(device=device)
                        depth_video2.inps[t] = depth_video1.inps[t].to(device=device)

                depth_video2.counter.value = t1
                backend(8, normalize=False)

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

        net = load_network(args.weights)
        # store images, depth, poses, intrinsics (shared between processes)
        self.video1 = DepthVideo(args.image_size, args.buffer, stereo=args.stereo)
        self.video2 = DepthVideo(args.image_size, args.buffer, stereo=args.stereo)

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(net, self.video1, thresh=args.filter_thresh)

        # frontend process
        self.frontend = DroidFrontend(net, self.video1, self.args)

        # backend process
        self.backend_proc = Process(
            target=backend_process, args=(args, self.video1, self.video2)
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

            # self.visualizer = Process(target=visualization_fn, args=(self.video2, None))
            # self.visualizer.start()

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
        torch.cuda.empty_cache()

        camera_trajectory = self.traj_filler(stream)
        return camera_trajectory.inv().data.cpu().numpy()
