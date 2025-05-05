import torch
import lietorch
import numpy as np

from lietorch import SE3
from factor_graph import FactorGraph


class DroidBackend:
    def __init__(self, net, video, args):
        self.video = video
        self.update_op = net.update

        # global optimization window
        self.t0 = 0
        self.t1 = 0

        self.upsample = args.upsample
        self.beta = args.beta
        self.backend_thresh = args.backend_thresh
        self.backend_radius = args.backend_radius
        self.backend_nms = args.backend_nms

    @torch.no_grad()
    def __call__(self, steps=12, normalize=True):
        """ main update """

        t = self.video.counter.value
        if normalize:
            if not self.video.stereo and not torch.any(self.video.disps_sens):
                self.video.normalize()

        graph = FactorGraph(self.video, self.update_op, corr_impl="alt", max_factors=16*t, upsample=self.upsample)

        graph.add_proximity_factors(rad=self.backend_radius,
                                    nms=self.backend_nms,
                                    thresh=self.backend_thresh,
                                    beta=self.beta)

        graph.update_lowmem(steps=steps)
        graph.clear_edges()
        self.video.dirty[:t] = True


class DroidAsyncBackend:
    def __init__(self, net, video, args, max_age = 7):
        self.video = video
        self.update_op = net.update
        self.max_age = max_age

        # global optimization window
        self.t0 = 0
        self.t1 = 0

        self.upsample = args.upsample
        self.beta = args.beta
        self.backend_thresh = args.backend_thresh
        self.backend_radius = args.backend_radius
        self.backend_nms = args.backend_nms

        self.graph = FactorGraph(
            self.video,
            self.update_op,
            corr_impl="alt",
            max_factors=-1,
            upsample=self.upsample,
        )

    @torch.no_grad()
    def __call__(self, steps=12, normalize=True):
        """main update"""

        t = self.video.counter.value
        if normalize:
            if not self.video.stereo and not torch.any(self.video.disps_sens):
                self.video.normalize()

        self.graph.add_proximity_factors(
            rad=self.backend_radius,
            nms=self.backend_nms,
            thresh=self.backend_thresh,
            beta=self.beta,
        )

        self.graph.update_lowmem(steps=steps, use_inactive=True)
        self.graph.rm_factors(self.graph.age > self.max_age, store=True)

        self.video.dirty[:t] = True
