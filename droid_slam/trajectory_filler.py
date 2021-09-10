import cv2
import torch
import lietorch

from lietorch import SE3
from collections import OrderedDict
from factor_graph import FactorGraph
from droid_net import DroidNet
import geom.projective_ops as pops


class PoseTrajectoryFiller:
    """ This class is used to fill in non-keyframe poses """

    def __init__(self, net, video, device="cuda:0"):
        
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.count = 0
        self.video = video
        self.device = device

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        
    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image)

    def __fill(self, tstamps, images, intrinsics):
        """ fill operator """

        tt = torch.as_tensor(tstamps, device="cuda")
        images = torch.stack(images, 0)
        intrinsics = torch.stack(intrinsics, 0)
        inputs = images[:,:,[2,1,0]].to(self.device) / 255.0
        
        ### linear pose interpolation ###
        N = self.video.counter.value
        M = len(tstamps)

        ts = self.video.tstamp[:N]
        Ps = SE3(self.video.poses[:N])

        t0 = torch.as_tensor([ts[ts<=t].shape[0] - 1 for t in tstamps])
        t1 = torch.where(t0<N-1, t0+1, t0)

        dt = ts[t1] - ts[t0] + 1e-3
        dP = Ps[t1] * Ps[t0].inv()

        v = dP.log() / dt.unsqueeze(-1)
        w = v * (tt - ts[t0]).unsqueeze(-1)
        Gs = SE3.exp(w) * Ps[t0]

        # extract features (no need for context features)
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)
        fmap = self.__feature_encoder(inputs)

        self.video.counter.value += M
        self.video[N:N+M] = (tt, images[:,0], Gs.data, 1, None, intrinsics / 8.0, fmap)

        graph = FactorGraph(self.video, self.update)
        graph.add_factors(t0.cuda(), torch.arange(N, N+M).cuda())
        graph.add_factors(t1.cuda(), torch.arange(N, N+M).cuda())

        for itr in range(6):
            graph.update(N, N+M, motion_only=True)
    
        Gs = SE3(self.video.poses[N:N+M].clone())
        self.video.counter.value -= M

        return [ Gs ]

    @torch.no_grad()
    def __call__(self, image_stream):
        """ fill in poses of non-keyframe images """

        # store all camera poses
        pose_list = []

        tstamps = []
        images = []
        intrinsics = []
        
        for (tstamp, image, intrinsic) in image_stream:
            tstamps.append(tstamp)
            images.append(image)
            intrinsics.append(intrinsic)

            if len(tstamps) == 16:
                pose_list += self.__fill(tstamps, images, intrinsics)
                tstamps, images, intrinsics = [], [], []

        if len(tstamps) > 0:
            pose_list += self.__fill(tstamps, images, intrinsics)

        # stitch pose segments together
        return lietorch.cat(pose_list, 0)

