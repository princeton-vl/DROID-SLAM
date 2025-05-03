import sys
sys.path.append("droid_slam")

import torch
import argparse

import droid_backends
import argparse
import open3d as o3d

from visualization import create_camera_actor
from lietorch import SE3

from cuda_timer import CudaTimer

def view_reconstruction(filename: str, filter_thresh = 0.005, filter_count=2):
    reconstruction_blob = torch.load(filename)
    images = reconstruction_blob["images"].cuda()[...,::2,::2]
    disps = reconstruction_blob["disps"].cuda()[...,::2,::2]
    poses = reconstruction_blob["poses"].cuda()
    intrinsics = 4 * reconstruction_blob["intrinsics"].cuda()

    disps = disps.contiguous()

    index = torch.arange(len(images), device="cuda")
    thresh = filter_thresh * torch.ones_like(disps.mean(dim=[1,2]))

    with CudaTimer("iproj"):
        points = droid_backends.iproj(SE3(poses).inv().data, disps, intrinsics[0])
    colors = images[:,[2,1,0]].permute(0,2,3,1) / 255.0

    with CudaTimer("filter"):
        counts = droid_backends.depth_filter(poses, disps, intrinsics[0], index, thresh)

    mask = (counts >= filter_count) & (disps > .25 * disps.mean())
    points_np = points[mask].cpu().numpy()
    colors_np = colors[mask].cpu().numpy()

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_np)
    point_cloud.colors = o3d.utility.Vector3dVector(colors_np)

    vis = o3d.visualization.Visualizer()
    vis.create_window(height=960, width=960)
    vis.get_render_option().load_from_json("misc/renderoption.json")

    vis.add_geometry(point_cloud)

    # get pose matrices as a nx4x4 numpy array
    pose_mats = SE3(poses).inv().matrix().cpu().numpy()

    ### add camera actor ###
    for i in range(len(poses)):
        cam_actor = create_camera_actor(False)
        cam_actor.transform(pose_mats[i])
        vis.add_geometry(cam_actor)

    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="path to image directory")
    parser.add_argument("--filter_threshold", type=float, default=0.005)
    parser.add_argument("--filter_count", type=int, default=3)
    args = parser.parse_args()

    view_reconstruction(args.filename, args.filter_threshold, args.filter_count)