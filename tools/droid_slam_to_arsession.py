import numpy as np
import torch
import torch.nn.functional as F
from os import listdir
import os.path as osp
import json
from lietorch import SE3
import argparse
import plyfile
import open3d as o3d
from typing import Union


def get_center(pts: torch.Tensor) -> torch.Tensor:
    center = pts.mean(0)
    dis = (pts - center[None, :]).norm(p=2, dim=-1)
    mean, std = dis.mean(), dis.std()
    q25, q75 = torch.quantile(dis, 0.25), torch.quantile(dis, 0.75)
    valid = (
            (dis > mean - 1.5 * std)
            & (dis < mean + 1.5 * std)
            & (dis > mean - (q75 - q25) * 1.5)
            & (dis < mean + (q75 - q25) * 1.5)
    )
    center = pts[valid].mean(0)
    return center


def normalize_poses(poses: torch.Tensor,
                    pts: torch.Tensor,
                    up_est_method: str,
                    center_est_method: str,
                    pts3d_normal: Union[torch.Tensor, None] = None):
    if center_est_method == "camera":
        # estimation scene center as the average of all camera positions
        center = poses[..., 3].mean(0)
    elif center_est_method == "lookat":
        # estimation scene center as the average of the intersection of selected pairs of camera rays
        cams_ori = poses[..., 3]
        cams_dir = poses[:, :3, :3] @ torch.as_tensor([0.0, 0.0, -1.0])
        cams_dir = F.normalize(cams_dir, dim=-1)
        A = torch.stack([cams_dir, -cams_dir.roll(1, 0)], dim=-1)
        b = -cams_ori + cams_ori.roll(1, 0)
        t = torch.linalg.lstsq(A, b).solution
        center = (
                torch.stack([cams_dir, cams_dir.roll(1, 0)], dim=-1) * t[:, None, :]
                + torch.stack([cams_ori, cams_ori.roll(1, 0)], dim=-1)
        ).mean((0, 2))
    elif center_est_method == "point":
        # first estimation scene center as the average of all camera positions
        # later we'll use the center of all points bounded by the cameras as the final scene center
        center = poses[..., 3].mean(0)
    else:
        raise NotImplementedError(
            f"Unknown center estimation method: {center_est_method}"
        )

    if up_est_method == "ground":
        # estimate up direction as the normal of the estimated ground plane
        # use RANSAC to estimate the ground plane in the point cloud
        import pyransac3d as pyrsc

        ground = pyrsc.Plane()
        plane_eq, inliers = ground.fit(
            pts.numpy(), thresh=0.01
        )  # TODO: determine thresh based on scene scale
        plane_eq = torch.as_tensor(plane_eq)  # A, B, C, D in Ax + By + Cz + D = 0
        z = F.normalize(plane_eq[:3], dim=-1)  # plane normal as up direction
        signed_distance = (
                torch.cat([pts, torch.ones_like(pts[..., 0:1])], dim=-1) * plane_eq
        ).sum(-1)
        if signed_distance.mean() < 0:
            z = -z  # flip the direction if points lie under the plane
    elif up_est_method == "camera":
        # estimate up direction as the average of all camera up directions
        z = F.normalize((poses[..., 3] - center).mean(0), dim=0)
    else:
        raise NotImplementedError(f"Unknown up estimation method: {up_est_method}")

    # new axis
    y_ = torch.as_tensor([z[1], -z[0], 0.0])
    x = F.normalize(y_.cross(z), dim=0)
    y = z.cross(x)

    if center_est_method == "point":
        # rotation
        Rc = torch.stack([x, y, z], dim=1)
        R = Rc.T
        poses_homo = torch.cat(
            [
                poses,
                torch.as_tensor([[[0.0, 0.0, 0.0, 1.0]]]).expand(
                    poses.shape[0], -1, -1
                ),
            ],
            dim=1,
        )
        inv_trans = torch.cat(
            [
                torch.cat([R, torch.as_tensor([[0.0, 0.0, 0.0]]).T], dim=1),
                torch.as_tensor([[0.0, 0.0, 0.0, 1.0]]),
            ],
            dim=0,
        )
        poses_norm = (inv_trans @ poses_homo)[:, :3]
        pts = (
                      inv_trans
                      @ torch.cat([pts, torch.ones_like(pts[:, 0:1])], dim=-1)[..., None]
              )[:, :3, 0]

        # translation and scaling
        poses_min, poses_max = (
            poses_norm[..., 3].min(0)[0],
            poses_norm[..., 3].max(0)[0],
        )
        pts_fg = pts[
            (poses_min[0] < pts[:, 0])
            & (pts[:, 0] < poses_max[0])
            & (poses_min[1] < pts[:, 1])
            & (pts[:, 1] < poses_max[1])
            ]
        center = get_center(pts_fg)
        tc = center.reshape(3, 1)
        t = -tc
        poses_homo = torch.cat(
            [
                poses_norm,
                torch.as_tensor([[[0.0, 0.0, 0.0, 1.0]]]).expand(
                    poses_norm.shape[0], -1, -1
                ),
            ],
            dim=1,
        )
        inv_trans = torch.cat(
            [
                torch.cat([torch.eye(3), t], dim=1),
                torch.as_tensor([[0.0, 0.0, 0.0, 1.0]]),
            ],
            dim=0,
        )
        poses_norm = (inv_trans @ poses_homo)[:, :3]
        scale = poses_norm[..., 3].norm(p=2, dim=-1).min()
        poses_norm[..., 3] /= scale
        pts = (
                      inv_trans
                      @ torch.cat([pts, torch.ones_like(pts[:, 0:1])], dim=-1)[..., None]
              )[:, :3, 0]
        # apply the rotation to the point cloud normal
        if pts3d_normal is not None:
            pts3d_normal = (R @ pts3d_normal.T).T
        pts = pts / scale
    else:
        # rotation and translation
        Rc = torch.stack([x, y, z], dim=1)
        tc = center.reshape(3, 1)
        R, t = Rc.T, -Rc.T @ tc
        poses_homo = torch.cat(
            [
                poses,
                torch.as_tensor([[[0.0, 0.0, 0.0, 1.0]]]).expand(
                    poses.shape[0], -1, -1
                ),
            ],
            dim=1,
        )
        inv_trans = torch.cat(
            [torch.cat([R, t], dim=1), torch.as_tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0
        )
        poses_norm = (inv_trans @ poses_homo)[:, :3]  # (N_images, 4, 4)

        # scaling
        scale = poses_norm[..., 3].norm(p=2, dim=-1).min()
        poses_norm[..., 3] /= scale

        # apply the transformation to the point cloud
        pts = (
                      inv_trans
                      @ torch.cat([pts, torch.ones_like(pts[:, 0:1])], dim=-1)[..., None]
              )[:, :3, 0]
        # apply the rotation to the point cloud normal
        if pts3d_normal is not None:
            pts3d_normal = (R @ pts3d_normal.T).T
        pts = pts / scale

    return poses_norm, pts, pts3d_normal


def main(args):
    root_dir = args.root_dir
    recon_dir = args.recon_dir
    space = json.load(open(osp.join(root_dir, "space.json"), "r"))
    frames = []

    # Read poses from poses.npy
    poses = torch.from_numpy(np.load(osp.join(recon_dir, "poses.npy")))
    Ps = SE3(poses).inv().matrix()
    dig_mat = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float32, device=Ps.device))[None]
    all_c2w = Ps @ dig_mat

    # Read timestamps from tstamps.npy
    Ts = np.load(osp.join(recon_dir, "tstamps.npy")).astype(np.int32)
    imgpaths = sorted(listdir(osp.join(root_dir, "images")))
    imgpaths = [f"images/{imgpaths[idx]}" for idx in Ts]

    assert len(imgpaths) == len(poses), "number of images and poses do not match"

    # Read point cloud from point_cloud.ply
    pcd_path = osp.join(recon_dir, "point_cloud.ply")
    plydata = plyfile.PlyData.read(pcd_path)
    vertex = plydata["vertex"]
    pts3d = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T
    if not all(key in vertex for key in ("red", "green", "blue")):
        rgbs = np.ones((pts3d.shape[0], 3))
    else:
        rgbs = np.vstack([vertex["red"], vertex["green"], vertex["blue"]]).T
    pts3d_normal = np.vstack([vertex["nx"], vertex["ny"], vertex["nz"]]).T
    pts3d = torch.from_numpy(pts3d).float()
    pts3d_normal = torch.from_numpy(pts3d_normal).float()

    # Normalize poses and point_cloud
    all_c2w = all_c2w[..., :3, :]
    all_c2w, pts3d, pts3d_normal = normalize_poses(
        all_c2w,
        pts3d,
        up_est_method="ground",
        center_est_method="camera",
        pts3d_normal=pts3d_normal,
    )
    Ps[..., :3, :] = all_c2w[:]

    # Save transformed point_cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pts3d.cpu().numpy())
    point_cloud.normals = o3d.utility.Vector3dVector(pts3d_normal.cpu().numpy())
    rgbs = rgbs / 255.0
    point_cloud.colors = o3d.utility.Vector3dVector(rgbs)
    pcd_save_path = osp.join(root_dir, "point_cloud.ply")
    o3d.io.write_point_cloud(pcd_save_path, point_cloud)
    print(f"point_cloud.ply saved to {pcd_save_path}")

    # Save poses to space.json
    for idx in range(len(imgpaths)):
        imgpath = imgpaths[idx]
        pose = Ps[idx]
        frame = {"file_path": imgpath, "transform_matrix": pose.tolist()}
        frames.append(frame)

    space["frames"] = frames
    with open(osp.join(root_dir, "space.json"), "w") as f:
        json.dump(space, f)
    print(f"space.json saved to {recon_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--recon_dir', type=str, required=True)
    args = parser.parse_args()

    main(args)
