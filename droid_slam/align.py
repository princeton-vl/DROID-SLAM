from lietorch import SE3

def align_pose_fragements(pose0, pose1):
    P0 = SE3(pose0.clone())
    P1 = SE3(pose1.clone())

    dP1 = P0[None, :].inv() * P0[:, None]
    dP2 = P1[None, :].inv() * P1[:, None]

    dt1 = dP1.matrix()[:, :, :3, 3].view(-1, 3)
    dt2 = dP2.matrix()[:, :, :3, 3].view(-1, 3)

    s = (dt1 * dt2).sum() / (dt1 * dt1).sum()

    P0.data[..., :3] *= s

    dP = P1 * P0.inv()
    dG = dP[[0]]

    for _ in range(3):
        e = (P1 * (dG * P0).inv()).log()
        dG = SE3.exp(e.mean(dim=0, keepdim=True)) * dG

    return dG, s
