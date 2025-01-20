import torch
from torch import nn
import argparse
import os
import cv2
import ipdb
class DepthMetrics(nn.Module):
    """Computation of error metrics between predicted and ground truth depths

    from:
        https://arxiv.org/abs/1806.01260

    Returns:
        abs_rel: normalized avg absolute realtive error
        sqrt_rel: normalized square-root of absolute error
        rmse: root mean square error
        rmse_log: root mean square error in log space
        a1, a2, a3: metrics
    """

    def __init__(self, tolerance: float = 0.1, **kwargs):
        self.tolerance = tolerance
        super().__init__()

    @torch.no_grad()
    def forward(self, pred, gt):
        mask = gt > self.tolerance

        thresh = torch.max((gt[mask] / pred[mask]), (pred[mask] / gt[mask]))
        a1 = (thresh < 1.25).float().mean()
        a2 = (thresh < 1.25**2).float().mean()
        a3 = (thresh < 1.25**3).float().mean()
        rmse = (gt[mask] - pred[mask]) ** 2
        rmse = torch.sqrt(rmse.mean())

        rmse_log = (torch.log(gt[mask]) - torch.log(pred[mask])) ** 2
        # rmse_log[rmse_log == float("inf")] = float("nan")
        rmse_log = torch.sqrt(rmse_log).nanmean()

        abs_rel = torch.abs(gt - pred)[mask] / gt[mask]
        abs_rel = abs_rel.mean()
        sq_rel = (gt - pred)[mask] ** 2 / gt[mask]
        sq_rel = sq_rel.mean()

        return (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--est", type=str, help="est depth path")
    parser.add_argument("--gt", type=str, help="gt depth path")
    args = parser.parse_args()
    depth_metrics = DepthMetrics(tolerance=0.1)

    depth_names = os.listdir(args.gt)
    depth_names = sorted(depth_names)

    batch_est_depth = []
    batch_gt_depth= []
    for i in range(len(depth_names)):
        # Read depth image and camera pose
        depth_gt_path = os.path.join(args.gt, depth_names[i])
        est_name = pose_name = depth_names[i].replace("depth_", "")
        depth_est_path = os.path.join(args.est, est_name)

        depth_im_gt = cv2.imread(depth_gt_path, -1).astype(float)

        depth_im_est = cv2.imread(depth_est_path, -1).astype(float)

        depth_im_est /= 10000.
        depth_im_gt /= 5000.

        est = torch.tensor(depth_im_est).unsqueeze(0)
        gt = torch.tensor(depth_im_gt).unsqueeze(0)

        batch_gt_depth.append(gt)
        batch_est_depth.append(est)


    batch_est_depth = torch.stack(batch_est_depth, dim=0)
    batch_gt_depth = torch.stack(batch_gt_depth, dim=0)

    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = depth_metrics(batch_gt_depth, batch_est_depth)
    print(abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3)