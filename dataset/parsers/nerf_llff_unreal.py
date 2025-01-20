from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import dataset.utils.io as data_io
from dataset.utils.cameras import PinholeCamera

import numpy as np
import os, sys, time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import torch.multiprocessing as mp
import PIL
import tqdm
import threading, queue
from easydict import EasyDict as edict
import copy

readOut = 3.7  # ns

pose_transfer = np.diag([1, 1, 1, 1])


def invert(pose, use_inverse=False):
    # invert a camera pose
    R, t = pose[..., :3], pose[..., 3:]
    R_inv = R.inverse() if use_inverse else R.transpose(-1, -2)
    t_inv = (-R_inv @ t)[..., 0]

    R_inv = R_inv.float()
    t_inv = t_inv.float()
    pose_inv = torch.cat([R_inv, t_inv[..., None]], dim=-1)  # [...,3,4]

    return pose_inv


def compose_pair(pose_a, pose_b):
    # pose_new(x) = pose_b o pose_a(x)
    R_a, t_a = pose_a[..., :3], pose_a[..., 3:]
    R_b, t_b = pose_b[..., :3], pose_b[..., 3:]
    R_new = R_b @ R_a
    t_new = (R_b @ t_a + t_b)[..., 0]

    R_new = R_new.float()
    t_new = t_new.float()
    pose_new = torch.cat([R_new, t_new[..., None]], dim=-1)  # [...,3,4]

    return pose_new


class Dataset():

    def __init__(self, root, scene, split="train", subset=None):
        self.raw_H, self.raw_W = 2160, 3840
        self.path = "{}/{}".format(root, scene)
        self.path_image = "{}/images".format(self.path)
        image_fnames = sorted(os.listdir(self.path_image))
        poses_raw, bounds = self.parse_cameras_and_bounds()
        self.list = list(zip(image_fnames, poses_raw, bounds))
        self.list = self.list[2:7]
        # np.random.seed(42)
        # np.random.shuffle(self.list)
        # self.list = self.list[:25]
        # manually split train/val subsets
        # num_val_split = int(len(self.list) * 0.2)
        # self.list = self.list[:-num_val_split] if split=="train" else self.list[-num_val_split:]
        # if subset: self.list = self.list[:subset]

    def parse_cameras_and_bounds(self):
        fname = "{}/poses_bounds.npy".format(self.path)
        data = torch.tensor(np.load(fname), dtype=torch.float32)
        # fname1 = "{}/groundtruth.txt".format(self.path)
        #
        # df_groundtruth = pd.read_csv(fname1, skiprows=3, delim_whitespace=True,
        #                              names=["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
        # image_poses = []
        # for index, row in df_groundtruth.iloc[1::480].iterrows():
        #     # 查找与cv1文件中时间最接近的时间
        #     closest_time_row = row
        #     # print(((df_groundtruth['timestamp'] - time).abs().idxmin()))
        #     # 获取坐标和相机姿态信息
        #     twi = closest_time_row.iloc[1:4]
        #     qwi = [closest_time_row["qx"], closest_time_row["qy"], closest_time_row["qz"],
        #            closest_time_row["qw"]]
        #
        #     Twi = np.identity(4)
        #     Rwi = Rotation.from_quat(qwi)
        #     Twi[:3, :3] = Rwi.as_matrix()
        #     Twi[:3, 3] = twi
        #     Twc = Twi @ np.array([1.0, 0.0, 0.0, 0.0,
        #       0.0, 0.0, 1.0, 0.0,
        #       0.0, -1.0, 0.0, 0.0,
        #       0.0, 0.0, 0.0, 1.0], dtype=np.float64).reshape(4, 4)
        #
        #     Rwc = torch.tensor(Twc[:3, :3]).float()
        #     twc = torch.tensor(Twc[:3, 3]).float()
        #     pose = torch.cat([Rwc, twc[..., None]], dim=-1)  # [...,3,4]
        #
        #     image_poses.append(pose)
        # parse cameras (intrinsics and poses)
        cam_data = data[:, :-2].view([-1, 3, 5])  # [N,3,5]
        poses_raw = cam_data[..., :4]  # [N,3,4]
        poses_raw[..., 0], poses_raw[..., 1] = poses_raw[..., 1], -poses_raw[..., 0]
        raw_H, raw_W, self.focal = cam_data[0, :, -1]
        assert (self.raw_H == raw_H and self.raw_W == raw_W)
        # parse depth bounds
        bounds = data[:, -2:]  # [N,2]
        scale = 1. / (bounds.min() * 0.75)  # not sure how this was determined
        poses_raw[..., 3] *= scale * 0.4
        bounds *= scale
        # poses_raw[..., 3] = torch.zeros_like(poses_raw[..., 3])
        # roughly center camera poses
        poses_raw = self.center_camera_poses(poses_raw)
        return poses_raw, bounds

    def center_camera_poses(self, poses):
        # compute average pose
        center = poses[..., 3].mean(dim=0)
        v1 = torch_F.normalize(poses[..., 1].mean(dim=0), dim=0)
        v2 = torch_F.normalize(poses[..., 2].mean(dim=0), dim=0)
        v0 = v1.cross(v2)
        pose_avg = torch.stack([v0, v1, v2, center], dim=-1)[None]  # [1,3,4]
        # apply inverse of averaged pose
        pose_avg_inv = invert(pose_avg)
        poses = compose_pair(poses, pose_avg_inv)
        return poses

    def get_all_camera_poses(self):
        pose_raw_all = [tup[1] for tup in self.list]
        pose_all = torch.stack([self.parse_raw_camera(p) for p in pose_raw_all], dim=0)
        return pose_all

    def __getitem__(self, idx):
        sample = dict(idx=idx)
        image = self.get_image(idx)
        intr, pose = self.get_camera(idx)
        sample.update(
            image=image,
            intr=intr,
            pose=pose,
        )
        return sample

    def get_image(self, idx):
        image_fname = "{}/{}".format(self.path_image, self.list[idx][0])
        return image_fname

    def get_camera(self, idx):
        intr = torch.tensor([[self.focal, 0, self.raw_W / 2],
                             [0, self.focal, self.raw_H / 2],
                             [0, 0, 1]]).float()
        pose_raw = self.list[idx][1]
        pose = self.parse_raw_camera(pose_raw)
        return intr, pose

    def parse_raw_camera(self, pose_raw):
        R = torch.diag(torch.tensor([1, 1, 1])).float()
        t = torch.zeros(R.shape[:-1], device=R.device).float()
        pose_flip = torch.cat([R, t[..., None]], dim=-1)  # [...,3,4]
        pose_flip_inv = invert(pose_flip)
        pose = compose_pair(pose_flip, pose_raw)

        return pose


def load_data(base_path: Path, scene: str, split: str, down_level=0, is_eval=False, is_rolling=True):
    splits = 'train' if split == "trainval" else split
    dataset = Dataset(base_path, scene, splits)
    n_down = down_level

    # cameras = [
    #     PinholeCamera(
    #         fx=548.409 / (2 ** n_down),
    #         fy=548.409 / (2 ** n_down),
    #         cx=dataset.raw_W / 2 / (2 ** n_down),
    #         cy=dataset.raw_H / 2 / (2 ** n_down),
    #         width=dataset.raw_W // (2 ** n_down),
    #         height=dataset.raw_H // (2 ** n_down),
    #         coord_type='opengl'
    #     )
    #
    # ]
    cameras = [
        PinholeCamera(
            fx=dataset.focal / (2 ** n_down),
            fy=dataset.focal / (2 ** n_down),
            cx=dataset.raw_W / 2 // (2 ** n_down),
            cy=dataset.raw_H / 2 // (2 ** n_down),
            width=dataset.raw_W // (2 ** n_down),
            height=dataset.raw_H // (2 ** n_down),
            coord_type='opencv'
        )
    ]
    cam_num = len(cameras)

    assert cam_num == 1

    frames, poses = {k: [] for k in range(len(cameras))}, {
        k: [] for k in range(len(cameras))
    }

    for idx in range(len(dataset.list)):
        sample = dataset[idx]
        print(sample["image"])
        if is_rolling:
            if idx % 2 != 0:
                frames[0].append(
                    {
                        'image_filename': Path(sample["image"]),
                        'image_times': 1.0,
                        'lossmult': 1.0,
                    })
            else:
                frames[0].append(
                    {
                        'image_filename': Path(sample["image"]),
                        'image_times': 1.0,
                        'lossmult': 1.0,
                    })
        else:
            frames[0].append(
                {
                    'image_filename': Path(
                        sample["image"].replace('whu_traj1/slow_colmap/images', 'WHU/trajectory_1/slow/gs/clean/img')),
                    'image_times': 1.0,
                    'lossmult': 1.0,
                })
        poses[0].append(sample["pose"].numpy().astype(np.float32))

    # aabb = np.array([- 3.5, - 3.5, - 3.5,  3.5,  3.5, 3.5])
    aabb = np.array([-7, -7, -7, 7, 7, 7])
    print(poses[0])
    outputs = {
        'frames': frames,
        'poses': poses,
        'cameras': cameras,
        'aabb': aabb,
        'read_out_time': readOut,
        "period": 1000000/30 ** n_down,
        'pose_transfer': pose_transfer
    }
    return outputs
