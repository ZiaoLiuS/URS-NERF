from pathlib import Path
import numpy as np

import dataset.utils.io as data_io
from dataset.utils.cameras import PinholeCamera

import numpy as np
import pandas as pd
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
from scipy.spatial.transform import Rotation
from dataset.utils import poses
import copy
import cv2

CAM_SENSOR_YAML1 = edict(
    sensor_type="camera",
    comment="Xiaomi Mi 8",
    rate_hz=20,
    resolution=[640, 448],
    camera_model="pinhole",
    intrinsics=[320, 320, 320, 224],
    distortion_model="radtan",
    distortion_coefficients=[0, 0, 0, 0]
)

readOut = float(0.11160714 * 1e-3)
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

        self.raw_H, self.raw_W = 448, 640
        self.path = "{}/{}".format(root, scene)

        self.instrinsic = CAM_SENSOR_YAML1.intrinsics
        self.DIM = CAM_SENSOR_YAML1.resolution

        self.K = [[self.instrinsic[0], 0., self.instrinsic[2]],
                  [0., self.instrinsic[1], self.instrinsic[3]],
                  [0, 0, 1]]

        self.K = np.array(self.K, dtype=np.float64)

        poses_raw, image_fnames, image_timeStamp = self.read_data_and_get_image_poses()
        self.list = list(zip(image_fnames, poses_raw, image_timeStamp))
        # manually split train/val subsets
        # num_val_split = int(len(self.list) * 0.1)
        # num_test_split = 0
        # step = len(self.list) // num_val_split
        # val_data = []
        # # self.list = self.list[:-num_val_split] if split == "train" else self.list[-num_val_split:]
        # self.list = [item for item in self.list if item not in val_data] if split == "train" else val_data

        # if subset: self.list = self.list[:subset]

    def read_data_and_get_image_poses(self):
        # 读取cam0的图像数据
        cam0_path = self.path + '/gt_vel.log'
        with open(cam0_path, 'r') as file:
            # 读取文件内容
            for line in file:
                if line[0] == "#":
                    continue
                vel = np.fromstring(line, dtype=float, sep=' ')
                vel[-3:] = vel[-3:]

        image_fnames = []
        for filename in os.listdir(self.path):
            # 检查文件名是否包含特定字符
            if "_rs.png" in filename:
                # 如果包含特定字符，打印文件名或执行其他操作
                image_fnames.append(filename)

        image_fnames = sorted(image_fnames)

        image_poses = []
        times = []
        se3 = self.raw_H * readOut * vel
        Pi = poses.lie.se3_to_SE3(torch.tensor(se3)).float()
        p0 = poses.lie.se3_to_SE3(torch.tensor(se3)).float()
        for i in range(len(image_fnames)):
            # se3 = self.raw_H * readOut * vel * i
            image_poses.append(Pi)
            times.append(self.raw_H * readOut * i)
            Pi = poses.pose.compose_pair(Pi, p0)

        poses_raw = self.center_camera_poses(torch.stack(image_poses, dim=0))
        # 初始化内插器： 中心化的pose， 时间（ns）
        return poses_raw, image_fnames, times

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

    def __getitem__(self, idx):
        sample = dict(idx=idx)
        image, time = self.get_image(idx)
        intr, pose = self.get_camera(idx)
        # poses = self.get_intr(idx)
        sample.update(
            image=image,
            time=time,
            intr=intr,
            pose=pose)
        return sample

    def get_image(self, idx):
        image_fname = "{}/{}".format(self.path, self.list[idx][0])
        print(image_fname)
        image_time = self.list[idx][2]
        return image_fname, image_time

    def get_camera(self, idx):
        intr = torch.tensor(self.K).float()
        pose_raw = self.list[idx][1]
        return intr, pose_raw


def load_data(base_path: Path, scene: str, split: str, down_level=0, is_eval=False, is_rolling=True):
    # splits = 'train' if split == "trainval" else split
    if is_eval:
        splits = 'val'
    else:
        splits = 'train'
    dataset = Dataset(base_path, scene, splits)

    n_down = down_level
    cameras = [
        PinholeCamera(
            fx=dataset.K[0, 0] / 2 ** n_down,
            fy=dataset.K[1, 1] / 2 ** n_down,
            cx=dataset.K[0, 2] / 2 ** n_down,
            cy=dataset.K[1, 2] / 2 ** n_down,
            width=int(dataset.raw_W / 2 ** n_down),
            height=int(dataset.raw_H / 2 ** n_down),
            coord_type='opencv'
        )
    ]

    cam_num = len(cameras)

    assert cam_num == 1

    frames, poses = ({k: [] for k in range(len(cameras))},
                     {k: [] for k in range(len(cameras))})

    for idx in range(len(dataset.list)):
        # if not is_rolling:
        #     if idx < 2 or idx > len(dataset.list)-2:
        #         continue

        sample = dataset[idx]
        if is_rolling:
            frames[0].append(
                {
                    'image_filename': Path(sample["image"]),
                    'image_times': sample['time'],
                    'lossmult': 1.0,
                })
        else:
            frames[0].append(
                {
                    'image_filename': Path(sample["image"].replace('_rs', '_gs_f')),
                    'image_times': sample['time'],
                    'lossmult': 1.0,
                })
        poses[0].append(sample["pose"].numpy().astype(np.float32))

    aabb = np.array([-20, -20, -20, 20, 20, 20])

    outputs = {
        'frames': frames,
        'poses': poses,
        'cameras': cameras,
        'aabb': aabb,
        'read_out_time': readOut,
        'pose_transfer': pose_transfer
    }

    return outputs
