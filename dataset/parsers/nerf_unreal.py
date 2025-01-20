from pathlib import Path
import numpy as np

import dataset.utils.io as data_io
from dataset.utils.cameras import PinholeCamera

import numpy as np
import pandas as pd
import os, sys, time
import torch
import re
import torch.nn.functional as torch_F

import torchvision
import torchvision.transforms.functional as torchvision_F
import torch.multiprocessing as mp
import PIL
import tqdm
import threading, queue
from easydict import EasyDict as edict
from scipy.spatial.transform import Rotation
import copy

CAM_SENSOR_YAML = edict(
    sensor_type="camera",
    comment="VI-Sensor cam0 (MT9M034)",
    T_BS=edict(
        cols=4,
        rows=4,
        data=[1.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 1.0, 0.0,
              0.0, -1.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 1.0]
    ),
    rate_hz=30,
    resolution=[480, 768],
    camera_model="pinhole",
    intrinsics=[548.409, 548.409, 384, 240],
    distortion_model="equidistant",
    distortion_coefficients=[0.023584346301328347, -0.006764098468377487, 0.010259071387776937, -0.0037561745737771414]
)

instrinsic = CAM_SENSOR_YAML.intrinsics

K = [[instrinsic[0], 0., instrinsic[2]],
     [0., instrinsic[1], instrinsic[3]],
     [0, 0, 1]]
K = np.array(K, dtype=np.float64)

D = CAM_SENSOR_YAML.distortion_coefficients
D = np.array(D, dtype=np.float64)
readOut = int(6.944 * 1e4)  # ns

pose_transfer = np.array([1.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 1.0, 0.0,
                                 0.0, -1.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 1.0], dtype=np.float64)

pose_transfer = pose_transfer.reshape(4, 4)

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

        self.raw_H, self.raw_W = 480, 768
        self.path = "{}/{}".format(root, scene)
        self.img_path = self.path + "/mid"
        poses_raw, image_fnames, image_timeStamp = self.read_data_and_get_image_poses()
        self.list = list(zip(image_fnames, poses_raw, image_timeStamp))
        # manually split train/val subsets
        # num_val_split = int(len(self.list) * 0.3)
        # step = len(self.list) // num_val_split
        # val_data = self.list[:: step]
        # # self.list = self.list[:-num_val_split] if split == "train" else self.list[-num_val_split:]
        # self.list = [item for item in self.list if item not in val_data] if split == "train" else val_data
        # if subset: self.list = self.list[:subset]

    def parse_cameras_and_bounds(self):
        fname = "{}/poses_bounds.npy".format(self.path)
        data = torch.tensor(np.load(fname),dtype=torch.float32)
        # parse cameras (intrinsics and poses)
        cam_data = data[:,:-2].view([-1,3,5]) # [N,3,5]
        poses_raw = cam_data[...,:4] # [N,3,4]
        poses_raw[...,0],poses_raw[...,1] = poses_raw[...,1],-poses_raw[...,0]
        raw_H,raw_W,self.focal = cam_data[0,:,-1]
        assert(self.raw_H==raw_H and self.raw_W==raw_W)
        # parse depth bounds
        bounds = data[:,-2:] # [N,2]
        scale = 1./(bounds.min()*0.75) # not sure how this was determined
        poses_raw[...,3] *= scale
        bounds *= scale
        # roughly center camera poses
        poses_raw = self.center_camera_poses(poses_raw)
        return poses_raw,bounds

    def read_data_and_get_image_poses(self):
        # 读取cam0的图像数据
        cam0_path = self.img_path
        # 读取真实值的数据
        ground_truth_path = self.path + "/groundtruth.txt"


        image_fnames = []

        for filename in os.listdir(cam0_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_fnames.append(os.path.join(cam0_path, filename))

        def extract_number(filename):
            match = re.search(r'\d+', filename)
            if match:
                return int(match.group())
            else:
                return float('inf')  # If no number found, return infinity

        image_fnames = sorted(image_fnames, key=extract_number)
        image_poses = []
        times = []

        df = pd.read_csv(ground_truth_path, sep=' ', header=None, names=["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])

        # 每489行执行操作
        for index, row in df.iterrows():
            if index % 480 == 0:
                time = row['timestamp']
                twc = row.iloc[1:4]
                qwc = [row["qx"], row["qy"], row["qz"], row["qw"]]

                Twc = np.identity(4)
                Rwc = Rotation.from_quat(qwc)
                Twc[:3, :3] = Rwc.as_matrix()
                Twc[:3, 3] = twc * 2
                Twc = Twc @ pose_transfer

                Rwc = torch.tensor(Twc[:3, :3]).float()
                twc = torch.tensor(Twc[:3, 3]).float()
                pose = torch.cat([Rwc, twc[..., None]], dim=-1)  # [...,3,4]

                image_poses.append(pose)
                times.append(time)

        image_poses = image_poses
        image_fnames = image_fnames
        times = times

        poses_raw = self.center_camera_poses(torch.stack(image_poses, dim=0))

        # poses_raw, bounds = self.parse_cameras_and_bounds()

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
        sample.update(
            image=image,
            time=time,
            intr=intr,
            pose=pose)
        return sample

    def get_image(self, idx):
        image_fname = self.list[idx][0]
        image_time = self.list[idx][2]
        return image_fname, image_time


    def get_camera(self, idx):
        intr = torch.tensor([[instrinsic[0], 0., instrinsic[2]],
                             [0., instrinsic[1], instrinsic[3]],
                             [0, 0, 1]]).float()
        pose_raw = self.list[idx][1]
        return intr, pose_raw


def load_data(base_path: Path, scene: str, split: str, down_level=0, is_eval=False, is_rolling=True):
    # splits = 'train' if split == "trainval" else split
    if is_eval:
        splits = 'val'
    else:
        splits = 'train'

    # splits = 'train'
    dataset = Dataset(base_path, scene, splits)

    n_down = down_level
    cameras = [
        PinholeCamera(
            fx=instrinsic[0] / 2 ** n_down,
            fy=instrinsic[1] / 2 ** n_down,
            cx=instrinsic[2] / 2 ** n_down,
            cy=instrinsic[3] / 2 ** n_down,
            width=int(dataset.raw_W / 2 ** n_down),
            height=int(dataset.raw_H / 2 ** n_down),
            coord_type='opencv',
        )
    ]

    cam_num = len(cameras)

    assert cam_num == 1

    frames, poses = {k: [] for k in range(len(cameras))}, {k: [] for k in range(len(cameras))}

    for idx in range(len(dataset.list)):
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
                    'image_filename': Path(sample["image"].replace('rs', 'gs')),
                    'image_times': sample['time'],
                    'lossmult': 1.0,
                })
        poses[0].append(sample["pose"].numpy().astype(np.float32))

    aabb = np.array([-3, -3, -3, 3, 3, 3])

    outputs = {
        'frames': frames,
        'poses': poses,
        'cameras': cameras,
        'aabb': aabb,
        'read_out_time': readOut,
        'pose_transfer': pose_transfer
    }
    return outputs
