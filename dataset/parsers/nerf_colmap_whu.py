from pathlib import Path
import numpy as np

import dataset.utils.io as data_io
from dataset.utils.cameras import PinholeCamera

import numpy as np
import pandas as pd
import os, sys, time
import torch
import torch.nn.functional as torch_F
from dataset.utils import poses
import torchvision
import torchvision.transforms.functional as torchvision_F
import torch.multiprocessing as mp
import PIL
import tqdm
import threading, queue
from easydict import EasyDict as edict
from scipy.spatial.transform import Rotation
from dataset.parsers.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, read_extrinsics_text, \
    read_intrinsics_text, qvec2rotmat
import copy
import ipdb

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
    resolution=[480, 640],
    camera_model="pinhole",
    intrinsics=[320.0, 320.0, 319.5, 239.5]
)

instrinsic = CAM_SENSOR_YAML.intrinsics

K = [[instrinsic[0], 0., instrinsic[2]],
     [0., instrinsic[1], instrinsic[3]],
     [0, 0, 1]]
K = np.array(K, dtype=np.float64)

# D = CAM_SENSOR_YAML.distortion_coefficients
# D = np.array(D, dtype=np.float64)
readOut = 1

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

    def __init__(self, root, scene, split="train", use_colmap_pose=False, subset=None):

        self.use_colmap_pose = use_colmap_pose
        self.raw_H, self.raw_W = 480, 640
        self.path = "{}/{}".format(root, scene)
        poses_raw, image_fnames, image_timeStamp = self.read_data_and_get_image_poses_colmap()
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
        data = torch.tensor(np.load(fname), dtype=torch.float32)
        # parse cameras (intrinsics and poses)
        cam_data = data[:, :-2].view([-1, 3, 5])  # [N,3,5]
        poses_raw = cam_data[..., :4]  # [N,3,4]
        poses_raw[..., 0], poses_raw[..., 1] = poses_raw[..., 1], -poses_raw[..., 0]
        raw_H, raw_W, self.focal = cam_data[0, :, -1]
        assert (self.raw_H == raw_H and self.raw_W == raw_W)
        # parse depth bounds
        bounds = data[:, -2:]  # [N,2]
        scale = 1. / (bounds.min() * 0.75)  # not sure how this was determined
        poses_raw[..., 3] *= scale
        bounds *= scale
        # roughly center camera poses
        poses_raw = self.center_camera_poses(poses_raw)
        return poses_raw, bounds

    def read_data_and_get_image_poses_colmap(self, ):
        # 读取cam0的图像数据
        image_folder = os.path.join(self.path, "rs_images")
        ground_truth_path = os.path.join(self.path, "tum_ground_truth_gs.txt")
        df_cam0 = pd.read_csv(os.path.join(self.path, "data.csv"))

        try:
            cameras_extrinsic_file = os.path.join(self.path, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(self.path, "sparse/0", "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)

        except:
            cameras_extrinsic_file = os.path.join(self.path, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(self.path, "sparse/0", "cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)

        colmap_poses = []
        colmap_path = []
        colmap_name = []

        Tic = CAM_SENSOR_YAML.T_BS.data
        Tic = np.array(Tic, dtype=np.float64)
        Tic = Tic.reshape(4, 4)

        for idx, key in enumerate(cam_extrinsics):
            extr = cam_extrinsics[key]
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)
            Rwc = torch.tensor(R).float()
            twc = torch.tensor(T).float() * 0.14
            # test
            Twi = np.identity(4)
            Twi[:3, :3] = R
            Twi[:3, 3] = T * 0.14
            Twc = Twi @ np.array([-1.0, 0.0, 0.0, 0.0,
                                  0.0, 1.0, 0.0, 0.0,
                                  0.0, 0.0, -1.0, 0.0,
                                  0.0, 0.0, 0.0, 1.0], dtype=np.float64).reshape(4, 4)
            Rwc = torch.tensor(Twc[:3, :3]).float()
            twc = torch.tensor(Twc[:3, 3]).float()

            pose = torch.cat([Rwc, twc[..., None]], dim=-1)  # [...,3,4]
            colmap_poses.append(pose)

            image_path = os.path.join(image_folder, extr.name)
            image_name = os.path.basename(image_path)
            colmap_path.append(image_path)
            colmap_name.append(image_name)

        df_groundtruth = pd.read_csv(ground_truth_path, skiprows=3, delim_whitespace=True,
                                     names=["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])

        gt_poses = []
        gt_names = []
        gt_times = []

        for index, row in df_cam0.iloc[1:].iterrows():
            time = row['#timestamp [ns]']
            image_filename = row['filename']
            gt_names.append(image_filename)
            # 查找与cv1文件中时间最接近的时间
            closest_time_row = df_groundtruth.iloc[(df_groundtruth['timestamp'] * 1e9 - time).abs().idxmin()]
            twi = closest_time_row.iloc[1:4]
            qwi = [closest_time_row["qx"], closest_time_row["qy"], closest_time_row["qz"],
                   closest_time_row["qw"]]
            Twi = np.identity(4)
            Rwi = Rotation.from_quat(qwi)
            Twi[:3, :3] = Rwi.as_matrix()
            Twi[:3, 3] = twi
            Twc = Twi @ Tic
            Twc = Twc @ pose_transfer
            Rwc = torch.tensor(Twc[:3, :3]).float()
            twc = torch.tensor(Twc[:3, 3]).float()
            pose = torch.cat([Rwc, twc[..., None]], dim=-1)  # [...,3,4]
            gt_poses.append(pose)
            gt_times.append(time)

        indices = [index for index, name in enumerate(gt_names) if name in colmap_name]
        colmap_gt_poses = [gt_poses[index] for index in indices]
        colmap_gt_names = colmap_path
        colmap_gt_times = [gt_times[index] for index in indices]

        if self.use_colmap_pose:
            poses_raw = self.center_camera_poses(torch.stack(colmap_poses, dim=0))
        else:
            poses_raw = self.center_camera_poses(torch.stack(colmap_gt_poses, dim=0))

        return poses_raw, colmap_gt_names, colmap_gt_times

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
        image, time, intr, pose = self.get_image_info(idx)
        sample.update(
            image=image,
            time=time,
            intr=intr,
            pose=pose)
        return sample

    def get_image_info(self, idx):
        image_fname = self.list[idx][0]
        image_time = self.list[idx][2]

        image_intr = torch.tensor([[instrinsic[0], 0., instrinsic[2]],
                                   [0., instrinsic[1], instrinsic[3]],
                                   [0, 0, 1]]).float()

        image_pose = self.list[idx][1]

        return image_fname, image_time, image_intr, image_pose


def load_data(base_path: Path, scene: str, split: str, down_level=0, is_eval=False, is_rolling=True,
              use_colmap_pose=False):
    # splits = 'train' if split == "trainval" else split
    if is_eval:
        splits = 'val'
    else:
        splits = 'train'

    dataset = Dataset(base_path, scene, splits, use_colmap_pose)

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
                    # 'image_filename': Path(sample["image"].replace('rs_images', 'diff').replace('.png', 'gs.png')),
                    'image_filename': Path(sample["image"]),
                    'image_times': sample['time'],
                    'lossmult': 1.0,
                })
        else:
            frames[0].append(
                {
                    'image_filename': Path(sample["image"].replace('rs_images', 'gs_images')),
                    'image_times': sample['time'],
                    'lossmult': 1.0,
                })
        poses[0].append(sample["pose"].numpy().astype(np.float32))

    aabb = np.array([-4, -4, -4, 4, 4, 4])
    outputs = {
        'frames': frames,
        'poses': poses,
        'cameras': cameras,
        'aabb': aabb,
        'read_out_time': readOut,
        "period": 400,
        'pose_transfer': pose_transfer
    }
    return outputs
