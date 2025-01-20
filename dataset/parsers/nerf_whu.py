import math
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
    resolution=[480, 640],
    camera_model="pinhole",
    intrinsics=[320.0, 320.0, 319.5, 239.5],
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


# 生成视角绕Z轴呈15°角度的圆锥旋转的姿势
def generate_poses(num_poses=100, cone_angle_degrees=15):
    image_poses = []

    # 将角度转换为弧度
    cone_angle = math.radians(cone_angle_degrees)

    for i in range(num_poses):
        # 计算旋转角度
        angle = 2 * np.pi * i / num_poses

        # 生成旋转矩阵，沿着与Z轴呈15°角度的圆锥旋转
        rotation_matrix = torch.tensor([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ], dtype=torch.float32)

        # 生成圆锥旋转矩阵
        cone_rotation_matrix = torch.tensor([
            [np.cos(cone_angle), -np.sin(cone_angle), 0],
            [np.sin(cone_angle), np.cos(cone_angle), 0],
            [0, 0, 1]
        ], dtype=torch.float32)

        # 合并旋转矩阵和圆锥旋转矩阵
        pose_matrix = torch.matmul(rotation_matrix, cone_rotation_matrix)

        # 创建3x4的姿势矩阵
        pose_matrix = torch.cat([pose_matrix, torch.zeros((3, 1), dtype=torch.float32)], dim=1)

        # 将姿势矩阵添加到列表中
        image_poses.append(pose_matrix)

    return image_poses

import torch
import numpy as np

# 生成绕向前方向做小倾角圆周摆动的姿势
def generate_swinging_circular_poses(num_poses=100, radius=1.0, tilt_angle_degrees=5):
    image_poses = []

    # 将角度转换为弧度
    tilt_angle = np.radians(tilt_angle_degrees)

    for i in range(num_poses):
        # 计算平移坐标
        x = radius * np.cos(2 * np.pi * i / num_poses)
        y = radius * np.sin(2 * np.pi * i / num_poses)

        # 计算小倾角
        tilt_x = x * np.tan(tilt_angle)

        # 创建3x4的姿势矩阵，包含平移和小倾角
        pose_matrix = torch.tensor([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, np.tan(tilt_angle), 1, tilt_x]
        ], dtype=torch.float32)

        # 将姿势矩阵添加到列表中
        image_poses.append(pose_matrix)

    return image_poses

# 生成包含100个3x4张量的列表，绕向前方向做小倾角圆周摆动的姿势
swinging_circular_poses = generate_swinging_circular_poses(100, radius=1.0, tilt_angle_degrees=5)

# 生成水平方向左右摆动的姿势
def generate_horizontal_swing_poses(num_poses=100, amplitude=2.0, frequency=5):
    image_poses = []

    for i in range(num_poses):
        # 计算平移坐标，使用正弦函数模拟摆动
        x = amplitude * np.sin(2 * np.pi * frequency * i / num_poses)

        # 创建3x4的姿势矩阵，只包含平移部分
        pose_matrix = torch.tensor([
            [1, 0, 0, x],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ], dtype=torch.float32)

        # 将姿势矩阵添加到列表中
        image_poses.append(pose_matrix)

    return image_poses
class Dataset():

    def __init__(self, root, scene, split="train", subset=None):
        self.raw_H, self.raw_W = 480, 640
        self.path = "{}/{}".format(root, scene)
        self.img_path = self.path + "/rs/clean"
        poses_raw, image_fnames, image_timeStamp = self.read_data_and_get_image_poses()
        self.list = list(zip(image_fnames, poses_raw, image_timeStamp))

    def read_data_and_get_image_poses(self):
        # 读取cam0的图像数据
        cam0_path = self.img_path
        # 读取真实值的数据
        ground_truth_path = self.path + "/gt/tum_ground_truth_gs.txt"

        df_cam0 = pd.read_csv(cam0_path + "/data2.csv")

        df_groundtruth = pd.read_csv(ground_truth_path, skiprows=3, delim_whitespace=True,
                                     names=["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])

        Tic = CAM_SENSOR_YAML.T_BS.data
        Tic = np.array(Tic, dtype=np.float64)
        Tic = Tic.reshape(4, 4)

        image_poses = []
        image_fnames = []
        times = []
        for index, row in df_cam0.iloc[1:].iterrows():
            time = row['#timestamp [ns]']
            image_filename = row['filename']
            image_fnames.append(image_filename)
            # 查找与cv1文件中时间最接近的时间
            closest_time_row = df_groundtruth.iloc[(df_groundtruth['timestamp'] * 1e9 - time).abs().idxmin()]
            # print(((df_groundtruth['timestamp'] - time).abs().idxmin()))
            # 获取坐标和相机姿态信息
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

            image_poses.append(pose)
            times.append(time)

        # trajectory1
        # fast
        # image_poses = image_poses[550:650:1]
        # image_fnames = image_fnames[550:650:1]
        # times = times[550:650:1]

        # medium
        # image_poses = image_poses[1000:1200:2]
        # image_fnames = image_fnames[1000:1200:2]
        # times = times[1000:1200:2]

        # slow
        # image_poses = image_poses[1850:2280:4]
        # image_fnames = image_fnames[1850:2280:4]
        # times = times[1850:2280:4]

        # trajectory2
        # fast
        # image_poses = image_poses[110:190:1]
        # image_fnames = image_fnames[110:190:1]
        # times = times[110:190:1]

        # # medium
        # image_poses = image_poses[270:310:1]
        # image_fnames = image_fnames[270:310:1]
        # times = times[270:310:1]

        # slow
        # image_poses = image_poses[1290:1350:1]
        # image_fnames = image_fnames[1290:1350:1]
        # times = times[1290:1350:1]

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
        sample.update(
            image=image,
            time=time,
            intr=intr,
            pose=pose)
        return sample

    def get_image(self, idx):
        image_fname = "{}/{}/{}".format(self.img_path, "img", self.list[idx][0])
        image_time = self.list[idx][2]
        return image_fname, image_time

    # def get_intr(self, index):
    #     timestamps = [int(tup[2]) for tup in self.list]
    #     target_timestamp = timestamps[index]
    #     interpolated_times = torch.tensor(
    #         list(range(target_timestamp, target_timestamp + self.raw_H * readOut, readOut)))
    #     interpolated_poses = self.intepolater(interpolated_times)
    #
    #     # for i in range(len(interpolated_poses)):
    #     #     pose = interpolated_poses[i]
    #     #     with open("/home/xubo/Tri-MipRF-main/intr.txt", "a") as f:
    #     #         f.write(str(interpolated_poses[i]) + " " )
    #
    #     return interpolated_poses

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
            if idx % 2 == 0:
                frames[0].append(
                    {
                        'image_filename': Path(sample["image"]),
                        'image_times': sample['time'],
                        'lossmult': 1.0,
                    })
            else:
                frames[0].append(
                    {
                        'image_filename': Path(sample["image"]),
                        'image_times': sample['time'],
                        'lossmult': 1.0,
                    })
        else:
            frames[0].append(
                {
                    'image_filename': Path(sample["image"].replace('/rs/', '/gs/')),
                    'image_times': sample['time'],
                    'lossmult': 1.0,
                })
        poses[0].append(sample["pose"].numpy().astype(np.float32))

    aabb = np.array([-5, -5, -5, 5, 5, 5])

    print(poses[0])
    outputs = {
        'frames': frames,
        'poses': poses,
        'cameras': cameras,
        'aabb': aabb,
        'read_out_time': readOut,
        "period": 480,
        'pose_transfer': pose_transfer
    }
    return outputs
