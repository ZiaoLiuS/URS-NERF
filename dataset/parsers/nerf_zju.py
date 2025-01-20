from pathlib import Path
import numpy as np

import dataset.utils.io as data_io
from dataset.utils.cameras import PinholeCamera

import numpy as np
import pandas as pd
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import torch.multiprocessing as mp
import PIL
import tqdm
import threading,queue
from easydict import EasyDict as edict
from scipy.spatial.transform import Rotation
import copy
import cv2

CAM_SENSOR_YAML1 = edict(
    sensor_type="camera",
    comment="Xiaomi Mi 8",
    rate_hz=30,
    resolution=[640, 480],
    camera_model="pinhole",
    intrinsics=[493.0167, 491.55953, 317.97856, 242.392],
    distortion_model="radtan",
    distortion_coefficients=[0, 0, 0, 0]
)


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

    def __init__(self,root, scene, split="train",subset=None):

        self.raw_H,self.raw_W = 480, 640
        self.path = "{}/{}".format(root, scene)
        self.img_path = "{}/{}".format(self.path, "camera")

        self.instrinsic = CAM_SENSOR_YAML1.intrinsics
        self.DIM = CAM_SENSOR_YAML1.resolution

        self.K = [[self.instrinsic[0], 0., self.instrinsic[2]],
             [0., self.instrinsic[1], self.instrinsic[3]],
             [0, 0, 1]]

        self.K = np.array(self.K, dtype=np.float64)


        poses_raw, image_fnames, image_timeStamp = self.read_data_and_get_image_poses()
        self.list = list(zip(image_fnames, poses_raw, image_timeStamp))
        # np.random.seed(42)
        # np.random.shuffle(self.list)
        # manually split train/val subsets
        # num_val_split = int(len(self.list) * 0.1)
        # num_test_split = 1
        # step = len(self.list) // num_val_split
        # val_data = self.list[:: step]
        # # self.list = self.list[:-num_val_split] if split == "train" else self.list[-num_val_split:]
        # self.list = [item for item in self.list if item not in val_data] if split == "train" else val_data
        # if subset: self.list = self.list[:subset]


    def read_data_and_get_image_poses(self):
        # 读取cam0的图像数据
        cam0_path = self.img_path
        # 读取真实值的数据
        ground_truth_path = self.path + "/groundtruth/"

        df_cam0 = pd.read_csv(cam0_path + "/data.csv")

        df_groundtruth = pd.read_csv(ground_truth_path + "euroc_gt.csv")
        # df_groundtruth = pd.read_csv(ground_truth_path + "euroc_gt_camera.csv")

        image_poses = []
        image_fnames = []
        times = []
        for index, row in df_cam0.iloc[1:].iterrows():
            time = row['#t[s:double]']
            image_filename = row['filename[string]']
            image_fnames.append(image_filename)
            # 查找与cv1文件中时间最接近的时间
            closest_time_row = df_groundtruth.iloc[(df_groundtruth['#timestamp[ns]'] - time * 1e9).abs().idxmin()]

            # 获取坐标和相机姿态信息
            twi = closest_time_row.iloc[1:4]
            qwi = [closest_time_row.iloc[5], closest_time_row.iloc[6], closest_time_row.iloc[7],
                   closest_time_row.iloc[4]]

            tic = np.array([-0.00165, -0.009950000000000001, 0.00067])
            qic = np.array([0.707107, -0.707107, 0, 0])

            Tic = np.identity(4)
            Ric = Rotation.from_quat(qic)
            Tic[:3, :3] = Ric.as_matrix()
            Tic[:3, 3] = tic

            Twi = np.identity(4)
            Rwi = Rotation.from_quat(qwi)
            Twi[:3, :3] = Rwi.as_matrix()
            Twi[:3, 3] = twi

            Twc = Twi @ Tic
            # Twc = Twi
            Twc = Twc @ pose_transfer
            Rwc = torch.tensor(Twc[:3, :3]).float()
            twc = torch.tensor(Twc[:3, 3]).float()
            pose = torch.cat([Rwc, twc[..., None]], dim=-1)  # [...,3,4]
            image_poses.append(pose)
            times.append(time)
        # # D0
        # image_poses = image_poses[180:401:3]
        # image_fnames = image_fnames[180:401:3]
        # times = times[180:401:3]
        # D1
        # image_poses = image_poses[100:201:2]
        # image_fnames = image_fnames[100:201:2]
        # times = times[100:201:2]
        # # D2
        # image_poses = image_poses[0:151:2]
        # image_fnames = image_fnames[0:151:2]
        # times = times[0:151:2]
        # # D3
        # image_poses = image_poses[0:101:2]
        # image_fnames = image_fnames[0:101:2]
        # times = times[0:101:2]
        # # # D5
        # image_poses = image_poses[0:101:2]
        # image_fnames = image_fnames[0:101:2]
        # times = times[0:101:2]
        # D8
        # image_poses = image_poses[524:724:2]
        # image_fnames = image_fnames[524:724:2]
        # times = times[524:724:2]
        # C5
        image_poses = image_poses[1350:1500:2]
        image_fnames = image_fnames[1350:1500:2]
        times = times[1350:1500:2]
        # # # c1
        # image_poses = image_poses[2540:2700:1]
        # image_fnames = image_fnames[2540:2700:1]
        # times = times[2540:2700:1]
        # # c11
        # image_poses = image_poses[250:400:3]
        # image_fnames = image_fnames[250:400:3]
        # times = times[250:400:3]
        poses_raw = self.center_camera_poses(torch.stack(image_poses, dim=0))
        # 初始化内插器： 中心化的pose， 时间（ns）
        return poses_raw, image_fnames, times

    def center_camera_poses(self,poses):
        # compute average pose
        center = poses[...,3].mean(dim=0)
        v1 = torch_F.normalize(poses[...,1].mean(dim=0),dim=0)
        v2 = torch_F.normalize(poses[...,2].mean(dim=0),dim=0)
        v0 = v1.cross(v2)
        pose_avg = torch.stack([v0,v1,v2,center],dim=-1)[None] # [1,3,4]
        # apply inverse of averaged pose
        pose_avg_inv = invert(pose_avg)
        poses = compose_pair(poses, pose_avg_inv)
        return poses

    def __getitem__(self,idx):
        sample = dict(idx=idx)
        image, time =  self.get_image(idx)
        intr,pose = self.get_camera(idx)
        # poses = self.get_intr(idx)
        sample.update(
            image=image,
            time=time,
            intr=intr,
            pose=pose)
        return sample

    def get_image(self, idx):
        image_fname = "{}/{}/{}".format(self.img_path, "images", self.list[idx][0])
        image_fname = self.undistorted_image(image_fname)
        image_time = self.list[idx][2]
        return image_fname, image_time

    def get_camera(self, idx):
        intr = torch.tensor(self.K).float()
        pose_raw = self.list[idx][1]
        return intr, pose_raw

    def undistorted_image(self, image_fname):
        img = cv2.imread(image_fname)
        image_pil = PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        image_convert = image_fname[:-4] + 'rgb.png'
        image_pil.save(image_convert)
        return image_convert



def load_data(base_path: Path, scene: str, split: str, down_level=0, is_eval=False, is_rolling=False):

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
            fy=dataset.K[1, 1]/ 2 ** n_down,
            cx=dataset.K[0, 2]/ 2 ** n_down,
            cy=dataset.K[1, 2]/ 2 ** n_down,
            width=int(dataset.raw_W / 2 ** n_down),
            height=int(dataset.raw_H / 2 ** n_down)
        )
    ]

    cam_num = len(cameras)

    assert cam_num == 1

    frames, poses = ({k: [] for k in range(len(cameras))},
                    {k: [] for k in range(len(cameras))})

    for idx in range(len(dataset.list)):
        sample = dataset[idx]
        frames[0].append(
            {
                'image_filename': Path(sample["image"]),
                'image_times': sample['time'],
                'lossmult': 1.0,
            })
        poses[0].append(sample["pose"].numpy().astype(np.float32))

    aabb = np.array([-2, -2, -2, 2, 2, 2])

    outputs = {
        'frames': frames,
        'poses': poses,
        'cameras': cameras,
        'aabb': aabb,
        'read_out_time': readOut,
        "period": 480 * 2 * 3 // 2 ** n_down,
        'pose_transfer': pose_transfer
    }

    return outputs


