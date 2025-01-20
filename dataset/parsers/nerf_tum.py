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
CAM_SENSOR_YAML = edict(
    sensor_type="camera",
    comment="VI-Sensor cam0 (MT9M034)",
    T_BS=edict(
        cols=4,
        rows=4,
        data=[-0.00596386151962109, -0.9999548067377452, 0.0074038393994521196, 0.05250867995679271,
              -0.9999628101053462, 0.006009708608102947, 0.006185613038727172, 0.0077584026106338085,
              -0.006229828408066805, -0.007366673911867888, -0.999953459593736, -0.04241050920541196,
              0.0, 0.0, 0.0, 1.0]
    ),
    rate_hz=20,
    resolution=[1280, 1024],
    camera_model="pinhole",
    intrinsics=[743.4286936207343, 743.5545205462922, 618.7186883884866, 506.7275058699658],
    distortion_model="equidistant",
    distortion_coefficients=[0.023584346301328347, -0.006764098468377487, 0.010259071387776937, -0.0037561745737771414]
)

CAM_SENSOR_YAML1 = edict(
    sensor_type="camera",
    comment="VI-Sensor cam1 (MT9M034)",
    T_BS=edict(
        cols=4,
        rows=4,
        data=[-0.0027687291789002095, -0.9999872674970001, 0.004218884048773266, -0.05688384754602901,
              -0.9999356528558058, 0.002814949729190873, 0.010989367855138411, 0.007618902284014501,
              -0.011001103879489809, -0.004188185992194848, -0.9999307150055582, -0.042436390295094266,
              0.0, 0.0, 0.0, 1.0]
    ),
    rate_hz=20,
    resolution=[1280, 1024],
    camera_model="pinhole",
    intrinsics=[739.1654756101043, 739.1438452683457, 625.826167006398, 517.3370973594253],
    distortion_model="equidistant",
    distortion_coefficients=[0.019327620961435945, 0.006784242994724914, -0.008658628531456217, 0.0051893686731546585]
)

instrinsic = CAM_SENSOR_YAML1.intrinsics

K = [[instrinsic[0], 0., instrinsic[2]],
     [0., instrinsic[1], instrinsic[3]],
     [0, 0, 1]]
K = np.array(K, dtype=np.float64)

D = CAM_SENSOR_YAML1.distortion_coefficients
D = np.array(D, dtype=np.float64)
readOut = 29.4737
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

        self.raw_H,self.raw_W = 1024, 1280
        self.path = "{}/{}".format(root, scene)
        self.img_path = "{}/{}".format(self.path, "cam1")
        poses_raw, image_fnames, image_timeStamp = self.read_data_and_get_image_poses()
        self.list = list(zip(image_fnames, poses_raw, image_timeStamp))
        # manually split train/val subsets
        # num_val_split = int(len(self.list) * 0.3)
        # step = len(self.list) // num_val_split
        # val_data = self.list[:: step]
        # # self.list = self.list[:-num_val_split] if split == "train" else self.list[-num_val_split:]
        # self.list = [item for item in self.list if item not in val_data] if split == "train" else val_data
        # if subset: self.list = self.list[:subset]


    def read_data_and_get_image_poses(self):
        # 读取cam0的图像数据
        cam0_path = self.img_path
        # 读取真实值的数据
        ground_truth_path = self.path + "/mocap0/"

        df_cam0 = pd.read_csv(cam0_path + "/data.csv")

        df_groundtruth = pd.read_csv(ground_truth_path + "data.csv")

        Tic = CAM_SENSOR_YAML1.T_BS.data
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
            closest_time_row = df_groundtruth.iloc[(df_groundtruth['#timestamp [ns]'] - time).abs().idxmin()]
            # print(((df_groundtruth['timestamp'] - time).abs().idxmin()))
            # 获取坐标和相机姿态信息
            twi = closest_time_row.iloc[1:4]
            qwi = [closest_time_row.iloc[5], closest_time_row.iloc[6], closest_time_row.iloc[7],
                   closest_time_row.iloc[4]]

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

        # dataset4
        image_poses = image_poses[10:100:2]
        image_fnames = image_fnames[10:100:2]
        times = times[10:100:2]
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
        image_fname = "{}/{}/{}".format(self.img_path, "data", self.list[idx][0])
        image_fname = self.undistorted_image(image_fname)
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


    def undistorted_image(self, image_fname):
        img = cv2.imread(image_fname)
        DIM = CAM_SENSOR_YAML1.resolution
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
        img_undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        # cv2.imshow('2',img)
        # cv2.waitKey(0)
        image_pil = PIL.Image.fromarray(cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2RGB))
        image_undist_fname = image_fname[:-4] + 'undist.png'
        image_pil.save(image_undist_fname)
        # image = PIL.Image.fromarray(
        #     imageio.imread(image_fname))  # directly using PIL.Image.open() leads to weird corruption....
        # image_rgb = image.convert('RGB')
        # image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        # cv2.imshow('2', image_cv)
        # cv2.waitKey(0)

        return image_undist_fname


def load_data(base_path: Path, scene: str, split: str, down_level=0, is_eval=False,  is_rolling=True):

    # splits = 'train' if split == "trainval" else split
    if is_eval:
        splits = 'val'
    else:
        splits = 'train'
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

    aabb = np.array([-3, -3, -3, 3, 3, 3])

    outputs = {
        'frames': frames,
        'poses': poses,
        'cameras': cameras,
        'aabb': aabb,
        'read_out_time': readOut,
        'period': 50000 * 2,
        'pose_transfer': pose_transfer
    }

    return outputs


