import ipdb
import numpy as np
import torch

from dataset.utils.poses import pose, lie
from thirdparty.LightGlue.lightglue import SuperPoint, LightGlue
from thirdparty.LightGlue.lightglue.utils import load_image, rbd
from utils import Spline
from sklearn.cluster import DBSCAN
from collections import Counter
import os
import pickle
import matplotlib.pyplot as plt
class PoseFilter:
    def __init__(self, point_matches, camera_instrinsic, pose_transfer, threshold=0.5):
        self.point_matches = point_matches
        self.camera_instrinsic =camera_instrinsic
        self.threshold = threshold
        self.pose_transfer = pose_transfer


    def compute_epipolar_constraint_error(self, cam_poses, cam_velocities):
        epipolar_errors = []
        for host_idx, target_poses_candidate_index in self.point_matches.items():
            duplicate_error = []
            for i in range(0, 2):
                target_idx = target_poses_candidate_index[i].idx
                pose0 = cam_poses[host_idx]
                pose1 = cam_poses[target_idx]
                points0 = target_poses_candidate_index[i].pt_i.cuda()
                points1 = target_poses_candidate_index[i].pt_j.cuda()
                pose0 = pose0 @ torch.tensor(self.pose_transfer).float().cuda()
                pose1 = pose1 @ torch.tensor(self.pose_transfer).float().cuda()
                if cam_velocities != None:
                    wv0 = cam_velocities[host_idx]
                    wv1 = cam_velocities[target_idx]

                    tcc0 = lie.se3_to_SE3(lie.SE3_to_se3(wv0) * points0[:, 1].view(-1, 1))
                    pose0 = pose.compose_pair(pose0, tcc0)
                    tcc1 = lie.se3_to_SE3(lie.SE3_to_se3(wv1) * points1[:, 1].view(-1, 1))
                    pose1 = pose.compose_pair(pose1, tcc1)
                K = self.camera_instrinsic.cuda()
                K_inv = torch.inverse(K)
                K_inv_t = K_inv.permute(-1, -2)
                R = pose.compose_pair(pose.invert(pose0), pose1)
                R_a, t_a = R[..., :3], R[..., 3:]
                # essential matrix
                E_matrix = lie.skew_symmetric(t_a.squeeze()) @ R_a
                error = points0.unsqueeze(-1).transpose(1, 2) @ K_inv_t @ E_matrix @ K_inv @ points1.unsqueeze(-1)
                error_sum = torch.sqrt(torch.sum(error * error) / len(error))
                duplicate_error.append(error_sum)
                target_poses_candidate_index[i].errors = error_sum
            epipolar_errors.append(duplicate_error)

        # for pose_indexs, point_matches in self.point_matches.items():
        #
        #     pose0 = cam_poses[pose_indexs[0]]
        #     pose1 = cam_poses[pose_indexs[1]]
        #     points0 = point_matches[0].cuda()
        #     points1 = point_matches[1].cuda()
        #
        #     # 重投影误差计算
        #     pose0 = pose0 @ torch.tensor(self.pose_transfer).float().cuda()
        #     pose1 = pose1 @ torch.tensor(self.pose_transfer).float().cuda()
        #
        #     if cam_velocities != None:
        #         wv0 = cam_velocities[pose_indexs[0]]
        #         wv1 = cam_velocities[pose_indexs[1]]
        #         tcc0 = lie.se3_to_SE3(lie.SE3_to_se3(wv0) * points0[:, 1].view(-1, 1))
        #         pose0 = pose.compose_pair(pose0, tcc0)
        #         tcc1 = lie.se3_to_SE3(lie.SE3_to_se3(wv1) * points1[:, 1].view(-1, 1))
        #         pose1 = pose.compose_pair(pose1, tcc1)
        #
        #     K = self.camera_instrinsic.cuda()
        #     K_inv = torch.inverse(K)
        #     K_inv_t = K_inv.permute(-1, -2)
        #     R = pose.compose_pair(pose.invert(pose0), pose1)
        #     R_a, t_a = R[..., :3], R[..., 3:]
        #     # essential matrix
        #     E_matrix = lie.skew_symmetric(t_a.squeeze()) @ R_a
        #     error = points0.unsqueeze(-1).transpose(1, 2) @ K_inv_t @ E_matrix @ K_inv @ points1.unsqueeze(-1)
        #     error_sum = torch.sqrt(torch.sum(error * error) / len(error))
        #
        #     epipolar_errors.append(error_sum)

        return epipolar_errors

    @torch.no_grad()
    def detect_bad_poses(self, estimated_poses, estimated_velocities):

        updated_poses = estimated_poses
        errors = self.compute_epipolar_constraint_error(estimated_poses, estimated_velocities)  # 计算重投影误差
        problematic_frames = []
        errors_tensor = torch.tensor(errors, dtype=torch.float32)
        dbscan = DBSCAN(eps=1, min_samples=2)
        cluster_data = errors_tensor.view(-1, 1).numpy()
        labels = dbscan.fit_predict(cluster_data)
        cluster_key_value = Counter(labels)
        key = sorted(cluster_key_value, key=cluster_key_value.get, reverse=True)

        mean_value = np.mean(cluster_data)
        std_value = np.std(cluster_data)

        if len(key) >=2 and cluster_key_value[key[0]] > 3 * cluster_key_value[key[1]]:
            index = np.where(labels==key[0])
            clustered_errors = cluster_data[index]
            mean_value = np.mean(clustered_errors)
            std_value = np.std(clustered_errors)

        threshold_all = mean_value + self.threshold * std_value  # 选择适当的倍数

        for i in range(1, len(errors)):
            target_poses_candidate_index = self.point_matches[i]
            errors_0 = target_poses_candidate_index[0].errors
            errors_1 = target_poses_candidate_index[1].errors
            if errors_0 > threshold_all and errors_1 > threshold_all:
                # print(errors_0,errors_1,threshold_all)
                problematic_frames.append(i)
                # idx0 = target_poses_candidate_index[0].idx
                # idx1 = target_poses_candidate_index[1].idx
                # lie0 = lie.SE3_to_se3(updated_poses[idx0])
                # lie1 = lie.SE3_to_se3(updated_poses[idx1])
                # 将输入的 updated_poses 转换为 numpy 数组，方便计算
                poses = np.array(lie.SE3_to_se3(updated_poses.cpu()))

                # 获取第 i 个位置
                target_pose = poses[i, :3]

                # 计算第 i 个位置与其他位置的距离
                distances = np.linalg.norm(poses[:, :3] - target_pose, axis=1)

                # 将第 i 个位置的距离设为无穷大，避免自己与自己比较
                distances[i] = np.inf

                # 找到距离最近的两个点的索引
                closest_indices = np.argpartition(distances, 2)[:2]
                # 获取最近的两个位置
                closest_poses = poses[closest_indices, :3]
                print(poses[i, :3])
                print(closest_poses)

                # 计算这两个位置的均值
                poses[i, :3] = np.mean(closest_poses, axis=0)
                updated_poses[i] = lie.se3_to_SE3(torch.tensor(poses[i])).cuda()

        print("bad frames: ", problematic_frames)
        return updated_poses, len(problematic_frames)


def plot_camera_poses(poses, name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 设置坐标轴
    ax.set_xlim([-1, 1])
    ax.set_ylim([-0.1, 0.1])
    ax.set_zlim([-0.1, 0.1])

    for pose in poses:
        # 提取旋转矩阵和平移向量
        R = pose[:3, :3].cpu().numpy()
        t = pose[:3, 3].cpu().numpy()

        # 绘制摄像机坐标系
        camera_center = t
        x_axis = R[:, 0]
        y_axis = R[:, 1]
        z_axis = R[:, 2]

        ax.quiver(camera_center[0], camera_center[1], camera_center[2],
                  x_axis[0], x_axis[1], x_axis[2], color='r', length=0.1)
        ax.quiver(camera_center[0], camera_center[1], camera_center[2],
                  y_axis[0], y_axis[1], y_axis[2], color='g', length=0.1)
        ax.quiver(camera_center[0], camera_center[1], camera_center[2],
                  z_axis[0], z_axis[1], z_axis[2], color='b', length=0.1)
        ax.scatter(camera_center[0], camera_center[1], camera_center[2], color='k')

    plt.savefig(name)