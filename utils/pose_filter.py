import numpy as np
import torch

from dataset.utils.poses import pose, lie
from thirdparty.LightGlue.lightglue import SuperPoint, LightGlue
from thirdparty.LightGlue.lightglue.utils import load_image, rbd
from utils import Spline

class PoseFilter:
    def __init__(self, point_matches, camera_instrinsic, pose_transfer, threshold=0.5):
        self.point_matches = point_matches
        self.camera_instrinsic =camera_instrinsic
        self.threshold = threshold
        self.pose_transfer = pose_transfer


    def compute_epipolar_constraint_error(self, cam_poses, cam_velocities):

        epipolar_errors = []

        for pose_indexs, point_matches in self.point_matches.items():

            pose0 = cam_poses[pose_indexs[0]]
            pose1 = cam_poses[pose_indexs[1]]
            points0 = point_matches[0].cuda()
            points1 = point_matches[1].cuda()

            # 重投影误差计算
            pose0 = pose0 @ torch.tensor(self.pose_transfer).float().cuda()
            pose1 = pose1 @ torch.tensor(self.pose_transfer).float().cuda()

            if cam_velocities != None:
                wv0 = cam_velocities[pose_indexs[0]]
                wv1 = cam_velocities[pose_indexs[1]]
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

            epipolar_errors.append(error_sum)

        return epipolar_errors

    @torch.no_grad()
    def detect_bad_poses(self, estimated_poses, estimated_velocities):
        errors = self.compute_epipolar_constraint_error(estimated_poses, estimated_velocities)  # 计算重投影误差
        problematic_frames = []
        errors_tensor = torch.tensor(errors, dtype=torch.float32)

        updated_poses = estimated_poses
        # 统计描述
        mean_value = torch.mean(errors_tensor)
        std_value = torch.std(errors_tensor)
        threshold_all = mean_value + self.threshold * std_value  # 选择适当的倍数


        # todo debug new intepolation algorithm
        for i in range(0, len(errors)-1):
            error_prev = errors[i]
            error_next = errors[i + 1]

            if error_prev > threshold_all and error_next > threshold_all:
                # 如果两帧的重投影误差都很大，则认为当前帧有问题
                problematic_frames.append(i+1)
                # TODO 插值获取错误帧的pose
                # updated_poses[i+1] = (updated_poses[i] + updated_poses[i+2])/2.0

        # for frame in problematic_frames:
        #
        #     left_frame, right_frame = frame, frame
        #
        #     while left_frame >= 0:
        #         left_frame -= 1
        #         if left_frame not in problematic_frames:
        #             break
        #
        #     while right_frame < len(estimated_poses):
        #         right_frame += 1
        #         if right_frame not in problematic_frames:
        #             break
        #
        #     # 如果是最左面没有好的，那么就用最右面的一帧
        #     if left_frame < 0:
        #         updated_poses[frame] = updated_poses[right_frame]
        #         continue
        #     # 如果最右面没有好的，那么就用最左面的一帧
        #     if right_frame >= len(estimated_poses):
        #         updated_poses[frame] = updated_poses[left_frame]
        #         continue
        #
        #     interval = right_frame - left_frame + 1
        #
        #     pose_nums = torch.arange(interval).reshape(1, -1).repeat(1, 1).to('cuda')
        #
        #     se3_start = lie.SE3_to_se3(updated_poses[left_frame]).cuda()
        #     se3_end = lie.SE3_to_se3(updated_poses[right_frame]).cuda()
        #
        #     spline_poses = Spline.SplineN_linear(se3_start, se3_end, pose_nums, interval).reshape(-1, interval, 3,
        #                                                                                           4)
        #     for i in range(left_frame, right_frame):
        #         updated_poses[i] = spline_poses[0][i-left_frame]

        print("bad frames: ", problematic_frames)

        return updated_poses, len(problematic_frames)
