import unittest

import gtsam
import numpy as np
import matplotlib.pyplot as plt
ROTATION_ANGLE_ERROR_THRESHOLD_DEG = 2

from dataset.parsers.nerf_whu import Dataset

from gtsam import (
    BetweenFactorPose3,
    BetweenFactorPose3s,
    LevenbergMarquardtParams,
    Pose3,
    Rot3,
    ShonanAveraging3,
    ShonanAveragingParameters3,
)

POSE3_DOF = 6
class GraphAveraging():

    # def translationAveraging(self, factors, global_rotation):
    #
    #     num_observations = len(factors) + 1
    #     num_variables = len(global_rotation)
    #     H_matrix = np.zeros((num_observations*3, num_variables*3))
    #     W_matrix = np.zeros((num_observations*3, num_observations*3))
    #     b_matrix = np.zeros((num_observations*3, 1))
    #
    #     row_idx = 0
    #     for factor in factors:
    #         veti = factor.keys()[0]
    #         vetj = factor.keys()[1]
    #         trans = factor.measured().translation()
    #         noise = factor.noiseModel().sigmas()
    #         weight = np.diag(1. / noise[3:])
    #         H_matrix[row_idx:row_idx+3, veti*3:(veti+1)*3] =  np.identity(3)
    #         H_matrix[row_idx:row_idx+3, vetj*3:(vetj+1)*3] = -np.identity(3)
    #         b_matrix[row_idx:row_idx+3, 0] = global_rotation[vetj].matrix()@trans
    #         W_matrix[row_idx: row_idx+3, row_idx:row_idx+3]= weight
    #         row_idx += 3
    #
    #     H_matrix[-3:, 0:3] = np.identity(3)
    #     b_matrix[-3:, 0] = np.zeros(3)
    #     W_matrix[-3:, -3:] = np.identity(3)
    #
    #
    #     W_matrix_sqrt = np.linalg.cholesky(W_matrix)
    #
    #     H_matrix_weighted = np.dot(W_matrix_sqrt, H_matrix)
    #     b_matrix_weighted = np.dot(W_matrix_sqrt, b_matrix)
    #
    #     x, residuals, rank, s = np.linalg.lstsq(H_matrix_weighted, b_matrix_weighted, rcond=None)
    #     result=[]
    #     for i in range(num_variables):
    #         estimated_poses = np.zeros((3, 4))
    #         estimated_poses[:3, :3]= global_rotation[i].matrix()
    #         estimated_poses[:3, 3:]= x[i*3: (i+1)*3]
    #         result.append(estimated_poses)
    #     return result
    def translationAveraging(self,
                             factors: gtsam.BetweenFactorPose3s,
                             rotations: gtsam.Values,
                             d: int = 3):

        I_d = np.eye(d)

        def R(j):
            return rotations.atRot3(j) if d == 3 else rotations.atRot2(j)

        def pose(R, t):
            return gtsam.Pose3(R, t) if d == 3 else gtsam.Pose2(R, t)

        graph = gtsam.GaussianFactorGraph()
        model = gtsam.noiseModel.Unit.Create(d)

        # Add a factor anchoring t_0
        graph.add(0, I_d, np.zeros((d,)), model)

        # Add a factor saying t_j - t_i = Ri*t_ij for all edges (i,j)
        for factor in factors:
            keys = factor.keys()
            i, j, Tij = keys[0], keys[1], factor.measured()
            measured = R(i).rotate(Tij.translation())
            graph.add(j, I_d, i, -I_d, measured, model)

        # Solve linear system
        translations = graph.optimize()


        result = []
        for j in range(rotations.size()):
            tj = translations.at(j)
            estimated_poses = np.zeros((3, 4))
            estimated_poses[:3, :3]= R(j).matrix()
            estimated_poses[:3, 3]= tj
            result.append(estimated_poses)

        return result

    def __get_shonan_params(self) -> ShonanAveragingParameters3:
        lm_params = LevenbergMarquardtParams.CeresDefaults()
        shonan_params = ShonanAveragingParameters3(lm_params)
        shonan_params.setUseHuber(False)
        shonan_params.setCertifyOptimality(True)
        return shonan_params



    i2Ri1_input = {}
    def generate_test_dataset(self):
        base_path = '/home/xubo/rsvi_t1_fast/'
        scene = 'rsvi_t1_fast'
        dataset = Dataset(base_path, scene)
        Twc = [i[1].numpy() for i in dataset.list]
        pose_input = []
        factors = gtsam.BetweenFactorPose3s()
        for i in range(len(Twc)-1):
            j = i+1
            pose_w_i = Pose3(Twc[i])
            pose_w_j = Pose3(Twc[j])

            # random_rotation_matrix = R.random().as_matrix()
            # random_rotation = gtsam.Rot3(random_rotation_matrix)
            # pose_w_i = Pose3(random_rotation, pose_w_i.translation())
            #
            # random_rotation_matrix = R.random().as_matrix()
            # random_rotation = gtsam.Rot3(random_rotation_matrix)
            # pose_w_j = Pose3(random_rotation, pose_w_j.translation())

            # wRi_input.append(pose_w_i.rotation())
            pose_input.append(pose_w_i)

            # pose_w_i^-1 * pose_w_j = pose_i_j
            factor = gtsam.BetweenFactorPose3(j, i, pose_w_i.between(pose_w_j), gtsam.noiseModel.Isotropic.Sigma(POSE3_DOF, 0.00001))
            factors.append(factor)

        # wRi_input.append(pose_w_j.rotation())
        pose_input.append(pose_w_j)

        # generate i2Ri1 rotations
        # (i1,i2) -> i2Ri1
        shonan = gtsam.ShonanAveraging3(factors, self.__get_shonan_params())
        # shonan = gtsam.ShonanAveraging3("/home/xubo/ShonanAveraging/gtsam/pose3example-grid.txt", self.__get_shonan_params())
        if shonan.nrUnknowns() == 0:
            raise ValueError("No 3D pose constraints found, try -d 2.")
        initial = shonan.initializeRandomly()
        rotations, _ = shonan.run(initial, 5, 30)

        # todo debug 验证factor代码对不对
        # factors = gtsam.parse3DFactors("/home/xubo/ShonanAveraging/gtsam/pose3example-grid.txt")
        # todo debug 验证factor代码对不对
        # rotations = gtsam.Values()
        # for i, pose in enumerate(pose_input):
        #     rotations.insert(i, pose.rotation())

        estimated_poses = self.translationAveraging(factors, rotations)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = [], [], []
        for i in range(len(estimated_poses)):
            x.append(estimated_poses[i][0, 3])
            y.append(estimated_poses[i][1, 3])
            z.append(estimated_poses[i][2, 3])
        ax.plot(x, y, z)
        plt.show()




graphAvg = GraphAveraging()
graphAvg.generate_test_dataset()

