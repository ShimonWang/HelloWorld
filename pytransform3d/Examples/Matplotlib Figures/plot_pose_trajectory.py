"""
===============
Pose Trajectory
===============

Plotting pose trajectories with pytransform3d is easy.
"""

import matplotlib.pyplot as plt
import numpy as np

from pytransform3d.batch_rotations import quaternion_slerp_batch
from pytransform3d.rotations import q_id  # 单位四元数 [1,0,0,0]
from pytransform3d.trajectories import plot_trajectory

n_steps = 100000
P = np.empty((n_steps, 7))
P[:, 0] = np.cos(np.linspace(-2 * np.pi, 2 * np.pi, n_steps))  # x 坐标
P[:, 1] = np.sin(np.linspace(-2 * np.pi, 2 * np.pi, n_steps))  # y 坐标
P[:, 2] = np.linspace(-1, 1, n_steps)  # z 坐标
q_end = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])  # 绕 [0,sqrt(2),sqrt(2)] 旋转 pi 弧度
P[:, 3:] = quaternion_slerp_batch(q_id, q_end, np.linspace(0, 1, n_steps))  # 球面线性插值批量步骤

ax = plot_trajectory(
    P=P, s=0.3, n_frames=100, normalize_quaternions=True, lw=2, c="k"
)
plt.show()
