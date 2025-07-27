"""
======================
Quaternion Integration
#
======================

Integrate angular velocities to a sequence of quaternions.
"""

import matplotlib.pyplot as plt
import numpy as np

from pytransform3d.rotations import (
    quaternion_integrate,
    matrix_from_quaternion,
    plot_basis,
)
from pytransform3d.plot_utils import plot_vector

angular_velocities = np.empty((21, 3))
angular_velocities[:, :] = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0])
angular_velocities *= np.pi

Q = quaternion_integrate(angular_velocities, dt=0.1)
ax = None
for t in range(len(Q)):
    R = matrix_from_quaternion(Q[t])  # t时刻的姿态（旋转矩阵）
    p = 2 * (t / (len(Q) - 1) - 0.5) * np.ones(3)  # t时刻的平移矩阵
    # p[0] = 0  # t时刻的平移矩阵
    p[2] = 0  # t时刻的平移矩阵
    ax = plot_basis(ax=ax, s=0.15, R=R, p=p)

plot_vector(
    start=[-1,-1,0],
    direction=np.array([1.0, 1.0, 0]),
    s=0.5,  # 将要被绘制的矢量的放缩比例
    ax_s=1.0,  # Scaling of 3D axes
    lw=0,  # Remove line around arrow
    color="orange",
)
plt.show()
