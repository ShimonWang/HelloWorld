"""
====================================================
Axis-Angle Representation from Two Direction Vectors
# 从两个方向向量获取轴角表示
====================================================

This example shows how we can compute the axis-angle representation of a
rotation that transforms a direction given by a vector 'a' to a direction
given by a vector 'b'. We show both vectors, the rotation about the rotation
axis and the initial and resulting coordinate frame, where the vector 'b'
and its corresponding frame after the rotation are represented by shorter
lines.
# 本示例展示了如何计算将向量‘a’给定的方向变换为向量‘b’给定的方向的旋转的轴角表示。
我们展示了这两个向量、绕旋转轴的旋转以及初始和结果坐标系，其中向量‘b’及其旋转后的对应坐标系用较短的线条表示。
"""

import matplotlib.pyplot as plt
import numpy as np

from pytransform3d.plot_utils import make_3d_axis, plot_vector
from pytransform3d.rotations import (
    axis_angle_from_two_directions,
    matrix_from_axis_angle,
    plot_axis_angle,
    plot_basis,
)

a = np.array([1.0, 0.0, 0.0])
b = np.array([0.76958075, -0.49039301, -0.40897453])
aa = axis_angle_from_two_directions(a, b)  # 从两个方向向量计算轴角表示

ax = make_3d_axis(ax_s=1)  # 生成新的三维坐标轴
plot_vector(ax, start=np.zeros(3), direction=a, s=1.0)
plot_vector(ax, start=np.zeros(3), direction=b, s=0.5)
plot_axis_angle(ax, aa)  # 绘制旋转轴和角度
plot_basis(ax)
plot_basis(ax, R=matrix_from_axis_angle(aa), s=0.5)
plt.show()

# ax = plot_basis()
# plot_vector(ax, start=np.zeros(3), direction=a, s=1.0)
# plot_vector(ax, start=np.zeros(3), direction=b, s=0.5)
# plot_axis_angle(ax, aa)  # 绘制旋转轴和角度
# plot_basis(ax, R=matrix_from_axis_angle(aa), s=0.5)
# plt.show()