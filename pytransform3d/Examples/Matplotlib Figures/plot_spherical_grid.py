"""
==============
Spherical Grid
# 球面网格
==============

Plot a grid in spherical coordinates with rho = 1 as Cartesian points.
We can see that the Cartesian distances between points are not regular and
there are many points that were converted to the same Cartesian point at the
poles of the sphere.
# 将 rho = 1 的球面坐标网格绘制为笛卡尔点。我们可以看到，点与点之间的笛卡尔距离并不规则，
有许多点在球面的两极被转换为同一个笛卡尔点。
"""

import matplotlib.pyplot as plt
import numpy as np

import pytransform3d.coordinates as pc
from pytransform3d.plot_utils import make_3d_axis

thetas, phis = np.meshgrid(  # 返回由坐标向量组成的坐标矩阵元组
    np.linspace(0, np.pi, 11), np.linspace(-np.pi, np.pi, 21)
)
rhos = np.ones_like(thetas)
# print(f"thetas.reshape(-1):{thetas.reshape(-1)}")
# print(f"rhos.reshape(-1):{rhos.reshape(-1)}")
# thetas_reshape = thetas.reshape(-1)  # 将数组平铺
spherical_grid = np.column_stack(  # 列合并 将一维数组堆叠成一个二维数组
    (rhos.reshape(-1), thetas.reshape(-1), phis.reshape(-1))
)
cartesian_grid = pc.cartesian_from_spherical(spherical_grid)  # 将球坐标转换为笛卡尔坐标

ax = make_3d_axis(ax_s=1, unit="m", n_ticks=6)
ax.scatter(cartesian_grid[:, 0], cartesian_grid[:, 1], cartesian_grid[:, 2])
ax.plot(cartesian_grid[:, 0], cartesian_grid[:, 1], cartesian_grid[:, 2])
plt.show()
