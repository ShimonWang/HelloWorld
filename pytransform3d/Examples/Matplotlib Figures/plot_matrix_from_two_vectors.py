"""
==========================================
Construct Rotation Matrix from Two
# 从两个向量构造旋转矩阵
==========================================

We compute rotation matrix from two vectors that form a plane. The x-axis will
point in the same direction as the first vector, the y-axis corresponds to the
normalized vector rejection of b on a, and the z-axis is the cross product of
the other basis vectors.
# 我们计算由两个形成平面的向量构成的旋转矩阵。x 轴将指向第一个向量的方向，
y 轴对应于向量 b 在 a 上的归一化向量排斥，z 轴是其他基向量的叉积。
"""

import matplotlib.pyplot as plt
import numpy as np

from pytransform3d.plot_utils import plot_vector
from pytransform3d.rotations import (
    matrix_from_two_vectors,
    plot_basis,
    random_vector,
)

rng = np.random.default_rng(1)
a = random_vector(rng, 3) * 0.3  # random_vector:生成具有正太分布分量的 nd 向量
b = random_vector(rng, 3) * 0.3
c = random_vector(rng, 2)
print(f"rng:{rng}")
print(f"{c}")
R = matrix_from_two_vectors(a, b)

ax = plot_vector(direction=a, color="r")
plot_vector(ax=ax, direction=b, color="g")
plot_basis(ax=ax, R=R)
plt.show()
