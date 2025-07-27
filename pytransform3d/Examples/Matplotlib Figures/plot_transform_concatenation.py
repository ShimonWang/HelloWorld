"""
=======================
Transform Concatenation
# 变换连接
=======================

In this example, we have a point p that is defined in a frame C, we know
the transform C2B and B2A. We can construct a transform C2A to extract the
position of p in frame A.
# 在这个示例中，我们有一个在坐标系 C 中定义的点 p，我们知道从 C 到 B 的变换以及从 B 到 A 的变换。
我们可以构建一个从 C 到 A 的变换来提取点 p 在坐标系 A 中的位置。
"""

import matplotlib.pyplot as plt
import numpy as np

import pytransform3d.rotations as pyrot
import pytransform3d.transformations as pytr

p = np.array([0.0, 0.0, -0.5])
a = np.array([0.0, 0.0, 1.0, np.pi])
B2A = pytr.transform_from(pyrot.matrix_from_axis_angle(a), p)

p = np.array([0.3, 0.4, 0.5])
a = np.array([0.0, 0.0, 1.0, -np.pi / 2.0])
C2B = pytr.transform_from(pyrot.matrix_from_axis_angle(a), p)

C2A = pytr.concat(C2B, B2A)
p = pytr.transform(C2A, np.ones(4))

C2A_my = B2A @ C2B

print(f"C2A:{np.round(C2A, decimals=2)}")
print(f"C2A_my:{np.round(C2A_my, decimals=2)}")

pyrot.plot_basis()
# ax = pytr.plot_transform(A2B=B2A, name="B2A")
# pytr.plot_transform(ax, A2B=C2A, name="C2A")
# pytr.plot_transform(ax, A2B=C2B, name="C2B")
# ax.scatter(p[0], p[1], p[2])
pytr.plot_transform(A2B=B2A, name="B2A")
pytr.plot_transform(A2B=C2A, name="C2A")
pytr.plot_transform(A2B=C2B, name="C2B")
plt.scatter(p[0], p[1], p[2])
plt.show()
