"""
=====================================
Axis-Angle Representation of Rotation
# 旋转矩阵的轴角表示
=====================================

Any rotation can be represented with a single rotation about some axis.
Here we see a frame that is rotated in multiple steps around a rotation
axis.
# 任何旋转都可以用单个绕某一轴的旋转来表示。在这里我们看到一个围绕旋转轴做多步旋转的坐标系
"""

import matplotlib.pyplot as plt
import numpy as np

from pytransform3d.rotations import (
    random_axis_angle,
    matrix_from_axis_angle,
    plot_basis,
    plot_axis_angle,
)

original = random_axis_angle(np.random.RandomState(5))
print(original)
# 容器用于慢速梅森旋转伪随机数生成器。考虑使用带有生成器容器的不同位生成器代替
ax = plot_axis_angle(a=original)  # 绘制旋转轴和角度 a:(x,y,z,angle)
for fraction in np.linspace(0, 1, 50):
    # a = original
    a = original.copy()
    a[-1] = fraction * original[-1]
    R = matrix_from_axis_angle(a)
    plot_basis(ax, R, alpha=0.2)
plt.subplots_adjust(left=0, right=1, bottom=0, top=1.1)  # 调整子图布局参数
# plt.subplots_adjust(left=0, right=1, bottom=0, top=1)  # 调整子图布局参数
ax.view_init(azim=105, elev=12)
# plt.tight_layout()
plt.show()
