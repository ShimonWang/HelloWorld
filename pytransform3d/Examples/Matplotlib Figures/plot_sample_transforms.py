"""
=================
Sample Transforms
=================
"""

import matplotlib.pyplot as plt
import numpy as np

import pytransform3d.transformations as pt

mean = pt.transform_from(R=np.eye(3), p=np.array([0.0, 0.0, 0.5]))
cov = np.diag([0.001, 0.001, 0.5, 0.001, 0.001, 0.001])
cov = np.diag([0.001, 0, 0, 0, 0.001, 0.001])
rng = np.random.default_rng(0)  # 构造一个新的生成器 Generator ，使用默认的的位生成器 BitGenerator (PCG64)
ax = None
poses = np.array(
    [pt.random_transform(rng=rng, mean=mean, cov=cov) for _ in range(1000)]  # 生成随机变换 rng:随机数生成器；
    # cov:协方差，在指数坐标系中噪声的协方差
)
for pose in poses:
    ax = pt.plot_transform(ax=ax, A2B=pose, s=0.3)
plt.show()
