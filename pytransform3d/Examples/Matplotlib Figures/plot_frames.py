"""
===============================================
Plot with Respect to Different Reference Frames
# 关于不同参考系下的绘图
===============================================

In this example, we will demonstrate how to use the TransformManager.
We will add several transforms to the manager and plot all frames in
two reference frames ('world' and 'A').
# 在这个示例中，我们将演示如何使用 TransformManager。我们将向管理器添加几个变换，
并在两个参考系（“world”和“A”）中绘制所有坐标系。
"""

import matplotlib.pyplot as plt
import numpy as np

from pytransform3d.plot_utils import make_3d_axis
from pytransform3d.transform_manager import TransformManager
from pytransform3d.transformations import random_transform

rng = np.random.default_rng(5)
A2world = random_transform(rng)
B2world = random_transform(rng)
A2C = random_transform(rng)
D2B = random_transform(rng)

tm = TransformManager()  # 管理坐标系之间的变换
tm.add_transform("A", "world", A2world)  # tm.add_transform:注册一个变换
tm.add_transform("B", "world", B2world)
tm.add_transform("A", "C", A2C)
tm.add_transform("D", "B", D2B)

plt.figure(figsize=(10, 5))

ax = make_3d_axis(2, 121)
ax = tm.plot_frames_in("world", ax=ax, alpha=0.6)  # 绘制给定参考系的所有坐标系
ax.view_init(30, 20)

ax = make_3d_axis(3, 122)
ax = tm.plot_frames_in("A", ax=ax, alpha=0.6)
ax.view_init(30, 20)

plt.show()
