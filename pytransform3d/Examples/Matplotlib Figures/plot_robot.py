"""
=====
Robot
=====

We see a 6-DOF robot arm with visuals.
# 我们看到了一个带有视觉效果的 6 自由度机械臂。
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from pytransform3d.urdf import UrdfTransformManager

BASE_DIR = "test/test_data/"
data_dir = BASE_DIR
search_path = "."
print(f"os.path.exists(data_dir):{os.path.exists(data_dir)}")
print(f"os.path.dirname(search_path),{os.path.dirname(search_path)}")
while (
    not os.path.exists(data_dir)  # 如果路径 path 存在，返回 True；如果路径 path 不存在或损坏，返回 False。
    and os.path.dirname(search_path) != "pytransform3d"  # 返回文件路径
):
    search_path = os.path.join(search_path, "..")  # 把目录和文件名合成一个路径
    data_dir = os.path.join(search_path, BASE_DIR)
    print(f"os.path.exists(data_dir):{os.path.exists(data_dir)}")
    print(f"os.path.dirname(search_path),{os.path.dirname(search_path)}")

tm = UrdfTransformManager()
filename = os.path.join(data_dir, "robot_with_visuals.urdf")
with open(filename, "r") as f:
    robot_urdf = f.read()
    tm.load_urdf(robot_urdf, mesh_path=data_dir)  # 将 URDF 文件加载到变换管理器中
tm.set_joint("joint2", 0.2 * np.pi)  # tm.set_joint:设置关节位置
tm.set_joint("joint3", 0.2 * np.pi)
tm.set_joint("joint5", 0.1 * np.pi)
tm.set_joint("joint6", 0.8 * np.pi)

tm.plot_visuals("robot_arm", ax_s=0.6, alpha=0.7)  # 绘制给定参考系的所有视觉元素
# tm.plot_visuals("robot_arm", ax_s=0.6, wireframe=True, alpha=0.7)  # 绘制给定参考系的所有视觉元素
plt.show()
