"""
==============
Animate Camera
==============

Animate a camera moving along a circular trajectory while looking at a target.
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from pytransform3d.plot_utils import Frame, Camera, make_3d_axis
from pytransform3d.rotations import matrix_from_euler
from pytransform3d.transformations import transform_from


def update_camera(step, n_frames, camera):
    phi = 2 * np.pi * step / n_frames
    tf = transform_from(  # transform_from(R, p, strict_check=True) 将旋转矩阵和平移转换为变换(齐次变换矩阵)
        matrix_from_euler([-0.5 * np.pi, phi, 0], 0, 1, 2, True),
        -10 * np.array([np.sin(phi), np.cos(phi), 0]),
    )  # e:绕轴i,j,k旋转的角度  i,j,k依次为旋转轴，extrinsic:是否使用extrinsic transformations 否则 intrinsic变换 R:主动旋转矩阵
    camera.set_data(tf)
    return camera


if __name__ == "__main__":
    n_frames = 50

    fig = plt.figure(figsize=(5, 5))
    ax = make_3d_axis(15)  # 生成新的三维坐标轴 make_3d_axis(ax_s, pos=111, unit=None, n_ticks=5) ax_s:缩放新的matplotlib3D坐标轴

    frame = Frame(np.eye(4), label="target", s=3, draw_label_indicator=False)
    # Frame(A2B, label=None, s=1.0, **kwargs) A2B:从A坐标系变换到B坐标系 s:基向量的长度 draw_label_indicator:是否绘制从坐标系原点到坐标系标签的线条
    frame.add_frame(ax)

    fl = 3000  # [pixels]
    w, h = 1920, 1080  # [pixels]
    M = np.array(((fl, 0, w // 2), (0, fl, h // 2), (0, 0, 1)))
    camera = Camera(
        M,
        np.eye(4),
        virtual_image_distance=5,
        sensor_size=(w, h),
        c="c",
    )  # Camera(M, cam2world, virtual_image_distance=1.0, sensor_size=(1920, 1080), **kwargs)
    # M:本征相机矩阵 cam2world:相机到世界坐标系的变换 virtual_image_distance:虚拟图像距离 sensor_size:图像传感器尺度
    camera.add_camera(ax)  # 将相机添加到3D坐标轴中

    anim = animation.FuncAnimation(
        fig,
        update_camera,
        n_frames,
        fargs=(n_frames, camera),
        interval=1000,
        blit=False,
        repeat=False
    )

    plt.show()
