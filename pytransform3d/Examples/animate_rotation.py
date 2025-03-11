# pytransform3d.rotations.axis_angle_slerp

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d  # noqa: F401

from pytransform3d import rotations as pr
from pytransform3d.plot_utils import Frame


def update_frame(step, n_frames, frame):
    print("step:", step)
    angle = 2.0 * np.pi * (step + 1) / n_frames
    R = pr.active_matrix_from_angle(0, angle)  # 从基向量旋转计算被动旋转矩阵。 basis int from [0, 1, 2]

    A2B = np.eye(4)
    A2B[:3, :3] = R
    frame.set_data(A2B)  # Set the transformation data.
    return frame


if __name__ == "__main__":
    n_frames = 50

    fig = plt.figure(figsize=(5, 5))

    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_zlim((-1, 1))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # plt.show()

    frame = Frame(np.eye(4), label="rotating frame", s=0.5)  # pytransform3d.plot_utils.Frame(A2B, label=None, s=1.0, **kwargs)
    frame.add_frame(ax)  # add_frame(axis) 添加坐标系到3D坐标轴中

    anim = animation.FuncAnimation(
        fig,
        update_frame,
        n_frames,
        fargs=(n_frames, frame),
        interval=50,  # interval:间隔时间
        blit=False,  # blit:是否根据blitting来优化绘图
    )

    plt.show()