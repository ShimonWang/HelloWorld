import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d  # noqa: F401

from pytransform3d.plot_utils import Trajectory
from pytransform3d.rotations import passive_matrix_from_angle, R_id, active_matrix_from_angle
from pytransform3d.transformations import transform_from, concat


def update_trajectory(step, n_frames, trajectory):
    progress = float(step + 1) / float(n_frames)
    H = np.zeros((100, 4, 4))
    H0 = transform_from(R_id, np.zeros(3))  # R_id:3*3单位矩阵
    H_mod = np.eye(4)
    for i, t in enumerate(np.linspace(0, progress, len(H))):
        # print("i：", i, "t:", t)
        H0[:3, 3] = np.array([t, 0, t])
        # H0[:3, 3] = np.array([t, t, t])  # sqrt(2)倍半径 与上面相比
        # H_mod[:3, :3] = passive_matrix_from_angle(2, 8 * np.pi * t)  # 顺时针
        H_mod[:3, :3] = active_matrix_from_angle(2, 8 * np.pi * t)  # 逆时针 课本采用主动旋转
        H[i] = concat(H0, H_mod)

    trajectory.set_data(H)
    return trajectory


if __name__ == "__main__":
    n_frames = 200

    fig = plt.figure(figsize=(5, 5))

    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_zlim((-1, 1))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    H = np.zeros((100, 4, 4))  # H:shape(n_steps,4,4)齐次矩阵的位置姿态序列
    H[:] = np.eye(4)
    trajectory = Trajectory(H, show_direction=True, s=0.2, c="k")  # 一个用于显示轨迹的Matplotlib的绘图对象 show_direction:绘制箭头指示方向
    trajectory.add_trajectory(ax)  # 将轨迹添加到 3D 坐标轴

    anim = animation.FuncAnimation(
        fig,
        update_trajectory,
        n_frames,
        fargs=(n_frames, trajectory),
        interval=50,
        blit=False,
    )

    plt.show()