"""
===========================================
Interpolate Between Quaternion Orientations
# 在四元数姿态之间进行插值
===========================================

We can interpolate between two orientations that are represented by quaternions
either linearly or with slerp (spherical linear interpolation).
Here we compare both methods and measure the angular velocity between two
successive steps. We can see that linear interpolation results in a
non-constant angular velocity. Usually it is a better idea to interpolate with
slerp.
# 我们可以在用四元数表示的两个方向之间进行插值，可以是线性的，也可以用slerp (球面线性插值)表示。
在这里，我们比较了这两种方法，并测量了连续两步之间的角速度。我们可以看到，线性插值导致了非常恒定的角速度。
通常用slerp进行插值是一种比较好的思路。
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d  # noqa: F401

from pytransform3d import rotations as pr

# 全局变量
velocity = None  # 用于存储旋转速度
last_R = None  # 用于存储上一帧的旋转矩阵


# 线性插值函数
def interpolate_linear(start, end, t):
    return (1 - t) * start + t * end


# 更新动画帧的函数
def update_lines(step, start, end, n_frames, rot, profile):
    global velocity
    global last_R

    # 初始化
    if step == 0:
        velocity = []
        last_R = pr.matrix_from_quaternion(start)

    print(f"step:{step}")

    # 生成四元数插值序列
    if step <= n_frames / 2 - 1:
        # 前半段：从 start 到 end 进行球面线性插值
        t = step / float(n_frames / 2 - 1)
        q = pr.quaternion_slerp(start, end, t)
        print(f"t:{t}")
        print(f"球面线性插值")
        print(f"q:{q}")
    else:
        # 后半段：从 end 到 start 进行线性插值
        t = (step - n_frames / 2) / float(n_frames / 2 - 1)
        q = interpolate_linear(end, start, t)
        print(f"t:{t}")
        print(f"线性插值")
        print(f"q:{q}")

    # 将四元数转换为旋转矩阵
    R = pr.matrix_from_quaternion(q)

    # Draw new frame  # 更新 3D 图形中的坐标轴
    rot[0].set_data(np.array([0, R[0, 0]]), [0, R[1, 0]])
    rot[0].set_3d_properties([0, R[2, 0]])

    rot[1].set_data(np.array([0, R[0, 1]]), [0, R[1, 1]])
    rot[1].set_3d_properties([0, R[2, 1]])

    rot[2].set_data(np.array([0, R[0, 2]]), [0, R[1, 2]])
    rot[2].set_3d_properties([0, R[2, 2]])

    # Update vector in frame  # 更新坐标系中的指示向量，方便查看坐标系方向变化
    test = R.dot(np.ones(3) / np.sqrt(3.0))  # 计算旋转后的向量
    rot[3].set_data(
        np.array([test[0] / 2.0, test[0]]), [test[1] / 2.0, test[1]]
    )
    rot[3].set_3d_properties([test[2] / 2.0, test[2]])

    # 计算速度并更新速度曲线
    velocity.append(np.linalg.norm(R - last_R))
    last_R = R  # 更新上一帧的旋转矩阵
    profile.set_data(np.linspace(0, 1, n_frames)[: len(velocity)], velocity)

    return rot


if __name__ == "__main__":
    # Generate random start and goal  # 生成随机的起始和目标(四元数)
    np.random.seed(3)  # 设置随机种子以确保结果可重复
    # 起始四元数
    start = np.array([0, 0, 0, np.pi])
    start[:3] = np.random.randn(3)
    start = pr.quaternion_from_axis_angle(start)
    # 目标四元数
    end = np.array([0, 0, 0, np.pi])
    end[:3] = np.random.randn(3)
    end = pr.quaternion_from_axis_angle(end)
    n_frames = 200

    print(f"start:{start}")
    print(f"end:{end}")

    # 创建图形窗口
    fig = plt.figure(figsize=(12, 5))

    # 创建3D子图
    ax = fig.add_subplot(121, projection="3d")
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_zlim((-1, 1))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # 计算起始和目标旋转矩阵
    Rs = pr.matrix_from_quaternion(start)
    Re = pr.matrix_from_quaternion(end)

    # 初始化3D图形中的线条对象
    rot = [
        ax.plot([0, 1], [0, 0], [0, 0], c="r", lw=3)[0],
        ax.plot([0, 0], [0, 1], [0, 0], c="g", lw=3)[0],
        ax.plot([0, 0], [0, 0], [0, 1], c="b", lw=3)[0],
        ax.plot([0, 1], [0, 1], [0, 1], c="gray", lw=3)[0],

        # 起始姿态
        ax.plot(
            [0, Rs[0, 0]], [0, Rs[1, 0]], [0, Rs[2, 0]], c="r", lw=3, alpha=0.5
        )[0],
        ax.plot(
            [0, Rs[0, 1]], [0, Rs[1, 1]], [0, Rs[2, 1]], c="g", lw=3, alpha=0.5
        )[0],
        ax.plot(
            [0, Rs[0, 2]], [0, Rs[1, 2]], [0, Rs[2, 2]], c="b", lw=3, alpha=0.5
        )[0],

        # 目标姿态
        ax.plot(
            [0, Re[0, 0]],
            [0, Re[1, 0]],
            [0, Re[2, 0]],
            c="orange",
            lw=3,
            alpha=0.5,
        )[0],
        ax.plot(
            [0, Re[0, 1]],
            [0, Re[1, 1]],
            [0, Re[2, 1]],
            c="turquoise",
            lw=3,
            alpha=0.5,
        )[0],
        ax.plot(
            [0, Re[0, 2]],
            [0, Re[1, 2]],
            [0, Re[2, 2]],
            c="violet",
            lw=3,
            alpha=0.5,
        )[0],
    ]

    # 创建速度曲线子图
    ax = fig.add_subplot(122)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    profile = ax.plot(0, 0)[0]  # 初始化速度曲线

    # 创建动画
    anim = animation.FuncAnimation(
        fig,
        update_lines,
        n_frames,
        fargs=(start, end, n_frames, rot, profile),
        interval=50,
        blit=False,
        repeat=False
    )

    # 显示动画
    plt.show()
