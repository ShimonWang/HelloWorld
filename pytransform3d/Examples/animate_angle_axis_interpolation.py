"""
==============================================
Interpolate Between Axis-Angle Representations
# 在轴角表示之间进行插值
==============================================

We can interpolate between two orientations that are represented by an axis and
an angle either linearly or with slerp (spherical linear interpolation).
Here we compare both methods and measure the angular
velocity between two successive steps. We can see that linear interpolation
results in a non-constant angular velocity. Usually it is a better idea to
interpolate with slerp.
# 我们可以在由轴和角度表示的两个方向之间进行插值，这些方向可以通过线性插值或 slerp（球面线性插值）来表示。
在这里，我们比较了这两种方法，并测量了连续步骤之间的角速度。
我们可以看到，线性插值会导致角速度非恒定。通常，使用 slerp 进行插值是一个更好的选择。
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d  # noqa: F401

from pytransform3d import rotations as pr

velocity = None
last_a = None
rotation_axis = None


def interpolate_linear(start, end, t):
    return (1 - t) * start + t * end


def update_lines(step, start, end, n_frames, rot, profile):
    global velocity  # 使用 global 可以在函数中修改全局变量。
    global last_a

    if step == 0:
        velocity = []
        last_a = start

    # 生成轴角序列
    if step <= n_frames / 2:  # 前半段球面线性插值
        t = step / float(n_frames / 2 - 1)
        a = pr.axis_angle_slerp(start, end, t)  # 球面线性插值
        # a=pr.axis_angle_slerp(start, end, t):a=(x,y,z,angle)
    else:  # 后半段线性插值
        t = (step - n_frames / 2) / float(n_frames / 2 - 1)  # 线性插值
        a = interpolate_linear(end, start, t)

    # 对应旋转矩阵序列
    R = pr.matrix_from_axis_angle(a)

    # Draw new frame  # 绘制新坐标系
    rot[0].set_data(np.array([0, R[0, 0]]), [0, R[1, 0]])  # set_data:Set the x and y data. 新的x轴 [0,0,0]->R[:,0]
    rot[0].set_3d_properties([0, R[2, 0]])  # Set the z position and direction of the line.

    rot[1].set_data(np.array([0, R[0, 1]]), [0, R[1, 1]])  # 新的y轴 [0,0,0]->R[:,1]
    rot[1].set_3d_properties([0, R[2, 1]])

    rot[2].set_data(np.array([0, R[0, 2]]), [0, R[1, 2]])  # 新的z轴 [0,0,0]->R[:,2]
    rot[2].set_3d_properties([0, R[2, 2]])

    # Update vector in frame  # 更新坐标系中的指示向量，方便查看坐标系方向变化
    test = R.dot(np.ones(3) / np.sqrt(3.0))  # 旋转矩阵R作用在单位向量 1/sqrt(3)[1,1,1]
    # test = np.ones(3) / np.sqrt(3.0)  # 固定向量
    rot[3].set_data(
        np.array([test[0] / 2.0, test[0]]), [test[1] / 2.0, test[1]]
    )
    rot[3].set_3d_properties([test[2] / 2.0, test[2]])

    velocity.append(
        pr.angle_between_vectors(a[:3], last_a[:3]) + a[3] - last_a[3]
    )  # angle=pr.angle_between_vectors(a,b):计算两个向量之间的角度
    print(f"step:{step}")
    print(f"a:{a}")
    print(f"velocity:{velocity}")
    print(f"test:{test}")
    print(f"test/{[test[0] / 2.0, test[0]], [test[1] / 2.0, test[1]], [test[2] / 2.0, test[2]]}")
    last_a = a
    profile.set_data(np.linspace(0, 1, n_frames)[: len(velocity)], velocity)  # (完成度，velocity)

    return rot


if __name__ == "__main__":
    # Generate random start and goal  # 生成随机的起点和终点(轴角表示)
    np.random.seed(3)
    # 起始轴角
    start = np.array([0, 0, 0, np.pi])
    start[:3] = pr.norm_vector(np.random.randn(3))  # 归一化向量 np.random.randn:从正态分布中返回随机数组
    # 结束轴角
    end = np.array([0, 0, 0, np.pi])
    end[:3] = pr.norm_vector(np.random.randn(3))
    n_frames = 100

    # 创建图窗
    fig = plt.figure(figsize=(12, 5))

    # 子图1
    ax = fig.add_subplot(121, projection="3d")
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_zlim((-1, 1))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # 起始和结束姿态(旋转矩阵表示)
    Rs = pr.matrix_from_axis_angle(start)
    Re = pr.matrix_from_axis_angle(end)

    rot = [
        # {world}坐标系
        ax.plot([0, 1], [0, 0], [0, 0], c="r", lw=3)[0],
        ax.plot([0, 0], [0, 1], [0, 0], c="g", lw=3)[0],
        ax.plot([0, 0], [0, 0], [0, 1], c="b", lw=3)[0],
        ax.plot([0, 1], [0, 1], [0, 1], c="gray", lw=3)[0],  # 指示向量，方便查看坐标系方向变化

        # 初始姿态
        ax.plot(
            [0, Rs[0, 0]], [0, Rs[1, 0]], [0, Rs[2, 0]], c="r", lw=3, alpha=0.5
        )[0],
        ax.plot(
            [0, Rs[0, 1]], [0, Rs[1, 1]], [0, Rs[2, 1]], c="g", lw=3, alpha=0.5
        )[0],
        ax.plot(
            [0, Rs[0, 2]], [0, Rs[1, 2]], [0, Rs[2, 2]], c="b", lw=3, alpha=0.5
        )[0],

        # 结束姿态
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
            c="turquoise",  # turquoise:蓝绿色
            lw=3,
            alpha=0.5,
        )[0],
        ax.plot(
            [0, Re[0, 2]],
            [0, Re[1, 2]],
            [0, Re[2, 2]],
            c="violet",  # violet:紫罗兰色
            lw=3,
            alpha=0.5,
        )[0],
    ]

    # 子图2
    ax = fig.add_subplot(122)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    profile = ax.plot(0, 0)[0]  # [0]：ax.plot 返回的是一个列表，其中包含一个 Line2D 对象（因为只绘制了一个点）。
    # [0] 用于从列表中提取第一个（也是唯一的）Line2D 对象。
    # profile, = ax.plot(0, 0)  # 解包列表中的唯一元素
    print(f"profile:,{profile}")
    print(f"ax.plot:,{ax.plot(0, 0)}")

    anim = animation.FuncAnimation(
        fig,
        update_lines,
        n_frames,
        fargs=(start, end, n_frames, rot, profile),
        interval=2000,
        blit=False,
    )

    plt.show()
