"""
======================
Quaternion Integration
======================

Integrate angular accelerations to a quaternion sequence and animate it.
# 将角加速度集成到四元数序列并使其动画化
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d  # noqa: F401

from pytransform3d import rotations as pr


def update_lines(step, Q, rot):
    R = pr.matrix_from_quaternion(Q[step])
    # print(f"step:{step}")

    # # Draw new frame
    # rot[0].set_data(np.array([0, R[0, 0]]), [0, R[1, 0]])  # Set the x and y data.
    # rot[0].set_3d_properties([0, R[2, 0]])  # Set the z position and direction of the line.
    #
    # rot[1].set_data(np.array([0, R[0, 1]]), [0, R[1, 1]])
    # rot[1].set_3d_properties([0, R[2, 1]])
    #
    # rot[2].set_data(np.array([0, R[0, 2]]), [0, R[1, 2]])
    # rot[2].set_3d_properties([0, R[2, 2]])
    # Draw new frame
    rot[3].set_data(np.array([0, R[0, 0]]), [0, R[1, 0]])  # Set the x and y data.
    rot[3].set_3d_properties([0, R[2, 0]])  # Set the z position and direction of the line.

    rot[4].set_data(np.array([0, R[0, 1]]), [0, R[1, 1]])
    rot[4].set_3d_properties([0, R[2, 1]])

    rot[5].set_data(np.array([0, R[0, 2]]), [0, R[1, 2]])
    rot[5].set_3d_properties([0, R[2, 2]])

    return rot


if __name__ == "__main__":
    rng = np.random.default_rng(3)  # Generator=np.random.default_rng(seed):随机数Generator
    start = pr.random_quaternion(rng)  # q=pr.random_quaternion(rng):生成随机四元数
    n_frames = 1000
    dt = 0.01
    angular_accelerations = np.empty((n_frames, 3))
    for i in range(n_frames):
        angular_accelerations[i] = pr.random_compact_axis_angle(rng)
        # a=pr.random_compact_axis_angle(rng):生成随机紧凑轴角，角度在[0,pi)中随机采样，旋转轴分量将从N(mu=0,sigma=1)中采样，然后轴将被归一化
    # Integrate angular accelerations to velocities  # 将角加速度积分为速度
    angular_velocities = np.vstack(  # 初始速度为0
        (np.zeros((1, 3)), np.cumsum(angular_accelerations * dt, axis=0))  # 所有数组元素累计求和
    )
    # Integrate angular velocities to quaternions  # 将角速度积分到四元数
    Q = pr.quaternion_integrate(angular_velocities, q0=start, dt=dt)  # Q=pr.quaternion_integrate(Qd,q0,dt):将角速度积分到四元数
    # Qd:角速度以紧凑轴角表示。每个角速度代表一个单位时间后的旋转偏移;q0:初始姿态(w,x,y,z) dt:时间间隔

    fig = plt.figure(figsize=(4, 3))

    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_zlim((-1, 1))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    R = pr.matrix_from_quaternion(start)
    # print(R[0,0])

    rot = [
        # {world}坐标轴 x、y、z
        ax.plot([0, 1], [0, 0], [0, 0], c="r", lw=3)[0],  # x轴 [0,0,0]->[0,0,1]
        ax.plot([0, 0], [0, 1], [0, 0], c="g", lw=3)[0],
        ax.plot([0, 0], [0, 0], [0, 1], c="b", lw=3)[0],

        # 初始的坐标轴
        ax.plot(
            [0, R[0, 0]], [0, R[1, 0]], [0, R[2, 0]], c="r", lw=3, alpha=0.3  # 新的x轴 [0,0,0]->R[:,0]
        )[0],
        ax.plot(
            [0, R[0, 1]], [0, R[1, 1]], [0, R[2, 1]], c="g", lw=3, alpha=0.3  # 新的y轴 [0,0,0]->R[:,1]
        )[0],
        ax.plot(
            [0, R[0, 2]], [0, R[1, 2]], [0, R[2, 2]], c="b", lw=3, alpha=0.3  # 新的z轴 [0,0,0]->R[:,2]
        )[0],
    ]

    anim = animation.FuncAnimation(
        fig, update_lines, n_frames, fargs=(Q, rot), interval=10, blit=False  # frags:额外参数
    )

    plt.show()
