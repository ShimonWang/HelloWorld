import numpy as np
from pytransform3d import rotations as pr

# 姿态1：z轴旋转90度
omega1 = np.array([0, 0, 1])
theta1 = np.deg2rad(90)
q1 = pr.quaternion_from_compact_axis_angle(omega1 * theta1)
Omega1 = pr.compact_axis_angle_from_quaternion(q1)

# 姿态2：x轴旋转90度
omega2 = np.array([1, 0, 0])
theta2 = np.deg2rad(90)
q2 = pr.quaternion_from_compact_axis_angle(omega2 * theta2)
Omega2 = pr.compact_axis_angle_from_quaternion(q2)

# 直接相减（错误的方法）
Omega_diff_direct = Omega2 - Omega1

# 通过积分逐步计算旋转差
num_steps = 100  # 100步积分
dt = 1.0 / num_steps
Omega_integrated = np.zeros(3)

for t in np.linspace(0, 1, num_steps):
    Omega_t = (1 - t) * Omega1 + t * Omega2  # 线性插值
    Omega_integrated += (Omega2 - Omega1) * dt  # 近似积分

# 四元数计算旋转差（正确的方法）
q_diff = pr.concatenate_quaternions(q2, pr.q_conj(q1))
Omega_diff_quat = pr.compact_axis_angle_from_quaternion(q_diff)

# 旋转矩阵计算旋转差
R = pr.matrix_from_quaternion(q2) @ pr.matrix_from_quaternion(q1).T
Omega_diff_matrix = pr.compact_axis_angle_from_matrix(R)

# 打印结果
print(f"直接相减（错误）: {Omega_diff_direct}")
print(f"积分方法（近似正确）: {Omega_integrated}")
print(f"四元数方法（正确）: {Omega_diff_quat}")
print(f"旋转矩阵方法（正确）: {Omega_diff_matrix}")
