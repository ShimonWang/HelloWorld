import numpy as np
import numpy.matlib
import matplotlib.path as mpath
from matplotlib.patches import PathPatch
import matplotlib.patches as patches
import scipy.linalg
from scipy.linalg import fractional_matrix_power

# Logarithmic map for R^2 x S^1 manifold
def logmap(f, f0):
    position_error = f[:, :2] - f0[:, :2]
    orientation_error = np.imag(np.log(np.exp(f0[:, -1]*1j).conj().T *
                                       np.exp(f[:, -1]*1j).T)).conj().reshape((-1,1))
    error = np.hstack((position_error, orientation_error))
    return error


# Forward kinematics for E-E
def fkin(param, x):
    x = x.T
    A = np.tril(np.ones([param.nbVarX, param.nbVarX]))
    f = np.vstack((param.linkLengths @ np.cos(A @ x), 
                   param.linkLengths @ np.sin(A @ x),
                   np.mod(np.sum(x, 0)+np.pi, 2*np.pi) - np.pi))  #x1,x2,o (orientation as single Euler angle for planar robot)
    return f.T

# Forward Kinematics for all joints
def fkin0(param, x):
    T = np.tril(np.ones([param.nbVarX, param.nbVarX]))
    T2 = np.tril(np.matlib.repmat(param.linkLengths, len(x), 1))
    f = np.vstack((
        T2 @ np.cos(T@x),
        T2 @ np.sin(T@x)
    )).T
    # f = np.vstack((
    #     np.zeros(2),
    #     f
    # ))
    return f

def fkin_ext_obj(x, l):
    x = x.T
    A = np.tril(np.ones([np.size(l, 0), np.size(l, 0)]))
    f = np.vstack((l @ np.cos(A @ x),
                   l @ np.sin(A @ x),
                   np.mod(np.sum(x, 0)+np.pi, 2*np.pi) - np.pi))  #x1,x2,o (orientation as single Euler angle for planar robot)
    return f.T

# Jacobian with analytical computation (for single time step)
def Jkin(param, x):
    T = np.tril(np.ones(np.size(x, 0)))
    J = np.vstack((
        -np.sin(T@x).T @ np.diag(param.linkLengths) @ T,
        np.cos(T@x).T @ np.diag(param.linkLengths) @ T,
        np.ones((1, np.size(x, 0)))
    ))
    return J

def Jkin_geo_obj(param, x, l, Jc):
    J = Jkin_ext_obj(x, l)
    N = np.identity(param.nbVarX+param.obj_num) - np.linalg.pinv(Jc) @ Jc
    J = J @ N
    return J
def Jkin_geo(param, x, Jc):
    J = Jkin_ext(param, x)
    N = np.identity(param.nbVarX) - np.linalg.pinv(Jc) @ Jc
    J = J @ N
    return J

def Jkin0(param, x):
    T = np.tril(np.ones(np.size(x, 0)))
    J = np.vstack((
        -np.sin(T@x).T @ np.diag(param.linkLengths[:np.size(x, 0)]) @ T,
        np.cos(T@x).T @ np.diag(param.linkLengths[:np.size(x, 0)]) @ T,
        np.ones((1, np.size(x, 0)))
    ))
    return J

# Forward kinematics (for end-effector)
def fkin_ext(param, x):
    f_end = fkin(param, x)  # input: [i, param.nbVarX] output: [i, x+o]
    f = np.zeros((np.size(x, 0), 3))
    for i in range(np.size(x, 0)):
        T_end = get_homogenouse_transformation_planar(f_end[i, :2], f_end[i, -1])  # in world
        T_tip = T_end @ param.T_tipINee
        f[i, :] = get_pose_from_homogenouse_transformation_planar(T_tip)
    return f  # tip in world
# Jacobian with numerical computation
def Jkin_ext_obj_num(param, x):
    eps = 1e-6
    # Matrix computation
    X = np.tile(x.reshape((-1, 1)), [1, param.nbVarX])
    F1 = fkin_ext(param, X.T)
    F2 = fkin_ext(param, X.T+np.identity(param.nbVarX)*eps)
    J = (F2-F1) / eps
    return J
def Jkin_ext(param, x):
    T = np.tril(np.ones(np.size(x, 0)))
    J = np.vstack((
        -np.sin(T@x).T @ np.diag(param.linkLengths) @ T,
        np.cos(T@x).T @ np.diag(param.linkLengths) @ T,
        np.ones((1, np.size(x, 0)))
    ))
    J[:, -1] = 0
    return J

def Jkin_ext_obj(x, l):
    T = np.tril(np.ones(np.size(x, 0)))
    J = np.vstack((
        -np.sin(T@x).T @ np.diag(l) @ T,
        np.cos(T@x).T @ np.diag(l) @ T,
        np.ones((1, np.size(x, 0)))
    ))
    J[:, -1] = 0
    return J
def get_homogenouse_transformation_planar(x, o):
    A = np.asarray([
        [np.cos(o), -np.sin(o)],
        [np.sin(o), np.cos(o)]
    ])
    T = np.vstack((np.hstack((A, x.reshape(-1, 1))), np.array([0, 0, 1])))
    return T
def get_pose_from_homogenouse_transformation_planar(T):
    x = T[:2, -1]
    o = np.arctan2(T[1, 0], T[0, 0])
    return np.asarray([*x, o])

# # Jacobian with analytical computation (for single time step)
# def Jkin0(param, x):
#     T = np.tril(np.ones(np.size(x, 0)))
#     J = np.vstack((
#         -np.sin(T@x).T @ np.diag(param.linkLengths) @ T,
#         np.cos(T@x).T @ np.diag(param.linkLengths) @ T,
#         np.ones((1, np.size(x, 0)))
#     ))
#     J = scipy.linalg.block_diag(J[:2], J[:2])
#     return J

# Jacobian with analytical computation (for single time step)
# def Jkin0(param, x):
#     T = np.tril(np.ones(np.size(x, 0)))
#     J = np.vstack((
#         -np.sin(T@x).T @ np.diag(param.linkLengths) @ T,
#         np.cos(T@x).T @ np.diag(param.linkLengths) @ T,
#         np.ones((1, np.size(x, 0)))
#     ))
#     return J[:2]

def Jkin0_r(param, x):
    T = np.tril(np.ones(np.size(x, 0)))
    J = np.vstack((
        -np.sin(T@x).T @ np.diag(param.linkLengths) @ T,
        np.cos(T@x).T @ np.diag(param.linkLengths) @ T,
        np.ones((1, np.size(x, 0)))
    ))
    J = scipy.linalg.block_diag(J[:2], J[:2])
    return J

# Residual and Jacobian
def f_reach(param, x, bounding_boxes=True):
    f = logmap(fkin(param, x), param.mu)
    J = np.zeros((len(x) * param.nbVarF, len(x) * param.nbVarX))

    for t in range(x.shape[0]):
        f[t, :2] = param.A[t].T @ f[t, :2]  # Object oriented fk
        Jtmp = Jkin(param, x[t])

        Jtmp[:2] = param.A[t].T @ Jtmp[:2]  # Object centered jac`obian
        # print(f)
        if bounding_boxes:
            for i in range(2):
                if f[t, i] > param.sizeObj_range[0, i] and f[t, i] < param.sizeObj_range[1, i]:
                    f[t, i] = 0
                    Jtmp[i, :] = 0
                elif f[t, i] < param.sizeObj_range[0, i]:
                    f[t, i] = f[t, i] - param.sizeObj_range[0, i]
                else:
                    f[t, i] = f[t, i] - param.sizeObj_range[1, i]
        J[t*param.nbVarF:(t+1)*param.nbVarF, t*param.nbVarX:(t+1)*param.nbVarX] = Jtmp

        # Bounding angle
        if bounding_boxes and t > 0:
            f[t, 2] = np.imag(np.log(np.exp(f[0, 2]*1j).conj().T *
                                       np.exp(f[t, 2]*1j).T)).conj().reshape((-1, 1))

    return f, J

# Residual and Jacobian
def f_reach_no_o_c(param, x, bounding_boxes=True):
    f = logmap(fkin(param, x), param.mu)
    J = np.zeros((len(x) * param.nbVarF, len(x) * param.nbVarX))

    for t in range(x.shape[0]):
        f[t, :2] = param.A[t].T @ f[t, :2]  # Object oriented fk
        Jtmp = Jkin(param, x[t])

        Jtmp[:2] = param.A[t].T @ Jtmp[:2]  # Object centered jac`obian
        # print(f)
        if bounding_boxes:
            for i in range(2):
                # if abs(f[t, i]) < param.sizeObj[i]:
                #     f[t, i] = 0
                #     Jtmp[i] = 0
                # else:
                #     f[t, i] -= np.sign(f[t, i]) * param.sizeObj[i]
                if f[t, i] > param.sizeObj_range[0, i] and f[t, i] < param.sizeObj_range[1, i]:
                    f[t, i] = 0
                    Jtmp[i, :] = 0
                elif f[t, i] < param.sizeObj_range[0, i]:
                    f[t, i] = f[t, i] - param.sizeObj_range[0, i]
                else:
                    f[t, i] = f[t, i] - param.sizeObj_range[1, i]
        J[t*param.nbVarF:(t+1)*param.nbVarF, t*param.nbVarX:(t+1)*param.nbVarX] = Jtmp

    return f, J


# Residual and Jacobian
def f_reach_no_c(param, x, bounding_boxes=True):
    f = logmap(fkin(param, x), param.mu)
    J = np.zeros((len(x) * param.nbVarF, len(x) * param.nbVarX))

    for t in range(x.shape[0]):
        f[t, :2] = param.A[t].T @ f[t, :2]  # Object oriented fk
        Jtmp = Jkin(param, x[t])

        Jtmp[:2] = param.A[t].T @ Jtmp[:2]  # Object centered jac`obian
        # print(f)
        J[t*param.nbVarF:(t+1)*param.nbVarF, t*param.nbVarX:(t+1)*param.nbVarX] = Jtmp

    return f, J

def rman(param, x):
    G = fractional_matrix_power(param.MuS, -0.5)
    f = np.zeros((3, np.size(x, 1)))
    for i in range(np.size(x, 1)):
        Jt = Jkin(param, x[:, i])[:2]  # Jacobian for center of mass
        St = np.linalg.pinv(Jt @ Jt.T)  # manipulability matrix
        D, V = np.linalg.eig(G @ St @ G)
        E = V @ np.diag(np.log(D)) @ np.linalg.pinv(V)

        E = np.tril(E) * (np.eye(2) + np.tril(np.ones(2), -1) * np.sqrt(2))
        f[:, i] = E[np.where(E != 0)]
    return f


# Jacobian for manipulability tracking with numerical computation
def Jman_num(param, x):
    e = 1E-6
    X = np.matlib.repmat(x, 1, param.nbVarX)
    F1 = rman(param, X)
    F2 = rman(param, X + np.eye(param.nbVarX) * e)
    J = (F2 - F1) / e
    return J


# Residuals f and Jacobians J for manipulability tracking
# (c=f'*f is the cost, g=J'*f is the gradient, H=J'*J is the approximated Hessian)
def f_manipulability(param, x):
    f = rman(param, x)  # Residuals
    for t in range(np.size(x, 1)):
        if t == 0:
            J = Jman_num(param, x[:, t].reshape(-1, 1))
        else:
            J = scipy.linalg.block_diag(J, Jman_num(param, x[:, t].reshape(-1, 1)))  # Jacobians
    return f, J


def get_force_mani_ellipse(x, param):
    J = Jkin(param, x)[:2, :]
    S = np.linalg.pinv(J @ J.T)
    return S


def plot_mani_ellipse(ax, x, param, type='F', scale=1, axes=False, facecolor=None, edgecolor=None, alpha=None, zorder=None, base=False, **kwargs):
    # Plot robot manipulability ellipsoid input the joint state
    if edgecolor is None:
        edgecolor = [0.3, 0.3, 0.3]
    if facecolor is None:
        facecolor = [0.4, 0.4, 0.4]
    if alpha is None:
        alpha = 0.9
    if zorder is None:
        zorder = 1
    fc = fkin(param, x)  # array shape 1x3
    # print(np.shape(fc))
    al = np.linspace(-np.pi, np.pi, 50)

    J = Jkin(param, x)[:2, :]
    if type == 'F':
        S = np.linalg.pinv(J @ J.T)
    elif type == 'V':
        S = J @ J.T
    # print(S)
    D, V = np.linalg.eig(scale * S)
    max_idx = np.where(np.abs(D) == max(np.abs(D)))[0][0]
    min_idx = np.where(np.abs(D) == min(np.abs(D)))[0][0]
    D = D[[max_idx, min_idx]]
    V = V[:, [max_idx, min_idx]]
    D = np.diag(D)
    R = np.real(V @ np.sqrt(D+0j))
    if base == False:
        msh = (R @ np.array([np.cos(al), np.sin(al)])).T + np.matlib.repmat(fc[:, :-1].reshape(-1, 1), 1, 50).T
    else:
        msh = (R @ np.array([np.cos(al), np.sin(al)])).T

    if axes == True:
        ax.annotate("", xy=fc[0, :2] + R[:, 0], xycoords='data', xytext=fc[0, :2] , textcoords='data',
                    arrowprops=dict(arrowstyle="->", linestyle="--", connectionstyle="arc3", color='r'))  # 画x轴
        ax.annotate("", xy=fc[0, :2] + R[:, 1], xycoords='data', xytext=fc[0, :2] , textcoords='data',
                    arrowprops=dict(arrowstyle="->", linestyle="--", connectionstyle="arc3", color='b'))  # 画x轴

    p = patches.Polygon(msh, closed=True, alpha=alpha, zorder=zorder, **kwargs)
    p.set_facecolor(facecolor)
    p.set_edgecolor(edgecolor)
    ax.add_patch(p)
    return S

# plot the joint you want
def plot_mani_ellipse0(ax, x, param, Joint_id=3 , type='F', scale=1, axes=False, facecolor=None, edgecolor=None, alpha=None, zorder=None, base=False):
    # Plot robot manipulability ellipsoid input the joint state
    if edgecolor is None:
        edgecolor = [0.3, 0.3, 0.3]
    if facecolor is None:
        facecolor = [0.4, 0.4, 0.4]
    if alpha is None:
        alpha = 0.9
    if zorder is None:
        zorder = 1
    fc = fkin0(param, x)[Joint_id, :]  # array shape 1x3
    # print(np.shape(fc))
    al = np.linspace(-np.pi, np.pi, 50)

    J = Jkin0(param, x[:Joint_id+1])[:2, :]
    if type == 'F':
        S = np.linalg.pinv(J @ J.T)
    elif type == 'V':
        S = J @ J.T
    # print(S)
    D, V = np.linalg.eig(scale * S)
    max_idx = np.where(np.abs(D) == max(np.abs(D)))[0][0]
    min_idx = np.where(np.abs(D) == min(np.abs(D)))[0][0]
    D = D[[max_idx, min_idx]]
    V = V[:, [max_idx, min_idx]]
    D = np.diag(D)
    R = np.real(V @ np.sqrt(D+0j))
    if base == False:
        msh = (R @ np.array([np.cos(al), np.sin(al)])).T + np.matlib.repmat(fc.reshape(-1, 1), 1, 50).T
    else:
        msh = (R @ np.array([np.cos(al), np.sin(al)])).T

    if axes == True:
        ax.annotate("", xy=fc[0, :2] + R[:, 0], xycoords='data', xytext=fc[0, :2] , textcoords='data',
                    arrowprops=dict(arrowstyle="->", linestyle="--", connectionstyle="arc3", color='r'))  # 画x轴
        ax.annotate("", xy=fc[0, :2] + R[:, 1], xycoords='data', xytext=fc[0, :2] , textcoords='data',
                    arrowprops=dict(arrowstyle="->", linestyle="--", connectionstyle="arc3", color='b'))  # 画x轴

    p = patches.Polygon(msh, closed=True, alpha=alpha, zorder=zorder)
    p.set_facecolor(facecolor)
    p.set_edgecolor(edgecolor)
    ax.add_patch(p)
    return S

def plot_mani_ellipse_ext(ax, x, param, type='F', scale=1, axes=False, facecolor=None, edgecolor=None, alpha=None, zorder=None, base=False):
    # Plot robot manipulability ellipsoid input the joint state
    if edgecolor is None:
        edgecolor = [0.3, 0.3, 0.3]
    if facecolor is None:
        facecolor = [0.4, 0.4, 0.4]
    if alpha is None:
        alpha = 0.9
    if zorder is None:
        zorder = 1
    x = x.reshape(1, -1)
    fc = fkin(param, x)
    # print(np.shape(fc))
    al = np.linspace(-np.pi, np.pi, 50)

    J = Jkin_ext(param, x.flatten())[:2, :]
    if type == 'F':
        S = np.linalg.pinv(J @ J.T)
    elif type == 'V':
        S = J @ J.T
    # print(S)
    D, V = np.linalg.eig(scale * S)
    max_idx = np.where(np.abs(D) == max(np.abs(D)))[0][0]
    min_idx = np.where(np.abs(D) == min(np.abs(D)))[0][0]
    D = D[[max_idx, min_idx]]
    V = V[:, [max_idx, min_idx]]
    D = np.diag(D)
    R = np.real(V @ np.sqrt(D+0j))
    if base == False:
        msh = (R @ np.array([np.cos(al), np.sin(al)])).T + np.matlib.repmat(fc[:, :-1].reshape(-1, 1), 1, 50).T
    else:
        msh = (R @ np.array([np.cos(al), np.sin(al)])).T

    if axes == True:
        ax.annotate("", xy=fc[0, :2] + R[:, 0], xycoords='data', xytext=fc[0, :2] , textcoords='data',
                    arrowprops=dict(arrowstyle="->", linestyle="--", connectionstyle="arc3", color='r'))  # 画x轴
        ax.annotate("", xy=fc[0, :2] + R[:, 1], xycoords='data', xytext=fc[0, :2] , textcoords='data',
                    arrowprops=dict(arrowstyle="->", linestyle="--", connectionstyle="arc3", color='b'))  # 画x轴

    p = patches.Polygon(msh, closed=True, alpha=alpha, zorder=zorder)
    p.set_facecolor(facecolor)
    p.set_edgecolor(edgecolor)
    ax.add_patch(p)
    return S

def plot_mani_ellipse_ext_J(ax, x, param, type='F', scale=1, axes=False, facecolor=None, edgecolor=None, alpha=None, zorder=None, base=False):
    # Plot robot manipulability ellipsoid input the joint state
    if edgecolor is None:
        edgecolor = [0.3, 0.3, 0.3]
    if facecolor is None:
        facecolor = [0.4, 0.4, 0.4]
    if alpha is None:
        alpha = 0.9
    if zorder is None:
        zorder = 1
    x = x.reshape(1, -1)
    fc = fkin(param, x)
    # print(np.shape(fc))
    al = np.linspace(-np.pi, np.pi, 50)

    J = Jkin_ext(param, x.flatten())[:2, :]
    if type == 'F':
        S = np.linalg.pinv(J @ J.T)
    elif type == 'V':
        S = J @ J.T
    # print(S)
    D, V = np.linalg.eig(scale * S)
    max_idx = np.where(np.abs(D) == max(np.abs(D)))[0][0]
    min_idx = np.where(np.abs(D) == min(np.abs(D)))[0][0]
    D = D[[max_idx, min_idx]]
    V = V[:, [max_idx, min_idx]]
    D = np.diag(D)
    R = np.real(V @ np.sqrt(D+0j))
    if base == False:
        msh = (R @ np.array([np.cos(al), np.sin(al)])).T + np.matlib.repmat(fc[:, :-1].reshape(-1, 1), 1, 50).T
    else:
        msh = (R @ np.array([np.cos(al), np.sin(al)])).T

    if axes == True:
        ax.annotate("", xy=fc[0, :2] + R[:, 0], xycoords='data', xytext=fc[0, :2] , textcoords='data',
                    arrowprops=dict(arrowstyle="->", linestyle="--", connectionstyle="arc3", color='r'))  # 画x轴
        ax.annotate("", xy=fc[0, :2] + R[:, 1], xycoords='data', xytext=fc[0, :2] , textcoords='data',
                    arrowprops=dict(arrowstyle="->", linestyle="--", connectionstyle="arc3", color='b'))  # 画x轴

    p = patches.Polygon(msh, closed=True, alpha=alpha, zorder=zorder)
    p.set_facecolor(facecolor)
    p.set_edgecolor(edgecolor)
    ax.add_patch(p)
    return J

def plot_mani_ellipse_ext_obj(ax, x, param, type='F', scale=1, axes=False, facecolor=None, edgecolor=None, alpha=None, zorder=None, base=False, center=False):
    # Plot robot manipulability ellipsoid input the joint state
    if edgecolor is None:
        edgecolor = [0.3, 0.3, 0.3]
    if facecolor is None:
        facecolor = [0.4, 0.4, 0.4]
    if alpha is None:
        alpha = 0.9
    if zorder is None:
        zorder = 1
    x = x.reshape(1, -1)

    l_ext = np.linalg.norm(param.T_tipINee[:2, -1].copy())
    l = np.array([*param.linkLengths, l_ext])
    theta_ext = np.arctan2(param.T_tipINee[1, 0], param.T_tipINee[0, 0])
    theta_ext_bias = np.arctan2((param.object_tip_local[1] - param.T_eeINobj[1, -1]),
                                (param.object_tip_local[0] - param.T_eeINobj[0, -1]))
    theta_ext = theta_ext + theta_ext_bias
    # theta_ext = np.arctan2(param.T_bias[1, -1], param.T_bias[0, -1])
    x_ext = np.array([*x.flatten(), theta_ext])
    fc = fkin_ext_obj(x_ext, l)
    # print(np.shape(fc))
    al = np.linspace(-np.pi, np.pi, 50)

    J = Jkin_ext_obj(x_ext.flatten(), l)[:2, :]
    if type == 'F':
        S = np.linalg.pinv(J @ J.T)
    elif type == 'V':
        S = J @ J.T
    # print(S)
    D, V = np.linalg.eig(scale * S)
    max_idx = np.where(np.abs(D) == max(np.abs(D)))[0][0]
    min_idx = np.where(np.abs(D) == min(np.abs(D)))[0][0]
    D = D[[max_idx, min_idx]]
    V = V[:, [max_idx, min_idx]]
    D = np.diag(D)
    R = np.real(V @ np.sqrt(D+0j))
    if base == False:
        msh = (R @ np.array([np.cos(al), np.sin(al)])).T + np.matlib.repmat(fc[:, :-1].reshape(-1, 1), 1, 50).T
    else:
        msh = (R @ np.array([np.cos(al), np.sin(al)])).T

    if axes == True:
        ax.annotate("", xy=fc[0, :2] + R[:, 0], xycoords='data', xytext=fc[0, :2] , textcoords='data',
                    arrowprops=dict(arrowstyle="->", linestyle="--", connectionstyle="arc3", color='r'))  # 画x轴
        ax.annotate("", xy=fc[0, :2] + R[:, 1], xycoords='data', xytext=fc[0, :2] , textcoords='data',
                    arrowprops=dict(arrowstyle="->", linestyle="--", connectionstyle="arc3", color='b'))  # 画x轴
    if center == True:
        ax.scatter(fc[0, 0], fc[0, 1], color='r', marker='o')

    p = patches.Polygon(msh, closed=True, alpha=alpha, zorder=zorder)
    p.set_facecolor(facecolor)
    p.set_edgecolor(edgecolor)
    ax.add_patch(p)
    return S

def plot_mani_ellipse_ext_obj0(ax, x, param, type='F', scale=1, axes=False, facecolor=None, edgecolor=None, alpha=None, zorder=None, base=False, center=False):
    # Plot robot manipulability ellipsoid input the joint state
    if edgecolor is None:
        edgecolor = [0.3, 0.3, 0.3]
    if facecolor is None:
        facecolor = [0.4, 0.4, 0.4]
    if alpha is None:
        alpha = 0.9
    if zorder is None:
        zorder = 1
    x = x.reshape(1, -1)

    l_ext = np.linalg.norm(param.T_tipINee0[:2, -1].copy())
    l = np.array([*param.linkLengths, l_ext])
    theta_ext = np.arctan2(param.T_tipINee0[1, 0], param.T_tipINee0[0, 0])
    theta_ext_bias = np.arctan2((param.object_tip_local0[1] - param.T_eeINobj0[1, -1]),
                                (param.object_tip_local0[0] - param.T_eeINobj0[0, -1]))
    theta_ext = theta_ext + theta_ext_bias
    # theta_ext = np.arctan2(param.T_bias[1, -1], param.T_bias[0, -1])
    x_ext = np.array([*x.flatten(), theta_ext])
    fc = fkin_ext_obj(x_ext, l)
    # print(np.shape(fc))
    al = np.linspace(-np.pi, np.pi, 50)

    J = Jkin_ext_obj(x_ext.flatten(), l)[:2, :]
    if type == 'F':
        S = np.linalg.pinv(J @ J.T)
    elif type == 'V':
        S = J @ J.T
    # print(S)
    D, V = np.linalg.eig(scale * S)
    max_idx = np.where(np.abs(D) == max(np.abs(D)))[0][0]
    min_idx = np.where(np.abs(D) == min(np.abs(D)))[0][0]
    D = D[[max_idx, min_idx]]
    V = V[:, [max_idx, min_idx]]
    D = np.diag(D)
    R = np.real(V @ np.sqrt(D+0j))
    if base == False:
        msh = (R @ np.array([np.cos(al), np.sin(al)])).T + np.matlib.repmat(fc[:, :-1].reshape(-1, 1), 1, 50).T
    else:
        msh = (R @ np.array([np.cos(al), np.sin(al)])).T

    if axes == True:
        ax.annotate("", xy=fc[0, :2] + R[:, 0], xycoords='data', xytext=fc[0, :2] , textcoords='data',
                    arrowprops=dict(arrowstyle="->", linestyle="--", connectionstyle="arc3", color='r'))  # 画x轴
        ax.annotate("", xy=fc[0, :2] + R[:, 1], xycoords='data', xytext=fc[0, :2] , textcoords='data',
                    arrowprops=dict(arrowstyle="->", linestyle="--", connectionstyle="arc3", color='b'))  # 画x轴
    if center == True:
        ax.scatter(fc[0, 0], fc[0, 1], color='r', marker='o')

    p = patches.Polygon(msh, closed=True, alpha=alpha, zorder=zorder)
    p.set_facecolor(facecolor)
    p.set_edgecolor(edgecolor)
    ax.add_patch(p)
    return S
def plot_mani_ellipse_ext_obj1(ax, x, param, type='F', scale=1, axes=False, facecolor=None, edgecolor=None, alpha=None, zorder=None, base=False, center=False):
    # Plot robot manipulability ellipsoid input the joint state
    if edgecolor is None:
        edgecolor = [0.3, 0.3, 0.3]
    if facecolor is None:
        facecolor = [0.4, 0.4, 0.4]
    if alpha is None:
        alpha = 0.9
    if zorder is None:
        zorder = 1
    x = x.reshape(1, -1)

    l_ext = np.linalg.norm(param.T_tipINee1[:2, -1].copy())
    l = np.array([*param.linkLengths, l_ext])
    theta_ext = np.arctan2(param.T_tipINee1[1, 0], param.T_tipINee1[0, 0])
    theta_ext_bias = np.arctan2((param.object_tip_local1[1] - param.T_eeINobj1[1, -1]),
                                (param.object_tip_local1[0] - param.T_eeINobj1[0, -1]))
    theta_ext = theta_ext + theta_ext_bias
    # theta_ext = np.arctan2(param.T_bias[1, -1], param.T_bias[0, -1])
    x_ext = np.array([*x.flatten(), theta_ext])
    fc = fkin_ext_obj(x_ext, l)
    # print(np.shape(fc))
    al = np.linspace(-np.pi, np.pi, 50)

    J = Jkin_ext_obj(x_ext.flatten(), l)[:2, :]
    if type == 'F':
        S = np.linalg.pinv(J @ J.T)
    elif type == 'V':
        S = J @ J.T
    # print(S)
    D, V = np.linalg.eig(scale * S)
    max_idx = np.where(np.abs(D) == max(np.abs(D)))[0][0]
    min_idx = np.where(np.abs(D) == min(np.abs(D)))[0][0]
    D = D[[max_idx, min_idx]]
    V = V[:, [max_idx, min_idx]]
    D = np.diag(D)
    R = np.real(V @ np.sqrt(D+0j))
    if base == False:
        msh = (R @ np.array([np.cos(al), np.sin(al)])).T + np.matlib.repmat(fc[:, :-1].reshape(-1, 1), 1, 50).T
    else:
        msh = (R @ np.array([np.cos(al), np.sin(al)])).T

    if axes == True:
        ax.annotate("", xy=fc[0, :2] + R[:, 0], xycoords='data', xytext=fc[0, :2] , textcoords='data',
                    arrowprops=dict(arrowstyle="->", linestyle="--", connectionstyle="arc3", color='r'))  # 画x轴
        ax.annotate("", xy=fc[0, :2] + R[:, 1], xycoords='data', xytext=fc[0, :2] , textcoords='data',
                    arrowprops=dict(arrowstyle="->", linestyle="--", connectionstyle="arc3", color='b'))  # 画x轴
    if center == True:
        ax.scatter(fc[0, 0], fc[0, 1], color='r', marker='o')

    p = patches.Polygon(msh, closed=True, alpha=alpha, zorder=zorder)
    p.set_facecolor(facecolor)
    p.set_edgecolor(edgecolor)
    ax.add_patch(p)
    return S

def plot_mani_ellipse_ext_obj2(ax, x, param, type='F', scale=1, axes=False, facecolor=None, edgecolor=None, alpha=None, zorder=None, base=False, center=False):
    # Plot robot manipulability ellipsoid input the joint state
    if edgecolor is None:
        edgecolor = [0.3, 0.3, 0.3]
    if facecolor is None:
        facecolor = [0.4, 0.4, 0.4]
    if alpha is None:
        alpha = 0.9
    if zorder is None:
        zorder = 1
    x = x.reshape(1, -1)

    l_ext = np.linalg.norm(param.T_tipINee2[:2, -1].copy())
    l = np.array([*param.linkLengths, l_ext])
    theta_ext = np.arctan2(param.T_tipINee2[1, 0], param.T_tipINee2[0, 0])
    theta_ext_bias = np.arctan2((param.object_tip_local2[1] - param.T_eeINobj2[1, -1]),
                                (param.object_tip_local2[0] - param.T_eeINobj2[0, -1]))
    theta_ext = theta_ext + theta_ext_bias
    # theta_ext = np.arctan2(param.T_bias[1, -1], param.T_bias[0, -1])
    x_ext = np.array([*x.flatten(), theta_ext])
    fc = fkin_ext_obj(x_ext, l)
    # print(np.shape(fc))
    al = np.linspace(-np.pi, np.pi, 50)

    J = Jkin_ext_obj(x_ext.flatten(), l)[:2, :]
    if type == 'F':
        S = np.linalg.pinv(J @ J.T)
    elif type == 'V':
        S = J @ J.T
    # print(S)
    D, V = np.linalg.eig(scale * S)
    max_idx = np.where(np.abs(D) == max(np.abs(D)))[0][0]
    min_idx = np.where(np.abs(D) == min(np.abs(D)))[0][0]
    D = D[[max_idx, min_idx]]
    V = V[:, [max_idx, min_idx]]
    D = np.diag(D)
    R = np.real(V @ np.sqrt(D+0j))
    if base == False:
        msh = (R @ np.array([np.cos(al), np.sin(al)])).T + np.matlib.repmat(fc[:, :-1].reshape(-1, 1), 1, 50).T
    else:
        msh = (R @ np.array([np.cos(al), np.sin(al)])).T

    if axes == True:
        ax.annotate("", xy=fc[0, :2] + R[:, 0], xycoords='data', xytext=fc[0, :2] , textcoords='data',
                    arrowprops=dict(arrowstyle="->", linestyle="--", connectionstyle="arc3", color='r'))  # 画x轴
        ax.annotate("", xy=fc[0, :2] + R[:, 1], xycoords='data', xytext=fc[0, :2] , textcoords='data',
                    arrowprops=dict(arrowstyle="->", linestyle="--", connectionstyle="arc3", color='b'))  # 画x轴
    if center == True:
        ax.scatter(fc[0, 0], fc[0, 1], color='r', marker='o')

    p = patches.Polygon(msh, closed=True, alpha=alpha, zorder=zorder)
    p.set_facecolor(facecolor)
    p.set_edgecolor(edgecolor)
    ax.add_patch(p)
    return S
def plot_mani_ellipse_ext_obj3(ax, x, param, type='F', scale=1, axes=False, facecolor=None, edgecolor=None, alpha=None, zorder=None, base=False, center=False):
    # Plot robot manipulability ellipsoid input the joint state
    if edgecolor is None:
        edgecolor = [0.3, 0.3, 0.3]
    if facecolor is None:
        facecolor = [0.4, 0.4, 0.4]
    if alpha is None:
        alpha = 0.9
    if zorder is None:
        zorder = 1
    x = x.reshape(1, -1)

    l_ext = np.linalg.norm(param.T_tipINee3[:2, -1].copy())
    l = np.array([*param.linkLengths, l_ext])
    theta_ext = np.arctan2(param.T_tipINee3[1, 0], param.T_tipINee3[0, 0])
    theta_ext_bias = np.arctan2((param.object_tip_local3[1] - param.T_eeINobj3[1, -1]),
                                (param.object_tip_local3[0] - param.T_eeINobj3[0, -1]))
    theta_ext = theta_ext + theta_ext_bias
    # theta_ext = np.arctan2(param.T_bias[1, -1], param.T_bias[0, -1])
    x_ext = np.array([*x.flatten(), theta_ext])
    fc = fkin_ext_obj(x_ext, l)
    # print(np.shape(fc))
    al = np.linspace(-np.pi, np.pi, 50)

    J = Jkin_ext_obj(x_ext.flatten(), l)[:2, :]
    if type == 'F':
        S = np.linalg.pinv(J @ J.T)
    elif type == 'V':
        S = J @ J.T
    # print(S)
    D, V = np.linalg.eig(scale * S)
    max_idx = np.where(np.abs(D) == max(np.abs(D)))[0][0]
    min_idx = np.where(np.abs(D) == min(np.abs(D)))[0][0]
    D = D[[max_idx, min_idx]]
    V = V[:, [max_idx, min_idx]]
    D = np.diag(D)
    R = np.real(V @ np.sqrt(D+0j))
    if base == False:
        msh = (R @ np.array([np.cos(al), np.sin(al)])).T + np.matlib.repmat(fc[:, :-1].reshape(-1, 1), 1, 50).T
    else:
        msh = (R @ np.array([np.cos(al), np.sin(al)])).T

    if axes == True:
        ax.annotate("", xy=fc[0, :2] + R[:, 0], xycoords='data', xytext=fc[0, :2] , textcoords='data',
                    arrowprops=dict(arrowstyle="->", linestyle="--", connectionstyle="arc3", color='r'))  # 画x轴
        ax.annotate("", xy=fc[0, :2] + R[:, 1], xycoords='data', xytext=fc[0, :2] , textcoords='data',
                    arrowprops=dict(arrowstyle="->", linestyle="--", connectionstyle="arc3", color='b'))  # 画x轴
    if center == True:
        ax.scatter(fc[0, 0], fc[0, 1], color='r', marker='o')

    p = patches.Polygon(msh, closed=True, alpha=alpha, zorder=zorder)
    p.set_facecolor(facecolor)
    p.set_edgecolor(edgecolor)
    ax.add_patch(p)
    return S

def plot_mani_ellipse_geo_obj(ax, x, param, type='F', Jc=np.zeros(3)+1, scale=1, axes=False, facecolor=None, edgecolor=None, alpha=None, zorder=None, base=False, center=False, **kwargs):
    # Plot robot manipulability ellipsoid input the joint state
    if edgecolor is None:
        edgecolor = [0.3, 0.3, 0.3]
    if facecolor is None:
        facecolor = [0.4, 0.4, 0.4]
    if alpha is None:
        alpha = 0.9
    if zorder is None:
        zorder = 1
    l_ext = np.linalg.norm(param.T_tipINee0[:2, -1].copy())
    l = np.array([*param.linkLengths, l_ext])
    theta_ext = np.arctan2(param.T_tipINee0[1, 0], param.T_tipINee0[0, 0])
    theta_ext_bias = np.arctan2((param.object_tip_local0[1] - param.T_eeINobj0[1, -1]),
                                (param.object_tip_local0[0] - param.T_eeINobj0[0, -1]))
    theta_ext = theta_ext + theta_ext_bias
    # theta_ext = np.arctan2(param.T_bias[1, -1], param.T_bias[0, -1])
    x_ext = np.array([*x.flatten(), theta_ext])
    fc = fkin_ext_obj(x_ext, l)
    # print(np.shape(fc))
    al = np.linspace(-np.pi, np.pi, 50)

    J = Jkin_geo_obj(param, x_ext, l,  Jc)[:2, :]
    if type == 'F':
        S = np.linalg.pinv(J @ J.T)
    elif type == 'V':
        S = J @ J.T
    # print(S)
    D, V = np.linalg.eig(scale * S)
    max_idx = np.where(np.abs(D) == max(np.abs(D)))[0][0]
    min_idx = np.where(np.abs(D) == min(np.abs(D)))[0][0]
    D = D[[max_idx, min_idx]]
    V = V[:, [max_idx, min_idx]]
    D = np.diag(D)
    R = np.real(V @ np.sqrt(D+0j))
    if base == False:
        msh = (R @ np.array([np.cos(al), np.sin(al)])).T + np.matlib.repmat(fc[:, :-1].reshape(-1, 1), 1, 50).T
    else:
        msh = (R @ np.array([np.cos(al), np.sin(al)])).T

    if axes == True:
        ax.annotate("", xy=fc[0, :2] + R[:, 0], xycoords='data', xytext=fc[0, :2] , textcoords='data',
                    arrowprops=dict(arrowstyle="->", linestyle="--", connectionstyle="arc3", color='r'))  # 画x轴
        ax.annotate("", xy=fc[0, :2] + R[:, 1], xycoords='data', xytext=fc[0, :2] , textcoords='data',
                    arrowprops=dict(arrowstyle="->", linestyle="--", connectionstyle="arc3", color='b'))  # 画x轴
    if center == True:
        ax.scatter(fc[0, 0], fc[0, 1], color='r', marker='o')

    p = patches.Polygon(msh, closed=True, alpha=alpha, zorder=zorder, **kwargs)
    p.set_facecolor(facecolor)
    p.set_edgecolor(edgecolor)
    ax.add_patch(p)
    return S


def plot_mani_ellipse_geo(ax, x, param, type='F', Jc=np.array([0, 0, 0]), scale=1, axes=False, facecolor=None, edgecolor=None, alpha=None, zorder=None, base=False, center=False, **kwargs):
    # Plot robot manipulability ellipsoid input the joint state
    if edgecolor is None:
        edgecolor = [0.3, 0.3, 0.3]
    if facecolor is None:
        facecolor = [0.4, 0.4, 0.4]
    if alpha is None:
        alpha = 0.9
    if zorder is None:
        zorder = 1

    fc = fkin(param, x)
    # print(np.shape(fc))
    al = np.linspace(-np.pi, np.pi, 50)

    J = Jkin_geo(param, x, Jc)[:2, :]

    if type == 'F':
        S = np.linalg.pinv(J @ J.T)
    elif type == 'V':
        S = J @ J.T
    # print(S)
    D, V = np.linalg.eig(scale * S)
    max_idx = np.where(np.abs(D) == max(np.abs(D)))[0][0]
    min_idx = np.where(np.abs(D) == min(np.abs(D)))[0][0]
    D = D[[max_idx, min_idx]]
    V = V[:, [max_idx, min_idx]]
    D = np.diag(D)
    R = np.real(V @ np.sqrt(D+0j))
    if base == False:
        msh = (R @ np.array([np.cos(al), np.sin(al)])).T + np.matlib.repmat(fc[:, :-1].reshape(-1, 1), 1, 50).T
    else:
        msh = (R @ np.array([np.cos(al), np.sin(al)])).T

    if axes == True:
        ax.annotate("", xy=fc[0, :2] + R[:, 0], xycoords='data', xytext=fc[0, :2] , textcoords='data',
                    arrowprops=dict(arrowstyle="->", linestyle="--", connectionstyle="arc3", color='r'))  # 画x轴
        ax.annotate("", xy=fc[0, :2] + R[:, 1], xycoords='data', xytext=fc[0, :2] , textcoords='data',
                    arrowprops=dict(arrowstyle="->", linestyle="--", connectionstyle="arc3", color='b'))  # 画x轴
    if center == True:
        ax.scatter(fc[0, 0], fc[0, 1], color='r', marker='o')

    p = patches.Polygon(msh, closed=True, alpha=alpha, zorder=zorder, **kwargs)
    p.set_facecolor(facecolor)
    p.set_edgecolor(edgecolor)
    ax.add_patch(p)
    return S


def plot_mani_ellipse_geo_W(ax, x, param, type='F', W=np.diag([1, 1 ,1]), scale=1, axes=False, facecolor=None,
                          edgecolor=None, alpha=None, zorder=None, base=False, center=False, **kwargs):
    # Plot robot manipulability ellipsoid input the joint state
    if edgecolor is None:
        edgecolor = [0.3, 0.3, 0.3]
    if facecolor is None:
        facecolor = [0.4, 0.4, 0.4]
    if alpha is None:
        alpha = 0.9
    if zorder is None:
        zorder = 1

    fc = fkin(param, x)
    # print(np.shape(fc))
    al = np.linspace(-np.pi, np.pi, 50)

    J = Jkin(param, x)[:2, :]

    if type == 'F':
        S = np.linalg.pinv(J @ np.linalg.pinv(W @ W.T) @ J.T)
    elif type == 'V':
        S = J @ (W @ W.T) @ J.T
    # print(S)
    D, V = np.linalg.eig(scale * S)
    max_idx = np.where(np.abs(D) == max(np.abs(D)))[0][0]
    min_idx = np.where(np.abs(D) == min(np.abs(D)))[0][0]
    D = D[[max_idx, min_idx]]
    V = V[:, [max_idx, min_idx]]
    D = np.diag(D)
    R = np.real(V @ np.sqrt(D + 0j))
    if base == False:
        msh = (R @ np.array([np.cos(al), np.sin(al)])).T + np.matlib.repmat(fc[:, :-1].reshape(-1, 1), 1, 50).T
    else:
        msh = (R @ np.array([np.cos(al), np.sin(al)])).T

    if axes == True:
        ax.annotate("", xy=fc[0, :2] + R[:, 0], xycoords='data', xytext=fc[0, :2], textcoords='data',
                    arrowprops=dict(arrowstyle="->", linestyle="--", connectionstyle="arc3", color='r'))  # 画x轴
        ax.annotate("", xy=fc[0, :2] + R[:, 1], xycoords='data', xytext=fc[0, :2], textcoords='data',
                    arrowprops=dict(arrowstyle="->", linestyle="--", connectionstyle="arc3", color='b'))  # 画x轴
    if center == True:
        ax.scatter(fc[0, 0], fc[0, 1], color='r', marker='o')

    p = patches.Polygon(msh, closed=True, alpha=alpha, zorder=zorder, **kwargs)
    p.set_facecolor(facecolor)
    p.set_edgecolor(edgecolor)
    ax.add_patch(p)
    return S


def plot_mani_ellipse_ext_obj_num(ax, x, param, type='F', scale=1, axes=False, facecolor=None, edgecolor=None, alpha=None, zorder=None, base=False):
    # Plot robot manipulability ellipsoid input the joint state
    if edgecolor is None:
        edgecolor = [0.3, 0.3, 0.3]
    if facecolor is None:
        facecolor = [0.4, 0.4, 0.4]
    if alpha is None:
        alpha = 0.9
    if zorder is None:
        zorder = 1
    x = x.reshape(1, -1)

    fc = fkin_ext(param, x)
    # print(np.shape(fc))
    al = np.linspace(-np.pi, np.pi, 50)

    J = Jkin_ext_obj_num(param, x)[:2, :]
    if type == 'F':
        S = np.linalg.pinv(J @ J.T)
    elif type == 'V':
        S = J @ J.T
    # print(S)
    D, V = np.linalg.eig(scale * S)
    max_idx = np.where(np.abs(D) == max(np.abs(D)))[0][0]
    min_idx = np.where(np.abs(D) == min(np.abs(D)))[0][0]
    D = D[[max_idx, min_idx]]
    V = V[:, [max_idx, min_idx]]
    D = np.diag(D)
    R = np.real(V @ np.sqrt(D+0j))
    if base == False:
        msh = (R @ np.array([np.cos(al), np.sin(al)])).T + np.matlib.repmat(fc[:, :-1].reshape(-1, 1), 1, 50).T
    else:
        msh = (R @ np.array([np.cos(al), np.sin(al)])).T

    if axes == True:
        ax.annotate("", xy=fc[0, :2] + R[:, 0], xycoords='data', xytext=fc[0, :2] , textcoords='data',
                    arrowprops=dict(arrowstyle="->", linestyle="--", connectionstyle="arc3", color='r'))  # 画x轴
        ax.annotate("", xy=fc[0, :2] + R[:, 1], xycoords='data', xytext=fc[0, :2] , textcoords='data',
                    arrowprops=dict(arrowstyle="->", linestyle="--", connectionstyle="arc3", color='b'))  # 画x轴

    p = patches.Polygon(msh, closed=True, alpha=alpha, zorder=zorder)
    p.set_facecolor(facecolor)
    p.set_edgecolor(edgecolor)
    ax.add_patch(p)
    return S

def plot_mani_ellipse_time(ax, x, t, param, type='F', facecolor=None, edgecolor=None, alpha=None, zorder=None):
    # Plot robot manipulability ellipsoid input the joint state
    if edgecolor is None:
        edgecolor = [0.3, 0.3, 0.3]
    if facecolor is None:
        facecolor = [0.4, 0.4, 0.4]
    if alpha is None:
        alpha = 0.9
    if zorder is None:
        zorder = 1
    # print(np.shape(fc))
    al = np.linspace(-np.pi, np.pi, 50)

    J = Jkin(param, x)[:2, :]
    if type == 'F':
        S = np.linalg.pinv(J @ J.T)
    elif type == 'V':
        S = J @ J.T
    # print(S)
    D, V = np.linalg.eig(S)
    D = np.diag(D)
    R = np.real(V @ np.sqrt(D+0j))

    msh = (R @ np.array([np.cos(al), np.sin(al)])).T + np.matlib.repmat(np.array([t, 0]).reshape(-1, 1), 1, 50).T

    p = patches.Polygon(msh, closed=True, alpha=alpha, zorder=zorder)
    p.set_facecolor(facecolor)
    p.set_edgecolor(edgecolor)

    ax.scatter(t, 0, color=facecolor)
    ax.add_patch(p)

def plot_ellipse(ax, p, S, scale=1, linestyle=None, linewidth=None, facecolor=None, edgecolor=None, alpha=None, zorder=None):
    # Plot robot manipulability ellipsoid input the joint state
    if linestyle is None:
        linestyle = '-'
    if linewidth is None:
        linewidth = 1.5
    if edgecolor is None:
        edgecolor = [0.3, 0.3, 0.3]
    if facecolor is None:
        facecolor = [0.4, 0.4, 0.4]
    if alpha is None:
        alpha = 0.9
    if zorder is None:
        zorder = 1

    # print(np.shape(fc))
    al = np.linspace(-np.pi, np.pi, 50)

    D, V = np.linalg.eig(scale*S)
    D = np.diag(D)
    R = np.real(V @ np.sqrt(D+0j))
    msh = (R @ np.array([np.cos(al), np.sin(al)])).T + np.matlib.repmat(p.reshape(-1, 1), 1, 50).T
    p_ = patches.Polygon(msh, closed=True, alpha=alpha, zorder=zorder)
    p_.set_facecolor(facecolor)
    p_.set_edgecolor(edgecolor)
    p_.set_linestyle(linestyle)
    p_.set_linewidth(linewidth)
    ax.add_patch(p_)