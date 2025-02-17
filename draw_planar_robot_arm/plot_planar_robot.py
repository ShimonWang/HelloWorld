import numpy as np
import numpy.matlib
import matplotlib.path as mpath
from matplotlib.patches import PathPatch
from matplotlib.lines import Line2D


def plotArmLink(ax, a, d, p, sz, facecol, edgecol, **kwargs):
    r"""
       绘制一个机械臂的单个关节。

       本函数根据关节的长度、角度和位置计算并绘制单个机械臂关节的形状。将计算的路径添加到提供的轴对象中。

       参数
       ----------
       ax : matplotlib.axes.Axes
           绘制机械臂关节的坐标轴对象。

       a : float
           关节的角度，相对于前一个关节，单位为弧度。

       d : float
           关节的长度。

       p : array-like, shape (2,)
           关节的基准位置，2D空间中的 (x, y) 坐标。

       sz : float
           关节的大小（半径）。

       facecol : tuple of float
           关节的面颜色 (RGB)。

       edgecol : tuple of float
           关节的边缘颜色 (RGB)。

       kwargs : 额外的关键字参数
           传递给 `PathPatch` 对象的额外参数。

       返回
       -------
       p2 : ndarray, shape (2,)
           关节末端的位置。

       备注
       -----
       通过计算给定参数，创建机械臂关节的路径并绘制。起始点和终点也会用圆形表示。
       """

    nbSegm = 30  # 用于逼近关节形状的段数

    Path = mpath.Path

    # calculate the link border  # 计算关节边界
    xTmp = np.zeros((2, nbSegm))
    p = p + np.array([0, 0]).reshape(2, -1)
    t1 = np.linspace(0, -np.pi, int(nbSegm/2))
    t2 = np.linspace(np.pi, 0, int(nbSegm/2))
    xTmp[0, :] = np.hstack((sz*np.sin(t1), d+sz*np.sin(t2)))
    xTmp[1, :] = np.hstack((sz*np.cos(t1), sz*np.cos(t2)))
    # xTmp[2, :] = np.zeros((1, nbSegm))

    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])  # 旋转矩阵
    x = R @ xTmp + np.matlib.repmat(p, 1, nbSegm)  # 应用旋转和平移
    p2 = R @ np.array([d, 0]).reshape(2, -1) + p  # 关节末端位置

    # add the link patch  # 创建并添加关节路径到坐标轴
    codes = Path.LINETO * np.ones(np.size(x[0:2, :], 1), dtype=Path.code_type)
    codes[0] = Path.MOVETO
    path = Path(x[0:2, :].T, codes)
    patch = PathPatch(path, facecolor=facecol, edgecolor=edgecol, **kwargs)
    ax.add_patch(patch)

    # add the initial point  # 添加起始点（圆形表示）
    msh = np.vstack((np.sin(np.linspace(0, 2*np.pi, nbSegm)),
                     np.cos(np.linspace(0, 2*np.pi, nbSegm)))) * sz * 0.4

    codes = Path.LINETO * np.ones(np.size(msh[0:2, :], 1), dtype=Path.code_type)
    codes[0] = Path.MOVETO
    path = Path((msh[0:2, :]+p).T, codes)
    patch = PathPatch(path, facecolor=facecol, edgecolor=edgecol, **kwargs)
    ax.add_patch(patch)

    # add the end point  # 添加末端点（圆形表示）
    path = Path((msh[0:2, :]+p2).T, codes)
    patch = PathPatch(path, facecolor=facecol, edgecolor=edgecol, **kwargs)
    ax.add_patch(patch)

    return p2  # 返回关节末端位置


def plotArmBasis(ax, p1, sz, facecol, edgecol, **kwargs):
    r"""
        绘制机械臂的基座。

        本函数绘制机械臂的初始支撑结构，包括基座和连接到关节的线条，主要用于绘制机械臂的基础框架或第一部分。

        参数
        ----------
        ax : matplotlib.axes.Axes
            绘制机械臂基座的坐标轴对象。

        p1 : array-like, shape (2,)
            机械臂基座在2D空间中的位置 (x, y)。

        sz : float
            基座的大小（半径）。

        facecol : tuple of float
            基座的面颜色 (RGB)。

        edgecol : tuple of float
            基座的边缘颜色 (RGB)。

        kwargs : 额外的关键字参数
            传递给 `PathPatch` 对象的额外参数。

        备注
        -----
        基座通过一系列线条和路径来表示，绘制机械臂基座的位置。
        """

    Path = mpath.Path

    nbSegm = 30  # 用于逼近基座形状的段数
    sz = sz*1.2  # 调整大小

    xTmp1 = np.zeros((2, nbSegm))
    t1 = np.linspace(0, np.pi, nbSegm-2)
    xTmp1[0, :] = np.hstack([sz*1.5, sz*1.5*np.cos(t1), -sz*1.5])
    xTmp1[1, :] = np.hstack([-sz*1.2, sz*1.5*np.sin(t1), -sz*1.2])
    x1 = xTmp1 + np.matlib.repmat(p1, 1, nbSegm)

    # add the link patch  # 创建并添加基座路径到坐标轴
    codes = Path.LINETO * np.ones(np.size(x1, 1), dtype=Path.code_type)
    codes[0] = Path.MOVETO
    path = Path(x1.T, codes)
    patch = PathPatch(path, facecolor=facecol, edgecolor=edgecol, **kwargs)
    ax.add_patch(patch)

    nb_line = 4  # 基座连接线的数量
    mult = 1.2
    xTmp2 = np.zeros((2, nb_line))  # 2D only  # 仅在2D中有效
    xTmp2[0, :] = np.linspace(-sz * mult, sz * mult, nb_line)
    xTmp2[1, :] = [-sz * mult] * nb_line

    x2 = xTmp2 + np.tile((p1.flatten() + np.array([0.0, 0.0]))[:, None], (1, nb_line))
    x3 = xTmp2 + np.tile((p1.flatten() + np.array([-0.2, -0.8])*sz)[:, None], (1, nb_line))

    # 绘制基座连接线
    for i in range(nb_line):
        tmp = np.zeros((2, 2))  # N*2
        tmp[0] = [x2[0, i], x2[1, i]]
        tmp[1] = [x3[0, i], x3[1, i]]
        patch = Line2D(tmp[:, 0], tmp[:, 1],  color=[0, 0, 0, 1], lw=2, zorder=1)
        ax.add_line(patch)

def plotArm(ax, a, d, p, sz=.1, facecol=None, edgecol=None, xlim=None, ylim=None, robot_base=False, **kwargs):
    r"""
        绘制由多个关节组成的机械臂。

        本函数根据关节的角度、长度和基座位置构造并可视化一个机械臂，支持绘制基座或初始框架。

        参数
        ----------
        ax : matplotlib.axes.Axes
            绘制机械臂的坐标轴对象。

        a : array-like, shape (n,)
            机械臂每个关节的角度，单位为弧度。

        d : array-like, shape (n,)
            机械臂每个关节的长度。

        p : array-like, shape (2,)
            机械臂基座的位置，2D空间中的 (x, y)。

        sz : float, optional
            关节的大小（半径），默认值为 0.1。

        facecol : tuple of float, optional
            关节的面颜色 (RGB)，默认为灰色。

        edgecol : tuple of float, optional
            关节的边缘颜色 (RGB)，默认为白色。

        xlim : tuple, optional
            设置 x 轴的范围。

        ylim : tuple, optional
            设置 y 轴的范围。

        robot_base : bool, optional
            如果为 True，绘制机械臂的基座，默认值为 False。

        kwargs : 额外的关键字参数
            传递给 `PathPatch` 对象的额外参数。

        备注
        -----
        本函数逐个绘制每个机械臂关节，并根据给定的角度旋转每个关节，并按基座位置进行平移。
        """

    if edgecol is None:
        edgecol = [.99, .99, .99]
    if facecol is None:
        facecol = [.5, .5, .5]

    p = np.reshape(p, (-1, 1))

    if robot_base:
        plotArmBasis(ax, p, sz, facecol, edgecol, **kwargs)

    # 逐个绘制机械臂的关节
    for i in range(len(a)):
        p = plotArmLink(ax=ax, a=np.sum(a[0:i+1]), d=d[i],
                    p=p+np.array([0., 0.]).reshape(2, -1),
                    sz=sz, facecol=facecol, edgecol=edgecol, **kwargs)

    # 设置坐标轴范围
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is None and ylim is None:
        ax.autoscale(True)

def plotArm_Tool(ax, a, d, p, sz=.1, facecol=None, edgecol=None, xlim=None, ylim=None, robot_base=False, **kwargs):
    r"""
       绘制机械臂并突出显示末端工具。

       本函数构造并可视化一个由多个关节组成的机械臂，并通过对最后一个关节应用透明度来突出显示末端工具（执行器）。

       参数
       ----------
       ax : matplotlib.axes.Axes
           绘制机械臂的坐标轴对象。

       a : array-like, shape (n,)
           机械臂每个关节的角度，单位为弧度。

       d : array-like, shape (n,)
           机械臂每个关节的长度。

       p : array-like, shape (2,)
           机械臂基座的位置，2D空间中的 (x, y)。

       sz : float, optional
           关节的大小（半径），默认值为 0.1。

       facecol : tuple of float, optional
           关节的面颜色 (RGB)，默认为灰色。

       edgecol : tuple of float, optional
           关节的边缘颜色 (RGB)，默认为白色。

       xlim : tuple, optional
           设置 x 轴的范围。

       ylim : tuple, optional
           设置 y 轴的范围。

       robot_base : bool, optional
           如果为 True，绘制机械臂的基座，默认值为 False。

       kwargs : 额外的关键字参数
           传递给 `PathPatch` 对象的额外参数。

       备注
       -----
       本函数通过应用透明度效果突出显示机械臂的最后一个关节（通常为末端工具）。
       """

    if edgecol is None:
        edgecol = [.99, .99, .99]
    if facecol is None:
        facecol = [.5, .5, .5]
    p = np.reshape(p, (-1, 1))
    if robot_base:
        plotArmBasis(ax, p, sz, facecol, edgecol, **kwargs)

    # 绘制每个关节并对最后一个关节应用透明度
    for i in range(len(a)):
        if i == len(a)-1:
            p = plotArmLink(ax=ax, a=np.sum(a[0:i+1]), d=d[i],
                        p=p+np.array([0., 0.]).reshape(2, -1),
                        sz=sz, facecol=facecol, edgecol=edgecol, alpha=0.4)
        else:
            p = plotArmLink(ax=ax, a=np.sum(a[0:i + 1]), d=d[i],
                            p=p + np.array([0., 0.]).reshape(2, -1),
                            sz=sz, facecol=facecol, edgecol=edgecol, **kwargs)

    # 设置坐标轴范围
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is None and ylim is None:
        ax.autoscale(True)

def plot_planar_axis(ax, p):
    r"""
        绘制2D平面坐标轴。

        本函数绘制一组2D笛卡尔坐标轴，并在指定的位置显示。坐标轴通过两条正交的线表示，一条为x轴，另一条为y轴。

        参数
        ----------
        ax : matplotlib.axes.Axes
            绘制坐标轴的坐标轴对象。

        p : array-like, shape (n, 3)
            数组的每一行表示一个点，格式为 [x, y, theta]，其中 (x, y) 为坐标位置，theta 为坐标轴的方向。

        备注
        -----
        本函数为每个坐标轴绘制两条线，一条表示主轴，另一条表示垂直方向。
        """

    length = 0.2  # 每条坐标轴线的长度
    num = np.size(p, 0)  # 绘制的坐标轴数量

    for i in range(num):
        # 绘制x轴线（红色）
        x_1 = np.array([p[i, 0], p[i, 0] + length * np.cos(p[i, 2])])
        y_1 = np.array([p[i, 1], p[i, 1] + length * np.sin(p[i, 2])])
        ln1, = ax.plot(x_1, y_1, lw=2, solid_capstyle='round', color='r', zorder=1)
        ln1.set_solid_capstyle('round')

        # 绘制y轴线（蓝色）
        x_2 = np.array([p[i, 0], p[i, 0] + length * np.cos(p[i, 2] + np.pi / 2)])
        y_2 = np.array([p[i, 1], p[i, 1] + length * np.sin(p[i, 2] + np.pi / 2)])
        ln2, = ax.plot(x_2, y_2, lw=2, solid_capstyle='round', color='b', zorder=1)
        ln2.set_solid_capstyle('round')

