

import numpy as np
import torch
from math import atan2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from SparseBEV.models.bbox import utils as boxutils

def plot_boxes_3d(tensors_900x10, ax=None, max_boxes=None, linewidth=1.0, alpha=1.0, show_center=False,color = 'red'):
    """
    在 matplotlib 3D 中绘制 3D 框。
    输入:
      tensors_900x10 : torch.Tensor or np.ndarray, shape (N, 10)  (N <= 900)
        列顺序假设为 [x, y, z, w, l, h, sin, cos, vx, vy]
      ax : matplotlib 3D axis (optional). 如果 None 会自动创建一个。
      max_boxes : int or None, 若指定则只绘制前 max_boxes 个框（方便调试）。
      linewidth : float, 边线宽度。
      alpha : float, 透明度。
      show_center : bool, 是否在每个框中心画一个小点。
    返回:
      ax (matplotlib 3D axis)
    """
    # 转 numpy
    if isinstance(tensors_900x10, torch.Tensor):
        data = tensors_900x10.detach().cpu().numpy()
    else:
        data = np.asarray(tensors_900x10)

    if data.ndim != 2 or data.shape[1] < 8:
        raise ValueError("输入应为 (N, 10) 或 (N, >=8)。")

    N = data.shape[0]
    if max_boxes is not None:
        N = min(N, max_boxes)
        data = data[:N]

    # 创建 ax
    created_ax = False
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        created_ax = True

    # 12 条边的顶点索引对（使用 8 个角点的索引 0..7）
    edges = [
        (0,1),(1,2),(2,3),(3,0),  # top rectangle
        (4,5),(5,6),(6,7),(7,4),  # bottom rectangle
        (0,4),(1,5),(2,6),(3,7)   # vertical edges
    ]

    # 遍历每个 box
    for i, row in enumerate(data):
        x_c, y_c, z_c = float(row[0]), float(row[1]), float(row[4])
        w, l, h = float(row[2]), float(row[3]), float(row[5])
        sin_v, cos_v = float(row[6]), float(row[7])

        # 计算 yaw（确保用 atan2(sin, cos)）
        yaw = atan2(sin_v, cos_v)

        # 局部角点 (以 box 中心为原点)
        # 约定：length 沿 x 轴正方向，width 沿 y 轴正方向，height 沿 z 轴向上
        lx = l / 2.0
        wy = w / 2.0
        hz = h / 2.0

        # 角点顺序：前4个为 top (z + h/2)，后4个为 bottom (z - h/2)
        x_local = np.array([ lx,  lx, -lx, -lx,  lx,  lx, -lx, -lx])
        y_local = np.array([ wy, -wy, -wy,  wy,  wy, -wy, -wy,  wy])
        z_local = np.array([ hz,  hz,  hz,  hz, -hz, -hz, -hz, -hz])

        # 旋转（绕 z 轴）
        c = np.cos(yaw)
        s = np.sin(yaw)
        x_rot = c * x_local - s * y_local
        y_rot = s * x_local + c * y_local
        z_rot = z_local

        # 平移到全局
        xs = x_c + x_rot
        ys = y_c + y_rot
        zs = z_c + z_rot

        # 画 12 条边
        for (a, b) in edges:
            ax.plot([xs[a], xs[b]], [ys[a], ys[b]], [zs[a], zs[b]],
                    linewidth=linewidth, alpha=alpha,color=color)

        # 可选：画中心点
        if show_center:
            ax.scatter([x_c], [y_c], [z_c], marker='.', s=0.1, depthshade=False)

    # 设置等比（让长宽高比例真实显示）
    # 这是 matplotlib 的常用技巧：按所有点范围设置相同的 scale
    try:
        all_x = data[:,0]
        all_y = data[:,1]
        all_z = data[:,2]
        max_range = max(all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min())
        if max_range == 0:
            max_range = 1.0
        mid_x = (all_x.max() + all_x.min()) / 2.0
        mid_y = (all_y.max() + all_y.min()) / 2.0
        mid_z = (all_z.max() + all_z.min()) / 2.0
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    except Exception:
        pass

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=-60)  # 可调整观察角度

    if created_ax:
        plt.tight_layout()

    return ax


loss_raw_data = torch.load('loss_raw_data_baseline.pt')

all_cls_scores = loss_raw_data['all_cls_scores']
all_bbox_preds = loss_raw_data['all_bbox_preds']
all_gt_bboxes_list = loss_raw_data['all_gt_bboxes_list']
all_gt_labels_list = loss_raw_data['all_gt_labels_list']



ax = plot_boxes_3d(all_bbox_preds[5,0],linewidth=0.1,color='blue')
gt_bboxes = all_gt_bboxes_list[0][0]
gt_bboxes = boxutils.normalize_bbox(gt_bboxes)
ax = plot_boxes_3d(gt_bboxes,ax=ax,linewidth=0.1)


plt.savefig('output.png',dpi=500)


import pdb;pdb.set_trace()
print('wtf')