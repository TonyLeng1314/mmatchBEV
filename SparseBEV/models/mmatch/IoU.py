import torch

def boxes3d_iou(preds: torch.Tensor, gts: torch.Tensor):
    """
    preds: (N, X, 10)
    gts:   (X, 10)
    return: (N, X) IoU
    """
    assert preds.shape[-1] == 10 and gts.shape[-1] == 10
    
    device = preds.device
    N, X, _ = preds.shape
    
    # ============ 参数解析 ============
    cx1, cy1, w1, l1, cz1, h1, sin1, cos1 = [preds[:,:,i] for i in range(8)]
    cx2, cy2, w2, l2, cz2, h2, sin2, cos2 = [gts[:,i] for i in range(8)]

    # broadcast gts -> (N,X)
    cx2 = cx2.unsqueeze(0).expand(N,-1)
    cy2 = cy2.unsqueeze(0).expand(N,-1)
    w2  = w2.unsqueeze(0).expand(N,-1)
    l2  = l2.unsqueeze(0).expand(N,-1)
    cz2 = cz2.unsqueeze(0).expand(N,-1)
    h2  = h2.unsqueeze(0).expand(N,-1)
    sin2 = sin2.unsqueeze(0).expand(N,-1)
    cos2 = cos2.unsqueeze(0).expand(N,-1)

    # yaw
    yaw1 = torch.atan2(sin1, cos1)
    yaw2 = torch.atan2(sin2, cos2)

    # ============ BEV IoU ============
    # 简化：先近似用 AABB IoU（旋转忽略），确保数值正常
    x1min = cx1 - w1/2; x1max = cx1 + w1/2
    y1min = cy1 - l1/2; y1max = cy1 + l1/2
    
    x2min = cx2 - w2/2; x2max = cx2 + w2/2
    y2min = cy2 - l2/2; y2max = cy2 + l2/2
    
    inter_xmin = torch.max(x1min, x2min)
    inter_ymin = torch.max(y1min, y2min)
    inter_xmax = torch.min(x1max, x2max)
    inter_ymax = torch.min(y1max, y2max)

    inter_w = (inter_xmax - inter_xmin).clamp(min=0)
    inter_l = (inter_ymax - inter_ymin).clamp(min=0)
    inter_area = inter_w * inter_l
    
    area1 = w1 * l1
    area2 = w2 * l2
    union_area = area1 + area2 - inter_area

    bev_iou = inter_area / (union_area + 1e-7)

    # ============ 高度 IoU ============
    z1min = cz1 - h1/2; z1max = cz1 + h1/2
    z2min = cz2 - h2/2; z2max = cz2 + h2/2

    inter_h = (torch.min(z1max, z2max) - torch.max(z1min, z2min)).clamp(min=0)
    inter_vol = inter_area * inter_h

    vol1 = area1 * h1
    vol2 = area2 * h2
    union_vol = vol1 + vol2 - inter_vol

    iou3d = inter_vol / (union_vol + 1e-7)

    return iou3d  # shape (N,X)
