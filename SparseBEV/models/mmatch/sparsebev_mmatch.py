import torch
from torch import Tensor
from ..bbox.utils import normalize_bbox
from .IoU import boxes3d_iou
import torch.nn.functional as F
from mmdet.models.losses import FocalLoss


def assign_topk_per_gt(iou, k=3, min_iou=0.0):
    """
    iou: (N,X)
    returns pos_mask: (N,X) bool
    """
    N, X = iou.shape
    # topk 返回 indices shape (k, X)
    k = min(k, N)
    topk_vals, topk_idx = torch.topk(iou, k=k, dim=0)  # (k,X)
    pos_mask = torch.zeros_like(iou, dtype=torch.bool)
    # set positions true where topk_vals >= min_iou
    for r in range(k):
        # only accept if value >= threshold
        mask = topk_vals[r] >= min_iou
        pos_mask[topk_idx[r, mask], torch.arange(X, device=iou.device)[mask]] = True
    return pos_mask

def loss_cal(
    all_cls_scores:Tensor, # [L, B, N, 10]
    all_bbox_preds:Tensor, # [L, B, N, 10]
    all_gt_bboxes_list, # [[X, 9] * B] * L
    all_gt_labels_list,  # [[X] * B] * L
    layer_list = [i for i in range(6)],
    topk = 3,
      
):
    # settings:
    with_cls_loss = False
    
    
    
    batches = all_cls_scores.shape[1]
    device = all_cls_scores.device
    focal = FocalLoss(use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0).to(device)
    
    loss_list = []
    
    for layer in layer_list:
        for batch in range(batches):
            
            bbox_preds = all_bbox_preds[layer,batch] # [N, 10]
            gt_bbox = all_gt_bboxes_list[layer][batch] # [X, 9]
            gt_bbox_normed = normalize_bbox(gt_bbox) # [X, 10] out = torch.cat([cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy], dim=-1)
            
            cls_scores = all_cls_scores[layer,batch] # [N, 10]
            gt_labels = all_gt_labels_list[layer][batch] # [X]
                
            with torch.no_grad():
                pred_coord = bbox_preds[:,[0,1,4]] # [N, 3]
                pred_distance = pred_coord.norm(dim=-1) # [N]
                gt_distance = gt_bbox_normed[:,[0,1,4]].norm(dim = -1) # [X]
                
                
                projected_coord = (pred_coord / (pred_distance[:,None] + 1e-8))[:,None,:] * gt_distance[None,:,None] # [N, X, 3]
                bbox_projected = bbox_preds[:,None,:].repeat(1,projected_coord.shape[1],1)
                bbox_projected[:,:,[0,1,4]] = projected_coord # [N, X, 10]
                IoU_3D = boxes3d_iou(bbox_projected,gt_bbox_normed) # [N, X]
                
                pos_mask = assign_topk_per_gt(IoU_3D,k=topk,min_iou=0.2)

                # torch.save(gt_bbox_normed,'gt_bbox.pt')
                # torch.save(bbox_projected,'bbox_projected.pt')
                # torch.save(bbox_preds,'bbox_preds.pt')
            N,_ = bbox_preds.shape
            X,_ = gt_bbox.shape
            pred = bbox_preds[:,None,:].repeat(1,X,1)
            gt = gt_bbox_normed[None,:,:].repeat(N,1,1)
            
            # mmatch_result = {'pred':pred[pos_mask],'gt':gt[pos_mask]}
            # torch.save(mmatch_result,'mmatch_result_dict.pt')
            # import pdb;pdb.set_trace()
            loss_bbox_single = F.l1_loss(pred[pos_mask], gt[pos_mask], reduction='mean')
            
            # for cls loss:
            cls_preds = cls_scores[:,None,:].repeat(1,X,1) # [N, X, 10]
            cls_gt = gt_labels[None,:].repeat(N,1) # [N, X]
            
            cls_preds_flat = cls_preds[pos_mask].view(-1,10)
            cls_gt_flat = cls_gt[pos_mask].view(-1)
            if cls_gt_flat.shape[0] != 0 and with_cls_loss:
                loss_cls_single = focal(cls_preds_flat, cls_gt_flat, reduction_override='mean')
                loss_list.extend([loss_bbox_single,loss_cls_single])
            else:
                loss_list.append(loss_bbox_single)
    
    
    all_losses = torch.stack(loss_list)   # shape = [num_layers * batches]
    mean_loss = all_losses.mean()
    
    return torch.nan_to_num(mean_loss)