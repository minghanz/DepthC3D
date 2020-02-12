### This is a re-implementation of compute_depth_losses in trainer.py, 
### which is used to report performance during training. 
### Compare with err_eval in compare_eval.py

import numpy as np

import torch
import torch.nn.functional as F
from layers import disp_to_depth
from compare_eval import depth_metric_names

import err_eval

def error_disp(disp, gt_depth, opt):
    """The version used in training
    resize-scale-inverse-resize
    """
    disp = F.interpolate(
                disp, [opt.height, opt.width], mode="bilinear", align_corners=False)

    # gt_height, gt_width = gt_depth.shape[2:]
    # disp = F.interpolate(
    #             disp, [gt_height, gt_width], mode="bilinear", align_corners=False)

    scaled_disp, pred_depth = disp_to_depth(disp, opt.min_depth, opt.max_depth)

    ###################### switch to eval mode ##########################
    # pred_depth_np = pred_depth.cpu().numpy()[0,0]
    # gt_depth_np = gt_depth.cpu().numpy()[0,0]
    # losses = err_eval.compute_depth_losses(gt_depth_np, pred_depth_np, depth_metric_names, opt)
    #####################################################################

    losses = compute_depth_losses(gt_depth, pred_depth, depth_metric_names, opt)

    return losses

def compute_depth_losses(depth_gt, depth_pred, depth_metric_names, opt):
    """Compute depth metrics, to allow monitoring during training

    This isn't particularly accurate as it averages over the entire batch,
    so is only used to give an indication of validation performance

    The same as in trainer.py
    depth_pred = outputs[("depth", 0, 0)]
    depth_gt = inputs["depth_gt"]
    """
    losses = {}

    depth_pred = torch.clamp(F.interpolate(
        depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)

    depth_pred = depth_pred.detach()

    ### creating the mask
    mask = depth_gt > 0

    if opt.eval_split == "eigen":
        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1

        # gt_height, gt_width = depth_gt.shape[2:]
        # crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
        #                     0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
        # crop_mask[:, :, crop[0]:crop[1], crop[2]:crop[3]] = 1

        mask = mask * crop_mask

    ### applying the mask
    depth_gt = depth_gt[mask]
    depth_pred = depth_pred[mask]

    ###################### switch to eval mode ##########################
    # gt_depth = depth_gt.cpu().numpy()
    # pred_depth = depth_pred.cpu().numpy()

    # ### median normalziation
    # pred_depth *= opt.pred_depth_scale_factor
    # if not opt.disable_median_scaling:
    #     ratio = np.median(gt_depth) / np.median(pred_depth)
    #     pred_depth *= ratio
    # ### clamp
    # MIN_DEPTH = 1e-3
    # MAX_DEPTH = 80
    # pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
    # pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

    ### calculating error
    # depth_pred_np = depth_pred.cpu().numpy()
    # depth_gt_np = depth_gt.cpu().numpy()
    # depth_errors = err_eval.compute_errors(gt_depth, pred_depth)
    # for i, metric in enumerate(depth_metric_names):
    #     losses[metric] = np.array(depth_errors[i])
    ######################################################################

    ### median normalization
    if not opt.disable_median_scaling:
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)
    ### clamp
    depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)


    depth_errors = compute_depth_errors(depth_gt, depth_pred)
    for i, metric in enumerate(depth_metric_names):
        losses[metric] = np.array(depth_errors[i].cpu())
    
    return losses

def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    The same as in layers.py
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / (gt**2))

    d = torch.log(pred) - torch.log(gt)
    d2 = d ** 2
    si_log = torch.sqrt(d2.mean() - d.mean()**2)

    irmse = torch.sqrt( ((1/gt - 1/pred)**2).mean() )

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, si_log, irmse