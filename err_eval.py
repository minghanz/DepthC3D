### This is a re-implementation of export_gt_depth.py + evaluate_depth.py, 
### Compare with err_train in compare_eval.py

import numpy as np
import cv2
from layers import disp_to_depth
from compare_eval import depth_metric_names

def error_disp(disp, gt_depth, opt):
    """The version used in evaluate_depth.py
    scale-resize-inverse
    """
    gt_height, gt_width = gt_depth.shape[:2]

    scaled_disp, _ = disp_to_depth(disp, opt.min_depth, opt.max_depth)
    scaled_disp = scaled_disp.cpu()[0, 0].numpy()

    # print("scaled_disp", scaled_disp.shape)
    # print("gt shape", gt_height, gt_width)
    
    scaled_disp = cv2.resize(scaled_disp, (gt_width, gt_height))
    pred_depth = 1 / scaled_disp

    losses = compute_depth_losses(gt_depth, pred_depth, depth_metric_names, opt)

    return losses

def compute_depth_losses(gt_depth, pred_depth, depth_metric_names, opt):
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    gt_height, gt_width = gt_depth.shape[:2]

    losses = {}

    ### creating the mask
    if opt.eval_split == "eigen":
        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                            0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

    else:
        mask = gt_depth > 0

    ### applying the mask
    pred_depth = pred_depth[mask]
    gt_depth = gt_depth[mask]

    ### median normalziation
    pred_depth *= opt.pred_depth_scale_factor
    if not opt.disable_median_scaling:
        ratio = np.median(gt_depth) / np.median(pred_depth)
        pred_depth *= ratio

    ### clamp
    pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
    pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

    ### calculating error
    depth_errors = compute_errors(gt_depth, pred_depth)

    for i, metric in enumerate(depth_metric_names):
        losses[metric] = np.array(depth_errors[i])

    return losses

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / (gt**2))

    d = np.log(pred) - np.log(gt)
    d2 = d ** 2
    si_log = np.sqrt(d2.mean() - d.mean()**2)

    irmse = np.sqrt( ((1/gt - 1/pred)**2).mean() )

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, si_log, irmse