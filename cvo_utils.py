import torch
import cvo_dense_samp
from torch.autograd import Function

class PtSampleInGrid(Function):
    @staticmethod
    def forward(ctx, pts, pts_info, grid_source, grid_valid, neighbor_range, ell, ignore_ib=False):
        """ pts: B*2*N, pts_info: B*C*N, grid_source: B*C*H*W (C could be xyz, rgb, ...), grid_valid: B*1*H*W, neighbor_range: int
        """
        outputs = cvo_dense_samp.forward(pts, pts_info, grid_source, grid_valid, neighbor_range, ell, ignore_ib)
        ctx.save_for_backward(outputs, pts, pts_info, grid_source, grid_valid)
        ctx.neighbor_range = neighbor_range
        ctx.ell = ell
        ctx.ignore_ib = ignore_ib
        return outputs

    @staticmethod
    def backward(ctx, dy):
        outputs, pts, pts_info, grid_source, grid_valid = ctx.saved_tensors
        dy = dy.contiguous()
        dx1, dx2 = cvo_dense_samp.backward(dy, outputs, pts, pts_info, grid_source, grid_valid, ctx.neighbor_range, ctx.ell, ctx.ignore_ib)
        return None, dx1, dx2, None, None, None, None


class PtSampleInGridAngle(Function):
    @staticmethod
    def forward(ctx, pts, pts_info, grid_source, grid_valid, neighbor_range, ell, ignore_ib=False):
        """ pts: B*2*N, pts_info: B*C*N, grid_source: B*C*H*W (C could be xyz, rgb, ...), grid_valid: B*1*H*W, neighbor_range: int
        """
        outputs = cvo_dense_samp.forward(pts, pts_info, grid_source, grid_valid, neighbor_range, ell, ignore_ib)
        ctx.save_for_backward(pts, pts_info, grid_source, grid_valid)
        ctx.neighbor_range = neighbor_range
        ctx.ell = ell
        ctx.ignore_ib = ignore_ib
        return outputs

    @staticmethod
    def backward(ctx, dy):
        pts, pts_info, grid_source, grid_valid = ctx.saved_tensors
        dy = dy.contiguous()
        dx1, dx2 = cvo_dense_samp.backward(dy, pts, pts_info, grid_source, grid_valid, ctx.neighbor_range, ctx.ell, ctx.ignore_ib)
        return None, dx1, dx2, None, None, None, None