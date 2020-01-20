import torch
import torch.nn.functional as F
import cvo_dense_samp, cvo_dense_angle, cvo_dense_with_normal, cvo_dense_normal
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
    def forward(ctx, pts, pts_info, grid_source, grid_valid, neighbor_range, ignore_ib=False):
        """ pts: B*2*N, pts_info: B*C*N, grid_source: B*C*H*W (C could be xyz, rgb, ...), grid_valid: B*1*H*W, neighbor_range: int
        """
        outputs = cvo_dense_angle.forward(pts, pts_info, grid_source, grid_valid, neighbor_range, ignore_ib)
        ctx.save_for_backward(pts, pts_info, grid_source, grid_valid)
        ctx.neighbor_range = neighbor_range
        ctx.ignore_ib = ignore_ib
        return outputs

    @staticmethod
    def backward(ctx, dy):
        pts, pts_info, grid_source, grid_valid = ctx.saved_tensors
        dy = dy.contiguous()
        dx1, dx2 = cvo_dense_angle.backward(dy, pts, pts_info, grid_source, grid_valid, ctx.neighbor_range, ctx.ignore_ib)
        return None, dx1, dx2, None, None, None, None


class PtSampleInGridWithNormal(Function):
    @staticmethod
    def forward(ctx, pts, pts_info, grid_source, grid_valid, pts_normal, grid_normal, pts_nres, grid_nres, neighbor_range, ell, mag_max, mag_min, ignore_ib=False, norm_in_dist=False):
        """ pts: B*2*N, pts_info: B*C*N, grid_source: B*C*H*W (C could be xyz, rgb, ...), grid_valid: B*1*H*W, neighbor_range: int
        """
        outputs = cvo_dense_with_normal.forward(pts, pts_info, grid_source, grid_valid, pts_normal, grid_normal, pts_nres, grid_nres, neighbor_range, ell, mag_max, mag_min, ignore_ib, norm_in_dist)
        ctx.save_for_backward(pts, pts_info, grid_source, grid_valid, pts_normal, grid_normal, pts_nres, grid_nres)
        ctx.neighbor_range = neighbor_range
        ctx.ell = ell
        ctx.mag_max = mag_max
        ctx.mag_min = mag_min
        ctx.ignore_ib = ignore_ib
        ctx.norm_in_dist = norm_in_dist
        return outputs

    @staticmethod
    def backward(ctx, dy):
        pts, pts_info, grid_source, grid_valid, pts_normal, grid_normal, pts_nres, grid_nres = ctx.saved_tensors
        dy = dy.contiguous()
        dx1, dx2, dn1, dn2, dr1, dr2 = cvo_dense_with_normal.backward(dy, pts, pts_info, grid_source, grid_valid, pts_normal, grid_normal, pts_nres, grid_nres, ctx.neighbor_range, ctx.ell, ctx.mag_max, ctx.mag_min, ctx.ignore_ib, ctx.norm_in_dist)
        # return None, dx1, dx2, None, dn1, dn2, dr1, dr2, None, None, None, None, None, None
        return None, dx1, dx2, None, None, None, dr1, dr2, None, None, None, None, None, None
        
class PtSampleInGridCalcNormal(Function):
    @staticmethod
    def forward(ctx, pts, grid_source, grid_valid, neighbor_range, ignore_ib):
        normals, norm_sq, ioffs = cvo_dense_normal.forward(pts, grid_source, grid_valid, neighbor_range, ignore_ib)
        ctx.save_for_backward(ioffs, pts, grid_source)
        ctx.ignore_ib = ignore_ib
        return normals, norm_sq
    
    @staticmethod
    def backward(ctx, dnormals, dnorms):
        # ioffs, pts, grid_source = ctx.saved_tensors
        # dgrid = cvo_dense_normal.backward(dnormals, dnorms, ioffs, pts, grid_source, ctx.ignore_ib)
        # return None, dgrid, None, None, None
        return None, None, None, None, None

def calc_normal(pts, grid_source, grid_valid, neighbor_range, ignore_ib, min_dist_2=0.05):
    raw_normals, norm_sq = PtSampleInGridCalcNormal.apply(pts, grid_source, grid_valid, neighbor_range, ignore_ib) ## raw_normals is 4*C*N, and norm_sq is 4*2*N

    # raw_normals = torch.ones((4,3,pts.shape[-1]), device=grid_source.device, dtype=grid_source.dtype)
    # norm_sq = torch.ones((4,2,pts.shape[-1]), device=grid_source.device, dtype=grid_source.dtype)

    normed_normal = F.normalize(raw_normals, p=2, dim=1) # 4*C*N

    norms = torch.sqrt(norm_sq + 1e-8) # |a|, |b|, 4*2*N
    normal_sin_scale = raw_normals / (norms[:,0:1] * norms[:,1:2]) # raw_normal |axb| = |a||b|sin(alpha), 4*C*N

    W_norms = 1 / ( torch.clamp(norms, min=min_dist_2).sum(dim=1, keepdim=True) ) # 4*1*N
    weighted_normal = (normal_sin_scale * W_norms).sum(dim=0, keepdim=True) # 1*C*N
    weighted_normal = F.normalize(weighted_normal, p=2, dim=1) # weighted_normal / torch.norm(weighted_normal, dim=0, keepdim=True) # 1*C*N

    W_norms_effective = torch.norm(normal_sin_scale, dim=1,keepdim=True) * W_norms # 4*1*N
    W_norms_sum = W_norms_effective.sum(dim=0, keepdim=True) # 1*1*N
    
    ## calculate residual
    res_sin_sq = 1- (normed_normal * weighted_normal).sum(dim=1, keepdim=True).pow(2) # 4*1*N
    res_weighted_sum = (res_sin_sq * W_norms_effective).sum(dim=0, keepdim=True) / (W_norms_sum + 1e-8) # 1*1*N

    single_cross = (W_norms_sum != 0) & (res_weighted_sum == 0) # 1*1*N
    single_cross_default_sin_sq = torch.ones_like(res_weighted_sum) * 0.5 # 1*1*N
    res_final = torch.where(single_cross, single_cross_default_sin_sq, res_weighted_sum) # 1*1*N

    ## weighted_normal is unit length normal vectors 1*C*N; res_final in [0,1], 1*1*N
    ## some vectors in weighted_normal could be zero if no neighboring pixels are found
    return weighted_normal, res_final