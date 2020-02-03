import torch
import torch.nn.functional as F
import cvo_dense_samp, cvo_dense_angle, cvo_dense_with_normal, cvo_dense_normal
from torch.autograd import Function

class PtSampleInGrid(Function):
    @staticmethod
    def forward(ctx, pts, pts_info, grid_source, grid_valid, neighbor_range, ell, ignore_ib=False, sqr=True, ell_basedist=0):
        """ pts: B*2*N, pts_info: B*C*N, grid_source: B*C*H*W (C could be xyz, rgb, ...), grid_valid: B*1*H*W, neighbor_range: int
        """
        outputs = cvo_dense_samp.forward(pts, pts_info, grid_source, grid_valid, neighbor_range, ell, ignore_ib, sqr, ell_basedist)
        # ctx.save_for_backward(outputs, pts, pts_info, grid_source, grid_valid)
        ctx.save_for_backward(pts, pts_info, grid_source, grid_valid)
        ctx.neighbor_range = neighbor_range
        ctx.ell = ell
        ctx.ignore_ib = ignore_ib
        ctx.sqr = sqr
        ctx.ell_basedist = ell_basedist
        return outputs

    @staticmethod
    def backward(ctx, dy):
        # outputs, pts, pts_info, grid_source, grid_valid = ctx.saved_tensors
        pts, pts_info, grid_source, grid_valid = ctx.saved_tensors
        dy = dy.contiguous()
        # dx1, dx2 = cvo_dense_samp.backward(dy, outputs, pts, pts_info, grid_source, grid_valid, ctx.neighbor_range, ctx.ell, ctx.ignore_ib)
        dx1, dx2 = cvo_dense_samp.backward(dy, pts, pts_info, grid_source, grid_valid, ctx.neighbor_range, ctx.ell, ctx.ignore_ib, ctx.sqr, ctx.ell_basedist)
        return None, dx1, dx2, None, None, None, None, None, None


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
    def forward(ctx, pts, pts_info, grid_source, grid_valid, pts_normal, grid_normal, pts_nres, grid_nres, neighbor_range, ell, mag_max, mag_min, ignore_ib=False, norm_in_dist=False, ell_basedist=0):
        """ pts: B*2*N, pts_info: B*C*N, grid_source: B*C*H*W (C could be xyz, rgb, ...), grid_valid: B*1*H*W, neighbor_range: int
        """
        outputs = cvo_dense_with_normal.forward(pts, pts_info, grid_source, grid_valid, pts_normal, grid_normal, pts_nres, grid_nres, neighbor_range, ell, mag_max, mag_min, ignore_ib, norm_in_dist, ell_basedist)
        ctx.save_for_backward(pts, pts_info, grid_source, grid_valid, pts_normal, grid_normal, pts_nres, grid_nres)
        ctx.neighbor_range = neighbor_range
        ctx.ell = ell
        ctx.mag_max = mag_max
        ctx.mag_min = mag_min
        ctx.ignore_ib = ignore_ib
        ctx.norm_in_dist = norm_in_dist
        ctx.ell_basedist =ell_basedist
        return outputs

    @staticmethod
    def backward(ctx, dy):
        pts, pts_info, grid_source, grid_valid, pts_normal, grid_normal, pts_nres, grid_nres = ctx.saved_tensors
        dy = dy.contiguous()
        dx1, dx2, dn1, dn2, dr1, dr2 = cvo_dense_with_normal.backward( \
            dy, pts, pts_info, grid_source, grid_valid, pts_normal, grid_normal, pts_nres, grid_nres, ctx.neighbor_range, ctx.ell, ctx.mag_max, ctx.mag_min, ctx.ignore_ib, ctx.norm_in_dist, ctx.ell_basedist)
        return None, dx1, dx2, None, dn1, dn2, dr1, dr2, None, None, None, None, None, None, None
        # return None, dx1, dx2, None, None, None, dr1, dr2, None, None, None, None, None, None
        
class PtSampleInGridCalcNormal(Function):
    @staticmethod
    def forward(ctx, pts, grid_source, grid_valid, neighbor_range, ignore_ib):
        normals, norm_sq, ioffs = cvo_dense_normal.forward(pts, grid_source, grid_valid, neighbor_range, ignore_ib)
        ctx.save_for_backward(ioffs, pts, grid_source)
        ctx.ignore_ib = ignore_ib
        return normals, norm_sq
    
    @staticmethod
    def backward(ctx, dnormals, dnorms):
        ioffs, pts, grid_source = ctx.saved_tensors
        dgrid = cvo_dense_normal.backward(dnormals, dnorms, ioffs, pts, grid_source, ctx.ignore_ib)
        return None, dgrid, None, None, None
        # return None, None, None, None, None

def recall_grad(pre_info, grad):
    # print(pre_info, grad)
    # print(pre_info, torch.isnan(grad).any())
    assert not torch.isnan(grad).any(), pre_info

def calc_normal(pts, grid_source, grid_valid, neighbor_range, ignore_ib, min_dist_2=0.05):
    raw_normals, norm_sq = PtSampleInGridCalcNormal.apply(pts.contiguous(), grid_source.contiguous(), grid_valid.contiguous(), neighbor_range, ignore_ib) ## raw_normals is 4*C*N, and norm_sq is 4*2*N

    if raw_normals.requires_grad:
        raw_normals.register_hook(lambda grad: recall_grad("raw_normals", grad) )

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

    ## the F.normalize will generally result in norm slightly larger than 1
    res_sin_sq = res_sin_sq.clamp(min=0)

    # with torch.no_grad():
    #     diff_normal = (normed_normal - weighted_normal).norm(dim=1) # 4*N
    #     weit_normal = weighted_normal.norm(dim=1) # 1*N
    #     print("identical #:", float(((diff_normal==0) & (weit_normal!=0)).sum() ) )

    #     single = ((raw_normals.norm(dim=1) > 0).sum(dim=0))==1 # N
    #     select_normal = diff_normal[:,single] # 4*N_sub
    #     selsel_normal = select_normal.min(dim=0)[0] # N_sub
    #     print(float(selsel_normal.min()), float(selsel_normal.max()))
        
    #     print("single #:", float(single.sum()))
    #     print("..............................")

    #     normed_norm = normed_normal.norm(dim=1)
    #     normed_normw = weighted_normal.norm(dim=1)

    #     print("normed_norm", float(normed_norm.min()), float(normed_norm.max()))
    #     print("normed_normw", float(normed_normw.min()), float(normed_normw.max()))
    #     print("res_sin_sq", float(res_sin_sq.min()), float(res_sin_sq.max()))

    res_weighted_sum = (res_sin_sq * W_norms_effective).sum(dim=0, keepdim=True) / (W_norms_sum + 1e-8) # 1*1*N

    single = ((raw_normals.norm(dim=1) > 0).sum(dim=0))==1 # N # the points whose normal is calculated from cross product of only 1 pair of points
    single_cross = single.view(1,1,-1)
    # single_cross = (W_norms_sum != 0) & (res_weighted_sum == 0) # 1*1*N
    single_cross_default_sin_sq = torch.ones_like(res_weighted_sum) * 0.5 # 1*1*N
    res_final = torch.where(single_cross, single_cross_default_sin_sq, res_weighted_sum) # 1*1*N

    ## weighted_normal is unit length normal vectors 1*C*N; res_final in [0,1], 1*1*N
    ## some vectors in weighted_normal could be zero if no neighboring pixels are found
    return weighted_normal, res_final