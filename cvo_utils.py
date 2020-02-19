import torch
import torch.nn.functional as F
import cvo_dense_samp, cvo_dense_angle, cvo_dense_normal, cvo_dense_with_normal
from torch.autograd import Function
from PIL import Image
import cvo_dense_with_normal_output
import numpy as np

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
    # def forward(ctx, pts, pts_info, grid_source, grid_valid, neighbor_range, ell, ignore_ib=False, sqr=True, ell_basedist=0):
    #     """ pts: B*2*N, pts_info: B*C*N, grid_source: B*C*H*W (C could be xyz, rgb, ...), grid_valid: B*1*H*W, neighbor_range: int
    #     dummy version for memory leak debug
    #     """
    #     outputs = torch.ones( (1,(2*neighbor_range+1)*(2*neighbor_range+1), pts.shape[-1]), device=pts_info.device )
    #     # ctx.save_for_backward(outputs, pts, pts_info, grid_source, grid_valid)
    #     ctx.save_for_backward(pts, pts_info, grid_source, grid_valid)
    #     ctx.neighbor_range = neighbor_range
    #     ctx.ell = ell
    #     ctx.ignore_ib = ignore_ib
    #     ctx.sqr = sqr
    #     ctx.ell_basedist = ell_basedist
    #     return outputs

    @staticmethod
    def backward(ctx, dy):
        # outputs, pts, pts_info, grid_source, grid_valid = ctx.saved_tensors
        pts, pts_info, grid_source, grid_valid = ctx.saved_tensors
        dy = dy.contiguous()
        # dx1, dx2 = cvo_dense_samp.backward(dy, outputs, pts, pts_info, grid_source, grid_valid, ctx.neighbor_range, ctx.ell, ctx.ignore_ib)
        dx1, dx2 = cvo_dense_samp.backward(dy, pts, pts_info, grid_source, grid_valid, ctx.neighbor_range, ctx.ell, ctx.ignore_ib, ctx.sqr, ctx.ell_basedist)
        return None, dx1, dx2, None, None, None, None, None, None
    # def backward(ctx, dy): ## dummy version for memory leak debug
    #     # outputs, pts, pts_info, grid_source, grid_valid = ctx.saved_tensors
    #     pts, pts_info, grid_source, grid_valid = ctx.saved_tensors
    #     dy = dy.contiguous()
    #     # dx1, dx2 = cvo_dense_samp.backward(dy, outputs, pts, pts_info, grid_source, grid_valid, ctx.neighbor_range, ctx.ell, ctx.ignore_ib)
    #     # dx1, dx2 = cvo_dense_samp.backward(dy, pts, pts_info, grid_source, grid_valid, ctx.neighbor_range, ctx.ell, ctx.ignore_ib, ctx.sqr, ctx.ell_basedist)
    #     dx1 = torch.zeros_like(pts_info)
    #     dx2 = torch.zeros_like(grid_source)
    #     return None, dx1, dx2, None, None, None, None, None, None


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
    def forward(ctx, pts, pts_info, grid_source, grid_valid, pts_normal, grid_normal, pts_nres, grid_nres, neighbor_range, ell, mag_max, mag_min, ignore_ib=False, norm_in_dist=False, ell_basedist=0, 
                return_nkern=False, filename=""):
        """ pts: B*2*N, pts_info: B*C*N, grid_source: B*C*H*W (C could be xyz, rgb, ...), grid_valid: B*1*H*W, neighbor_range: int
        """
        if not return_nkern:
            y = cvo_dense_with_normal.forward(pts, pts_info, grid_source, grid_valid, pts_normal, grid_normal, pts_nres, grid_nres, neighbor_range, ell, mag_max, mag_min, ignore_ib, norm_in_dist, ell_basedist)
        else:
            outputs = cvo_dense_with_normal_output.forward(pts, pts_info, grid_source, grid_valid, pts_normal, grid_normal, pts_nres, grid_nres, neighbor_range, ell, mag_max, mag_min, ignore_ib, norm_in_dist, ell_basedist, return_nkern)
            y = outputs[0]
            nkern = outputs[1]
            save_nkern(nkern, pts, grid_source.shape, mag_max, mag_min, filename)

        ctx.save_for_backward(pts, pts_info, grid_source, grid_valid, pts_normal, grid_normal, pts_nres, grid_nres)
        ctx.neighbor_range = neighbor_range
        ctx.ell = ell
        ctx.mag_max = mag_max
        ctx.mag_min = mag_min
        ctx.ignore_ib = ignore_ib
        ctx.norm_in_dist = norm_in_dist
        ctx.ell_basedist =ell_basedist
        return y

    @staticmethod
    def backward(ctx, dy):
        pts, pts_info, grid_source, grid_valid, pts_normal, grid_normal, pts_nres, grid_nres = ctx.saved_tensors
        dy = dy.contiguous()
        dx1, dx2, dn1, dn2, dr1, dr2 = cvo_dense_with_normal.backward( \
            dy, pts, pts_info, grid_source, grid_valid, pts_normal, grid_normal, pts_nres, grid_nres, ctx.neighbor_range, ctx.ell, ctx.mag_max, ctx.mag_min, ctx.ignore_ib, ctx.norm_in_dist, ctx.ell_basedist)
        # return None, dx1, dx2, None, dn1, dn2, dr1, dr2, None, None, None, None, None, None, None, None, None
        # return None, dx1, dx2, None, None, None, dr1, dr2, None, None, None, None, None, None, None, None, None
        return None, dx1, dx2, None, dn1, dn2, None, None, None, None, None, None, None, None, None, None, None

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

def res_normal_dense(xyz, normal, K):
    """
    xyz: B*C*H*W, C=3
    normal: B*C*H*W (normalized), C=3
    K: cam intr
    """
    batch_size = xyz.shape[0]
    channel = xyz.shape[1]
    xyz_patches = F.unfold(xyz, kernel_size=3, padding=1).reshape(batch_size, channel, 9, -1)    # B*(C*9)*(H*W) -> B*C*9*(H*W)
    xyz_patch_proj = (xyz_patches * normal.reshape(batch_size, channel, 1, -1 )).sum(dim=1)  # B*9*(H*W)
    xyz_patch_proj_res = xyz_patch_proj[:, [0,1,2,3,5,6,7,8], :] - xyz_patch_proj[:,[4], :] # B*8*(H*W)
    xyz_patch_diff = ( xyz_patches[:, :, [0,1,2,3,5,6,7,8], :] - xyz_patches[:, :, [4], :] ).norm(dim=1)  # B*8*(H*W)
    
    xyz_patch_res_sin = ( xyz_patch_proj_res/ (xyz_patch_diff+1e-8) ).abs().mean(dim=1, keepdim=True).reshape(batch_size, 1, xyz.shape[2], xyz.shape[3]) # B*1*H*W between 0~1

    return xyz_patch_res_sin

class NormalFromDepthDense(torch.nn.Module):
    def __init__(self):
        super(NormalFromDepthDense, self).__init__()
        self.sobel_grad = SobelGrad()

    def forward(self, depth, K):
        grad_x, grad_y = self.sobel_grad(depth)
        normal = normal_from_grad(grad_x, grad_y, depth, K)
        # tan_x = tan_from_grad(grad_x, depth, K, mode="x")
        # tan_y = tan_from_grad(grad_x, depth, K, mode="y")
        # normal = normal_from_tan(tan_x, tan_y)
        return normal

class SobelGrad(torch.nn.Module):
    def __init__(self):
        super(SobelGrad, self).__init__()
        filter_shape = (1, 1, 3, 3) # out_c, in_c/group, kH, kW
        kern_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(filter_shape) / 8.0 # normalize so that the value is a real gradient delta(d)/delta(x)
        kern_y = kern_x.transpose(2,3)
        self.register_buffer("kern_x", kern_x)
        self.register_buffer("kern_y", kern_y)
        # self.pad_layer = torch.nn.ReflectionPad2d(1) # (left,right,top,bottom) or an int
        self.pad_layer = torch.nn.ReplicationPad2d(1)
        ## use a dedicated padding layer because the padding in F.conv2d only pads zeros.

    def forward(self, img):
        """expect the img channel to be 1 if used in NormalFromDepthDense, 
        Otherwise the return channel number is the same as input
        """
        img_pad = self.pad_layer(img)
        img_pad[:,:,-1, :] = 2 * img_pad[:,:,-2, :] - img_pad[:,:,-3, :]        ## so that the last row's vertical gradient is decided by the last two rows
        grad_x = torch.zeros_like(img)
        grad_y = torch.zeros_like(img)
        for ic in range(img.shape[1]):
            grad_x[:,ic:ic+1,:,:] = F.conv2d(img_pad[:,ic:ic+1,:,:], self.kern_x)
            grad_y[:,ic:ic+1,:,:] = F.conv2d(img_pad[:,ic:ic+1,:,:], self.kern_y)
        
        return grad_x, grad_y
            
def tan_from_grad(grad, depth, K, mode):
    """
    grad: grad_x or grad_y, B*1*H*W
    depth: B*1*H*W
    K: cam intrinsic mat
    mode: "x" or "y"
    """
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    y_range = torch.arange(grad.shape[2], device=grad.device, dtype=grad.dtype) # height, y, v
    x_range = torch.arange(grad.shape[3], device=grad.device, dtype=grad.dtype) # width, x, u
    grid_y, grid_x = torch.meshgrid(y_range, x_range) ## [height * width]
    
    ## x_hat and y_hat
    x_hat = (grid_x - cx) / fx  # [h*w]
    y_hat = (grid_y - cy) / fy

    if mode== "x":
        tan_0 = x_hat * grad + depth / fx     #B*1*H*W
        tan_1 = y_hat * grad
        tan_2 = grad
        tan = torch.cat( (tan_0, tan_1, tan_2), dim=1 ) # B*3*H*W
    elif mode== "y":
        tan_0 = x_hat * grad    #B*1*H*W
        tan_1 = y_hat * grad + depth / fy
        tan_2 = grad
        tan = torch.cat( (tan_0, tan_1, tan_2), dim=1 ) # B*3*H*W
    else:
        raise ValueError("mode {} not recognized".format(mode))

def normal_from_tan(tan_x, tan_y):
    normal = torch.cross(tan_x, tan_y, dim=1) #B*3*H*W
    return F.normalize(normal, p=2, dim=1)

def normal_from_grad(grad_x, grad_y, depth, K):
    """grad_x: B*1*H*W"""
    y_range = torch.arange(grad_x.shape[2], device=grad_x.device, dtype=grad_x.dtype) # height, y, v
    x_range = torch.arange(grad_x.shape[3], device=grad_x.device, dtype=grad_x.dtype) # width, x, u
    grid_y, grid_x = torch.meshgrid(y_range, x_range) ## H*W

    fx = K[:,0,0].reshape(-1, 1, 1, 1) # B*1*1*1
    fy = K[:,1,1].reshape(-1, 1, 1, 1)
    cx = K[:,0,2].reshape(-1, 1, 1, 1)
    cy = K[:,1,2].reshape(-1, 1, 1, 1)

    normal_0 = -fy * grad_x
    normal_1 = -fx * grad_y
    normal_2 = (grid_x - cx) * grad_x + (grid_y - cy) * grad_y + depth
    normal = torch.cat([normal_0, normal_1, normal_2], dim=1)
    return F.normalize(normal, p=2, dim=1)

def grid_from_concat_flat_func(uvb_split, flat_info, grid_shape):
    """
    uvb_split: a tuple of 3 elements of tensor N*1, only long/byte/bool tensors can be used as indices
    flat_info: 1*C*N
    grid_shape: [B,C,H,W]
    """
    C_info = flat_info.shape[1]
    grid_info = torch.zeros((grid_shape[0], C_info, grid_shape[2], grid_shape[3]), dtype=flat_info.dtype, device=flat_info.device) # B*C*H*W
    flat_info_t = flat_info.squeeze(0).transpose(0,1).unsqueeze(1) # N*1*C
    # print(max(uvb_split[2]), max(uvb_split[1]), max(uvb_split[0]))
    grid_info[uvb_split[2], :, uvb_split[1], uvb_split[0]] = flat_info_t
    return grid_info

def save_nkern(nkern, pts, grid_shape, mag_max, mag_min, filename):
    """nkern is a 1*NN*N tensor, need to turn it into grid and save as image"""
    pts_coords = pts.to(dtype=torch.long).squeeze(0).transpose(0,1).split(1,dim=1)
    list_spn = []

    dim_n = int(np.sqrt(nkern.shape[1]))
    half_dim_n = int( (dim_n-1)/2 )
    list_spn.append( half_dim_n )

    mid_n = ( nkern.shape[1]-1 ) / 2
    mid_n = int(mid_n)
    list_spn.append(mid_n)
    list_spn.append(mid_n - half_dim_n)
    list_spn.append(mid_n + half_dim_n)
    list_spn.append( int(nkern.shape[1] - 1 - half_dim_n) )

    for spn in list_spn:
        nkern_center = nkern[:, spn:spn+1, :] ## only take one slice 1*1*N
        nkern_center_grid = grid_from_concat_flat_func(pts_coords, nkern_center, grid_shape) # B*1*H*W

        nkern_center_grid = nkern_center_grid.squeeze(1).cpu().numpy() # B*H*W
        nkern_center_grid = (nkern_center_grid / mag_max * 255).astype(np.uint8) # normalize to 0-255

        for ib in range(nkern_center_grid.shape[0]):
            img = Image.fromarray(nkern_center_grid[ib])
            img.save( "{}_{}_{}.png".format(filename, spn, ib) )

def save_tensor_to_img(tsor, filename, mode):
    """Input is B*C*H*W"""
    nparray = tsor.cpu().detach().numpy()
    nparray = nparray.transpose(0,2,3,1)
    if mode =="rgb":
        nparray = (nparray * 255).astype(np.uint8)
        Imode = "RGB"
    elif mode == "dep":
        nparray = (nparray[:,:,:,0] /nparray.max() * 255).astype(np.uint8)
        Imode = "L"
    elif "nml" in mode:
        nparray = (nparray * 255).astype(np.uint8)
        Imode = "RGB"
    else:
        raise ValueError("mode {} not recognized".format(mode))
    for ib in range(nparray.shape[0]):
        img = Image.fromarray(nparray[ib], mode=Imode)
        img.save("{}_{}_{}.png".format(filename, mode, ib))