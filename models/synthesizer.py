import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torchvision import models
import sys
import math
from torch.nn.utils import spectral_norm
from models.antialias import Downsample
from pytorch_memlab import profile, MemReporter
try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)

def sample_gumbel(x):
    noise = torch.cuda.FloatTensor(x.size()).uniform_()
    eps = 1e-20
    noise.add_(eps).log_().neg_()
    noise.add_(eps).log_().neg_()
    return noise

def gumbel_softmax_sample(x, tau=0.2, hard=True):
    noise = sample_gumbel(x)
    y = (torch.log(x+1e-20) + noise) / tau
    ysft = F.softmax(y,dim=1)
    if hard:
        max_v, max_idx = ysft.max(dim=1,keepdim=True)
        one_hot = ysft.data.new(ysft.size()).zero_().scatter_(1, max_idx.data, ysft.data.new(max_idx.size()).fill_(1.)) - ysft.data
        # Which is the right way to do this?
        y_out = one_hot.detach() + ysft
        #y_out = one_hot + ysft
        return y_out.view_as(x)
    return ysft.view_as(x)


def snconv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1, padding=0, dilation=1, bias=False, use_sn=True):
    if use_sn:
        return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation))
    else:
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation)

def snlinear(indim, outdim, bias=False, use_sn=True):
    if use_sn:
        return spectral_norm(nn.Linear(indim, outdim, bias=bias))
    else:
        return nn.Linear(indim, outdim, bias=bias)

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_channels, use_sn=True):
        super(Self_Attn, self).__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0, use_sn=use_sn)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0, use_sn=use_sn)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0, use_sn=use_sn)
        self.snconv1x1_attn = snconv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0, use_sn=use_sn)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax  = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch//8, h*w)
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch//8, h*w//4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch//2, h*w//4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.snconv1x1_attn(attn_g)
        # Out
        out = x + self.sigma*attn_g
        return out


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_norm_2d(input, adaptive_params):
    size = input.shape
    nFeat = size[1]
    if len(adaptive_params.shape) == 2:
        adaptive_params = adaptive_params.view(adaptive_params.shape[0], adaptive_params.shape[1],1,1)
        adaptive_params = adaptive_params.expand(adaptive_params.shape[0], adaptive_params.shape[1], size[2], size[3])
    elif (adaptive_params.shape[2] != size[2] or adaptive_params.shape[3] != size[3]):
        adaptive_params = F.interpolate(adaptive_params, (size[2],size[3]), mode='nearest')
    w = adaptive_params[:,:nFeat,::]
    b = adaptive_params[:,nFeat:,::]
    return (w+1.)*F.instance_norm(input) + b

def get_conv_inorm_relu_block(i, o, k, s, p, slope=0.1, padtype='zero', dilation=1, affine=True, use_antialias=False):
    layers = []
    if padtype == 'reflection':
        layers.append(nn.ReflectionPad2d(p)); p=0
    elif padtype == 'replication':
        layers.append(nn.ReplicationPad2d(p)); p=0
    if use_antialias and s >1:
        aa_str = s
        s = 1
    layers.append(nn.Conv2d(i, o, kernel_size=k, stride=s, padding=p, dilation=dilation, bias=False))
    layers.append(nn.InstanceNorm2d(o, affine=affine))
    layers.append(nn.LeakyReLU(slope,inplace=True))
    if use_antialias and s >1:
        layers.append(Downsample(channels=o, filt_size=3, stride=aa_str))
    return layers

def get_conv_bnorm_relu_block(i, o, k, s, p, slope=0.1, padtype='zero', dilation=1, affine=True, use_antialias=False):
    layers = []
    if padtype == 'reflection':
        layers.append(nn.ReflectionPad2d(p)); p=0
    elif padtype == 'replication':
        layers.append(nn.ReplicationPad2d(p)); p=0
    if use_antialias and s >1:
        aa_str = s
        s = 1
    layers.append(nn.Conv2d(i, o, kernel_size=k, stride=s, padding=p, dilation=dilation, bias=False))
    layers.append(nn.BatchNorm2d(o, affine=affine))
    layers.append(nn.LeakyReLU(slope, inplace=True))
    if use_antialias and s >1:
        layers.append(Downsample(channels=o, filt_size=3, stride=aa_str))
    return layers

def get_conv_relu_block(i, o, k, s, p, slope=0.2, padtype='zero', dilation=1, use_antialias=False):
    layers = []
    if padtype == 'reflection':
        layers.append(nn.ReflectionPad2d(p)); p=0
    elif padtype == 'replication':
        layers.append(nn.ReplicationPad2d(p)); p=0
    if use_antialias and s >1:
        aa_str = s
        s = 1
    layers.append(nn.Conv2d(i, o, kernel_size=k, stride=s, padding=p, dilation=dilation, bias=False))
    layers.append(nn.LeakyReLU(slope,inplace=True))
    if use_antialias and s >1:
        layers.append(Downsample(channels=o, filt_size=3, stride=aa_str))
    return layers


class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.weight = nn.Parameter(0.01*torch.randn(1, channel, 1, 1))

    def forward(self, feat, noise = None):
        if noise is None:
            noise= torch.randn(feat.shape[0], 1, feat.shape[2], feat.shape[3], dtype= feat.dtype, device=feat.device)
        return feat + self.weight * noise

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels=3, kernel_size=3, sigma=3, dim=2):
        super(GaussianSmoothing, self).__init__()
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
        mean = (kernel_size - 1)/2.
        variance = sigma**2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1./(2.*math.pi*variance)) *\
                          torch.exp(
                              -torch.sum((xy_grid - mean)**2., dim=-1) /\
                              (2*variance)
                          )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        self.gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, groups=channels, bias=False, padding=(kernel_size-1)//2)

        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.gaussian_filter(input)



class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False, only_last = False, final_feat_size=8):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.only_last = only_last
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            if final_feat_size <=64 or type(vgg_pretrained_features[x]) is not nn.MaxPool2d:
                self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            if final_feat_size <=32 or type(vgg_pretrained_features[x]) is not nn.MaxPool2d:
                self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            if final_feat_size <=16 or type(vgg_pretrained_features[x]) is not nn.MaxPool2d:
                self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            if final_feat_size <=8 or type(vgg_pretrained_features[x]) is not nn.MaxPool2d:
                self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        if self.only_last:
            return h_relu5
        else:
            out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
            return out

class AdaptiveScaleTconv(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out, scale=2, use_deform=True, n_filters=1):
        super(AdaptiveScaleTconv, self).__init__()
        if int(torch.__version__.split('.')[1])<4:
            self.upsampLayer = nn.Upsample(scale_factor=scale, mode='bilinear')
        else:
            self.upsampLayer = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)

        if n_filters > 1:
            self.convFilter = nn.Sequential(*[nn.Conv2d(dim_in if i==0 else dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False) for i in xrange(n_filters)])
        else:
            self.convFilter = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.use_deform = use_deform
        if use_deform:
            self.coordfilter = nn.Conv2d(dim_in, 2, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
            self.coordfilter.weight.data.zero_()
        #Identity transform used to create a regular grid!

    def forward(self, x, extra_inp=None):
        # First upsample the input with transposed/ upsampling
        # Compute the warp co-ordinates using a conv
        # Warp the conv
        up_out = self.upsampLayer(x)
        filt_out = self.convFilter(up_out if extra_inp is None else torch.cat([up_out,extra_inp], dim=1))

        if self.use_deform:
            cord_offset = self.coordfilter(up_out)
            reg_grid = Variable(torch.FloatTensor(np.stack(np.meshgrid(np.linspace(-1,1, up_out.size(2)), np.linspace(-1,1, up_out.size(3))))).cuda(),requires_grad=False)
            deform_grid = reg_grid.detach() + torch.tanh(cord_offset)
            deformed_out = F.grid_sample(filt_out, deform_grid.transpose(1,3).transpose(1,2), mode='bilinear', padding_mode='zeros')
            feat_out = (deform_grid, reg_grid, cord_offset)
        else:
            deformed_out = filt_out
            feat_out = []

        #Deformed out
        return deformed_out, feat_out

class DeformableLayer(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out, use_deform=True, n_filters=1):
        super(DeformableLayer, self).__init__()

        if n_filters > 1:
            self.convFilter = nn.Sequential(*[nn.Conv2d(dim_in if i==0 else dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False) for i in xrange(n_filters)])
        else:
            self.convFilter = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.use_deform = use_deform
        if use_deform:
            self.coordfilter = nn.Conv2d(dim_in, 2, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
            self.coordfilter.weight.data.zero_()
        #Identity transform used to create a regular grid!

    def forward(self, x, extra_inp=None):
        # First upsample the input with transposed/ upsampling
        # Compute the warp co-ordinates using a conv
        # Warp the conv
        filt_out = self.convFilter(x if extra_inp is None else torch.cat([x, extra_inp], dim=1))

        if self.use_deform:
            cord_offset = self.coordfilter(x)
            reg_grid = Variable(torch.FloatTensor(np.stack(np.meshgrid(np.linspace(-1,1, x.size(2)), np.linspace(-1,1, x.size(3))))).cuda(),requires_grad=False)
            deform_grid = reg_grid.detach() + torch.tanh(cord_offset)
            deformed_out = F.grid_sample(filt_out, deform_grid.transpose(1,3).transpose(1,2), mode='bilinear', padding_mode='zeros')
            feat_out = (deform_grid, reg_grid, cord_offset)
        else:
            deformed_out = filt_out
            feat_out = []

        #Deformed out
        return deformed_out#, feat_out

def mls_rigid_deformation_inv(image, p, q, density=1.0, weight='euclid'):
    ''' Rigid inverse deformation
    ### Params:
        * image - ndarray: original image
        * p - ndarray: an array with size [n, 2], original control points
        * q - ndarray: an array with size [n, 2], final control points
        * alpha - float: parameter used by weights
        * density - float: density of the grids
    ### Return:
        A deformed image.
    '''
    #image = torch.FloatTensor(image)
    kps = p['code']['mean']
    kpt = q['code']['mean']
    bsz = kps.shape[0]

    height = image.shape[-1]
    width = image.shape[-2]
    grid = make_coordinate_grid(image.shape[2:], image.type())
    #
    #kps = torch.cat([(kps[:,:,:1]+1.)*width/2., (kps[:,:,1:]+1.)*height/2.], dim=-1)
    #kpt = torch.cat([(kpt[:,:,:1]+1.)*width/2., (kpt[:,:,1:]+1.)*height/2.], dim=-1)


    # Change (x, y) to (row, col)
    #k = q[:, [1, 0]]
    #p = p[:, [1, 0]]

    # Make grids on the original image
    #gridX = torch.linspace(0, width-1, steps=int(width*density), device=kps.device)
    #gridY = torch.linspace(0, height-1, steps=int(height*density), device=kps.device)
    #vy, vx = torch.meshgrid(gridX, gridY)
    grow = width #vx.shape[0]  # grid rows
    gcol = height # grid cols
    ctrls = kps.shape[1]  # control points

    # Compute
    reshaped_p = kps.reshape(bsz, ctrls, 2, 1, 1)                                           # [ctrls, 2, 1, 1]
    reshaped_q = kpt.reshape(bsz, ctrls, 2, 1, 1)                                           # [ctrls, 2, 1, 1]
    #reshaped_v = torch.cat((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))         # [2, grow, gcol]
    reshaped_v = grid.permute(2,0,1)

    if weight == 'euclid':
        w = 1.0 / (torch.sum((reshaped_p - reshaped_v) ** 2, dim=2) + 1e-12)                                        # [ctrls, grow, gcol]
    elif weight == 'gauss':
        var = p['code']['var']
        dist = torch.sum((reshaped_p - reshaped_v) ** 2, dim=2) + 1e-12
        w = 1.0/(dist * var)
    elif weight == 'allvarAndStr':
        var1 = p['code']['var']
        var2 = q['code']['var']
        str1 = p['str'].view(bsz,ctrls,1,1)
        str2 = q['str'].view(bsz,ctrls,1,1)
        dist = torch.sum((reshaped_p - reshaped_v) ** 2, dim=2) + 1e-12
        w = (1.0/(dist * var1*var2))*str1*str2
    #w[w == np.inf] = 2**31 - 1
    pstar = (torch.sum(w * reshaped_p.permute(2, 0, 1, 3, 4), dim=2) / torch.sum(w, dim=1)).permute(1,0,2,3)    # [2, grow, gcol]
    phat = reshaped_p - pstar[:,None,::]                                                                        # [ctrls, 2, grow, gcol]
    qstar = (torch.sum(w * reshaped_q.permute(2, 0, 1, 3, 4), dim=2) / torch.sum(w, dim=1)).permute(1,0,2,3)    # [2, grow, gcol]
    qhat = reshaped_q - qstar[:,None,::]                                                                        # [ctrls, 2, grow, gcol]
    reshaped_phat1 = phat.reshape(bsz, ctrls, 1, 2, grow, gcol)                                                 # [ctrls, 1, 2, grow, gcol]
    reshaped_phat2 = phat.reshape(bsz, ctrls, 2, 1, grow, gcol)                                                 # [ctrls, 2, 1, grow, gcol]
    reshaped_qhat = qhat.reshape(bsz, ctrls, 1, 2, grow, gcol)                                                  # [ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(bsz, ctrls, 1, 1, grow, gcol)                                                        # [ctrls, 1, 1, grow, gcol]

    mu = torch.sum(torch.matmul(reshaped_w.permute(0, 1, 4, 5, 2, 3) *
                          reshaped_phat1.permute(0, 1, 4, 5, 2, 3),
                          reshaped_phat2.permute(0, 1, 4, 5, 2, 3)), dim=1)                 # [grow, gcol, 1, 1]
    reshaped_mu = mu.reshape(bsz, 1, grow, gcol)                                            # [1, grow, gcol]
    neg_phat_verti = phat[:, :, [1, 0],...]                                                 # [ctrls, 2, grow, gcol]
    neg_phat_verti[:, :, 1,...] = -neg_phat_verti[:, :, 1,...]
    reshaped_neg_phat_verti = neg_phat_verti.reshape(bsz, ctrls, 1, 2, grow, gcol)          # [ctrls, 1, 2, grow, gcol]
    mul_right = torch.cat((reshaped_phat1, reshaped_neg_phat_verti), dim=2)                 # [ctrls, 2, 2, grow, gcol]
    mul_left = reshaped_qhat * reshaped_w                                                   # [ctrls, 1, 2, grow, gcol]
    Delta = torch.sum(torch.matmul(mul_left.permute(0, 1, 4, 5, 2, 3),
                             mul_right.permute(0, 1, 4, 5, 2, 3)),
                   dim=1).permute(0, 1, 2, 4, 3)                                            # [grow, gcol, 2, 1]
    Delta_verti = Delta[...,[1, 0],:]                                                       # [grow, gcol, 2, 1]
    Delta_verti[...,0,:] = -Delta_verti[...,0,:]
    B = torch.cat((Delta, Delta_verti), dim=-1)                                             # [grow, gcol, 2, 2]
    inv_B = B.transpose(-2,-1) / (((B[:,:,:,0,:]**2).sum(dim=-1)+1e-12).unsqueeze(-1).unsqueeze(-1))
    flag = False
    #try:
    #    inv_B = torch.inverse(B)                                                           # [grow, gcol, 2, 2]
    #    flag = False
    #except np.linalg.linalg.LinAlgError:
    #    flag = True
    #    det = torch.det(B)                                                                 # [grow, gcol]
    #    det[det < 1e-8] = np.inf
    #    reshaped_det = det.reshape(grow, gcol, 1, 1)                                       # [grow, gcol, 1, 1]
    #    adjoint = B[:,:,[[1, 0], [1, 0]], [[1, 1], [0, 0]]]                                # [grow, gcol, 2, 2]
    #    adjoint[:,:,[0, 1], [1, 0]] = -adjoint[:,:,[0, 1], [1, 0]]                         # [grow, gcol, 2, 2]
    #    inv_B = (adjoint / reshaped_det).permute(2, 3, 0, 1)                               # [2, 2, grow, gcol]

    vqstar = reshaped_v[None,::] - qstar                                                    # [2, grow, gcol]
    reshaped_vqstar = vqstar.reshape(bsz, 1, 2, grow, gcol)                                 # [1, 2, grow, gcol]

    # Get final image transfomer -- 3-D array
    temp = torch.matmul(reshaped_vqstar.permute(0, 3, 4, 1, 2),
                     inv_B).reshape(bsz, grow, gcol, 2).permute(0, 3, 1, 2)                 # [2, grow, gcol]
    norm_temp = torch.norm(temp, dim=1, keepdim=True) + 1e-12                               # [1, grow, gcol]
    norm_vqstar = torch.norm(vqstar, dim=1, keepdim=True) + 1e-12                           # [1, grow, gcol]
    transformers = temp / norm_temp * norm_vqstar + pstar                                   # [2, grow, gcol]

    # Correct the points where pTwp is singular
    if flag:
        blidx = det == np.inf    # bool index
        transformers[0][blidx] = vx[blidx] + qstar[0][blidx] - pstar[0][blidx]
        transformers[1][blidx] = vy[blidx] + qstar[1][blidx] - pstar[1][blidx]

    # Removed the points outside the border
    #if torch.isnan(transformers).any():
    #    import ipdb; ipdb.set_trace()
    transformers[transformers < -1] = -1
    transformers[transformers > 1] = 1
    transformers[torch.isnan(transformers)] = 0

    # Mapping original image
    #transformed_image = image[tuple(transformers.long().transpose(1,2).data.numpy())]    # [grow, gcol]
    transformed_image =  F.grid_sample(image, transformers.permute(0,2,3,1),mode='bilinear')

    # Rescale image
    #transformed_image = rescale(transformed_image.data.numpy(), scale=1.0 / density, mode='reflect')

    return transformed_image

def map_feats_to_keypoints(feat, kp_src, kp_type=0, noise_params=None, heidelberg_version=False):
    # Src map with
    kp_src_img = kp_src['img'] if not heidelberg_version else kp_src['heatmap']
    #if kp_type == 2:
    #    kp_src_img = kp_src_img[:,:-1,::]
    kp_src_img  = F.adaptive_avg_pool2d(kp_src_img,  (feat.shape[2], feat.shape[3]))

    bsz = kp_src_img.shape[0]
    n_kp = kp_src_img.shape[1]
    n_feat = feat.shape[1]

    if not heidelberg_version:
        weight_feat = torch.matmul(kp_src_img.view(bsz,n_kp,-1), feat.view(bsz,n_feat,-1).permute(0,2,1))/ (1+kp_src_img.sum(dim=(2,3))).unsqueeze(-1)
    else:
        weight_feat = torch.matmul(kp_src_img.view(bsz,n_kp,-1), feat.view(bsz,n_feat,-1).permute(0,2,1))
    #import ipdb; ipdb.set_trace()
    if noise_params is not None:
        noise = torch.randn_like(weight_feat)*noise_params[1,:].view(1,-1,1) + noise_params[0,:].view(1,-1,1)
        weight_feat = kp_src['str'].unsqueeze(-1) * weight_feat +  (1. - kp_src['str'].unsqueeze(-1)) * noise
    #if kp_type ==2:
    #    weight_feat = weight_feat /  (kp_src_img.sum(dim=2).sum(dim=2) + 1e-2)[:,:,None]

    return weight_feat

def remap_kpfeat_to_target(weight_feat, kp_dest, shape, kp_type=0, heidelberg_version=False):
    # Src map with
    kp_dest_img = kp_dest['img']
    #if kp_type == 2:
    #    kp_dest_img = kp_dest_img[:,:-1,::]
    kp_dest_img = F.adaptive_avg_pool2d(kp_dest_img, (shape[0], shape[1]))

    bsz = kp_dest_img.shape[0]
    n_kp = kp_dest_img.shape[1]
    n_feat = weight_feat.shape[2]

    if heidelberg_version:
        normed_targ = (kp_dest_img/(1+kp_dest_img.sum(dim=1).unsqueeze(1)))
        remaped_feat = torch.matmul(weight_feat.permute(0,2,1), normed_targ.view(bsz,n_kp,-1)).view(bsz, n_feat, shape[0], shape[1])
    else:
        remaped_feat = torch.matmul(weight_feat.permute(0,2,1), kp_dest_img.view(bsz,n_kp,-1)).view(bsz, n_feat, shape[0], shape[1])

    return remaped_feat

def remap_feats_with_keypoints(feat, kp_src, kp_dest, kp_type=0, noise_params=None, heidelberg_version=False):
    #import ipdb; ipdb.set_trace()
    # Src map with
    kp_src_img = kp_src['img'] if not heidelberg_version else kp_src['heatmap']
    kp_dest_img = kp_dest['img']
    #if kp_type == 2:
    #    kp_src_img = kp_src_img[:,:-1,::]
    #    kp_dest_img = kp_dest_img[:,:-1,::]
    kp_src_img  = F.adaptive_avg_pool2d(kp_src_img,  (feat.shape[2], feat.shape[3]))
    kp_dest_img = F.adaptive_avg_pool2d(kp_dest_img, (feat.shape[2], feat.shape[3]))

    bsz = kp_src_img.shape[0]
    n_kp = kp_src_img.shape[1]
    n_feat = feat.shape[1]

    if not heidelberg_version:
        weight_feat = torch.matmul(kp_src_img.view(bsz,n_kp,-1), feat.view(bsz,n_feat,-1).permute(0,2,1))/ (1+kp_src_img.sum(dim=(2,3))).unsqueeze(-1)
    else:
        weight_feat = torch.matmul(kp_src_img.view(bsz,n_kp,-1), feat.view(bsz,n_feat,-1).permute(0,2,1))
    #import ipdb; ipdb.set_trace()
    if noise_params is not None:
        noise = torch.randn_like(weight_feat)*noise_params[1,:].view(1,-1,1) + noise_params[0,:].view(1,-1,1)
        weight_feat = kp_src['str'].unsqueeze(-1) * weight_feat +  (1. - kp_src['str'].unsqueeze(-1)) * noise
    #if kp_type ==2:
    #    weight_feat = weight_feat /  (kp_src_img.sum(dim=2).sum(dim=2) + 1e-2)[:,:,None]
    if heidelberg_version:
        normed_targ = (kp_dest_img/(1+kp_dest_img.sum(dim=1).unsqueeze(1)))
        remaped_feat = torch.matmul(weight_feat.permute(0,2,1), normed_targ.view(bsz,n_kp,-1)).view(bsz, n_feat, feat.shape[2], feat.shape[3])
    else:
        remaped_feat = torch.matmul(weight_feat.permute(0,2,1), kp_dest_img.view(bsz,n_kp,-1)).view(bsz, n_feat, feat.shape[2], feat.shape[3])

    return remaped_feat

class ResidualProjectBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in,filters, dilation=1, padtype = 'zero',down_samp_factor=2):
        super(ResidualProjectBlock, self).__init__()
        layers = []
        pad=dilation
        #if padtype == 'reflection':
        #    layers.append(nn.ReflectionPad2d(pad)); pad=0
        #elif padtype == 'replication':
        #    layers.append(nn.ReplicationPad2d(pad)); pad=0
        st = down_samp_factor

        layers.extend([
            nn.Conv2d(dim_in, filters[0], kernel_size=1, stride=st, padding=0, dilation=dilation, bias=False),
            nn.BatchNorm2d(filters[0], affine=True),
            nn.ReLU(inplace=True)
            ])

        layers.extend([
            nn.Conv2d(filters[0], filters[1], kernel_size=3, stride=1, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(filters[1], affine=True),
            nn.ReLU(inplace=True)
            ])

        layers.extend([
            nn.Conv2d(filters[1], filters[2], kernel_size=1, stride=1, padding=0, dilation=dilation, bias=False),
            nn.BatchNorm2d(filters[2], affine=True),
            ])
        self.shortcut = nn.Conv2d(dim_in, filters[2], kernel_size=1, stride=st, padding=0, dilation=dilation, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.relu(self.shortcut(x) + self.main(x))

class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dilation=1, padtype = 'zero'):
        super(ResidualBlock, self).__init__()
        pad = dilation
        layers = []
        if padtype== 'reflection':
            layers.append(nn.ReflectionPad2d(pad)); pad=0
        elif padtype == 'replication':
            layers.append(nn.ReplicationPad2d(pad)); pad=0

        layers.extend([ nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=1, padding=pad, dilation=dilation, bias=False),
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.1,inplace=True)])

        pad = dilation
        if padtype== 'reflection':
            layers.append(nn.ReflectionPad2d(pad)); pad=0
        elif padtype == 'replication':
            layers.append(nn.ReplicationPad2d(pad)); pad=0

        layers.extend([
            nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=1, padding=pad, dilation=dilation, bias=False),
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.1,inplace=True)
            ])

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.main(x)

class ResidualBlockBnorm(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dilation=1, padtype = 'zero'):
        super(ResidualBlockBnorm, self).__init__()
        pad = dilation
        layers = []
        if padtype == 'reflection':
            layers.append(nn.ReflectionPad2d(pad)); pad=0
        elif padtype == 'replication':
            layers.append(nn.ReplicationPad2d(pad)); pad=0

        layers.extend([ nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=1, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2,inplace=True)])

        pad = dilation
        if padtype== 'reflection':
            layers.append(nn.ReflectionPad2d(pad)); pad=0
        elif padtype == 'replication':
            layers.append(nn.ReplicationPad2d(pad)); pad=0

        layers.extend([
            nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=1, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2,inplace=True)
            ])

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.main(x)

class ResidualBlockNoNorm(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dilation=1, padtype = 'zero', apply_noise=False, use_bias=False, no_residual=False, out_dim=None, use_sn=False):
        super(ResidualBlockNoNorm, self).__init__()
        pad = dilation
        layers = []
        self.proj = False
        self.no_residual = no_residual
        if out_dim == None:
            out_dim = dim_in
        elif out_dim != dim_in and (not self.no_residual):
            self.proj = True
            self.projConv = snconv2d(dim_in, out_dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False, use_sn=use_sn)

        if padtype == 'reflection':
            layers.append(nn.ReflectionPad2d(pad)); pad=0
        elif padtype == 'replication':
            layers.append(nn.ReplicationPad2d(pad)); pad=0

        layers.extend([ snconv2d(dim_in, out_dim, kernel_size=3, stride=1, padding=pad, dilation=dilation, bias=use_bias, use_sn=use_sn),
            nn.LeakyReLU(0.2,inplace=True)])

        pad = dilation
        if padtype== 'reflection':
            layers.append(nn.ReflectionPad2d(pad)); pad=0
        elif padtype == 'replication':
            layers.append(nn.ReplicationPad2d(pad)); pad=0

        layers.extend([
            snconv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=pad, dilation=dilation, bias=use_bias, use_sn=use_sn),
            #nn.LeakyReLU(0.1,inplace=True)
            ])
        if apply_noise:
            layers.append(NoiseInjection(out_dim))
        # Note the last leaky relu is removed so that instance norm can be applied before non-linearity

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        proj_x = x if not self.proj else self.projConv(x)
        return self.main(x) if self.no_residual else proj_x + self.main(x)

class GeneratorLocPred(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, out_size=8, cond_input=False, use_drop=False):
        super(GeneratorLocPred, self).__init__()

        # out_size has to be a power of 2. (8, 16, 32)

        layers = []
        layers.append(nn.Conv2d(6, conv_dim, kernel_size=7, stride=2, padding=3, bias=False))
        layers.append(nn.BatchNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))
        if use_drop:
            layers.append(nn.Dropout(p=0.25))
        #layers.append(nn.MaxPool2d(3,stride=2))

        # Down-Sampling
        curr_dim = conv_dim
        layers.append(ResidualProjectBlock(64,[64,64,128]))
        if use_drop:
            layers.append(nn.Dropout(p=0.25))
        layers.append(ResidualProjectBlock(128,[64,64,128],   down_samp_factor = 2 if out_size < 32 else 1))
        if use_drop:
            layers.append(nn.Dropout(p=0.25))
        layers.append(ResidualProjectBlock(128,[128,128,512], down_samp_factor = 2 if out_size < 16 else 1))
        if use_drop:
            layers.append(nn.Dropout(p=0.25))

        self.sharedLayers = nn.Sequential(*layers)
        self.c_dim = c_dim

        locBranch = []
        dil = 2
        self.cond_input = cond_input
        loc_out_size = c_dim if not cond_input else 1
        locBranch.append(nn.Conv2d(512 + (c_dim if cond_input else 0), 64, kernel_size=3, dilation=dil, padding=dil, bias=False))
        locBranch.append(nn.ReLU(inplace=True))
        locBranch.append(nn.Conv2d(64, loc_out_size, kernel_size=3, dilation=dil, padding=dil, bias=False))
        self.locBranch = nn.Sequential(*locBranch)

        self.szConv = nn.Sequential(*[nn.Conv2d(512+ (c_dim if cond_input else 0), 512, kernel_size=3, dilation=2, padding=2, bias=False), nn.ReLU(inplace=True)])
        szBins = int(out_size*out_size * c_dim) if not cond_input else int(out_size*out_size)
        self.szOut = nn.Sequential(*[
            nn.Conv2d(512, szBins, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(szBins, szBins, kernel_size=1, padding=0, bias=True)
            ])

    def forward(self, x, locIdx, mode='train',query_class=None):
        # replicate spatially and concatenate domain information
        sharedFeat = self.sharedLayers(x)
        if self.cond_input:
            c = query_class.unsqueeze(2).unsqueeze(3)
            c = c.expand(c.size(0), c.size(1), sharedFeat.size(2), sharedFeat.size(3))
            sharedFeat = torch.cat([sharedFeat, c],dim=1)

        locScores = self.locBranch(sharedFeat)

        if mode == 'train':
            szConvOut = self.szConv(sharedFeat)
            bsz = szConvOut.shape[0]
            szRoiOut = torch.cat([F.adaptive_avg_pool2d(szConvOut[i,:,::][:,np.maximum(locIdx[1][i].data[0]-1,0):locIdx[1][i].data[0]+2, np.maximum(locIdx[0][i].data[0]-1,0):locIdx[0][i].data[0]+2][None,::], 1) for i in xrange(bsz)],dim=0)
            szScores = self.szOut(szRoiOut).view(bsz,self.c_dim if not self.cond_input else 1,-1)
        else:
            szScores = sharedFeat
        return locScores, szScores

    def forward_boxpred(self, sharedFeat, locIdx):
        szConvOut = self.szConv(sharedFeat)
        bsz = locIdx[0].shape[0]
        szRoiOut = torch.cat([F.adaptive_avg_pool2d(szConvOut[:,:,np.maximum(locIdx[1][i].item()-1,0):locIdx[1][i].item()+2, np.maximum(locIdx[0][i].item()-1,0):locIdx[0][i].item()+2], 1) for i in xrange(bsz)],dim=0)
        szScores = self.szOut(szRoiOut).view(bsz,self.c_dim if not self.cond_input else 1,-1)
        return szScores


def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    #x = x / (w - 1)
    #y = y / (h - 1)
    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed

def matrix_inverse(batch_of_matrix, eps=0):
    if eps != 0:
        init_shape = batch_of_matrix.shape
        a = batch_of_matrix[..., 0, 0].unsqueeze(-1)
        b = batch_of_matrix[..., 0, 1].unsqueeze(-1)
        c = batch_of_matrix[..., 1, 0].unsqueeze(-1)
        d = batch_of_matrix[..., 1, 1].unsqueeze(-1)

        det = a * d - b * c
        out = torch.cat([d, -b, -c, a], dim=-1)
        eps = torch.tensor(eps).type(out.type())
        out /= det.max(eps)

        return out.view(init_shape)
    else:
        b_mat = batch_of_matrix
        eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
        b_inv, _ = torch.solve(eye, b_mat)
        return b_inv


def matrix_det(batch_of_matrix):
    a = batch_of_matrix[..., 0, 0].unsqueeze(-1)
    b = batch_of_matrix[..., 0, 1].unsqueeze(-1)
    c = batch_of_matrix[..., 1, 0].unsqueeze(-1)
    d = batch_of_matrix[..., 1, 1].unsqueeze(-1)

    det = a * d - b * c
    return det


def matrix_trace(batch_of_matrix):
    a = batch_of_matrix[..., 0, 0].unsqueeze(-1)
    d = batch_of_matrix[..., 1, 1].unsqueeze(-1)

    return a + d


def smallest_singular(batch_of_matrix):
    a = batch_of_matrix[..., 0, 0].unsqueeze(-1)
    b = batch_of_matrix[..., 0, 1].unsqueeze(-1)
    c = batch_of_matrix[..., 1, 0].unsqueeze(-1)
    d = batch_of_matrix[..., 1, 1].unsqueeze(-1)

    s1 = a ** 2 + b ** 2 + c ** 2 + d ** 2
    s2 = (a ** 2 + b ** 2 - c ** 2 - d ** 2) ** 2
    s2 = torch.sqrt(s2 + 4 * (a * c + b * d) ** 2)

    norm = torch.sqrt((s1 - s2) / 2)
    return norm

def stddev2Linv(stddev):
    a_sq = stddev[:, :, 0, 0]
    a_b = stddev[:, :, 0, 1]
    b_sq_add_c_sq = stddev[:, :, 1, 1]
    eps = 1e-12

    a = torch.sqrt(a_sq + eps)
    b =  a_b / (a + eps)
    c = torch.sqrt(torch.clamp(b_sq_add_c_sq - b ** 2 + eps,eps))
    z = torch.zeros_like(a)
    row1 = torch.cat([c.unsqueeze(-1),z.unsqueeze(-1)],dim=-1).unsqueeze(-2)
    row2 = torch.cat([-b.unsqueeze(-1),a.unsqueeze(-1)],dim=-1).unsqueeze(-2)
    det = a*c

    L_inv_scale = 0.8
    L_inv = L_inv_scale/(det[:,:,None,None]+eps) * torch.cat([row1,row2],dim=-2)
    return L_inv

def gaussian2kp(heatmap, kp_variance='matrix', clip_variance=None):
    """
    Extract the mean and the variance from a heatmap
    """
    shape = heatmap.shape
    #adding small eps to avoid 'nan' in variance
    heatmap = heatmap.unsqueeze(-1) + 1e-12
    grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
    #import ipdb; ipdb.set_trace()

    mean = (heatmap * grid).sum(dim=2).sum(dim=2)

    #kp = {'mean': mean.permute(0, 2, 1)}
    kp = {'mean': mean}
    #if kp_variance == 'matrixv2':
    #    mean_sub = grid - mean.unsqueeze(-2).unsqueeze(-2)
    #    var = torch.matmul(mean_sub.unsqueeze(-1), mean_sub.unsqueeze(-2))
    #    var = var * heatmap.unsqueeze(-1)
    #    var = var.sum(dim=(2, 3))
    #    kp['var'] = var
    #    #gridOP = torch.einsum('ijm,ijn->ijmn',grid[0,0],grid[0,0])
    #    #meanOP= torch.einsum('ijm,ijn->ijmn',mean,mean)
    #    #stddev = torch.einsum('ijmn,akij->akmn', gridOP,heatmap.squeeze()) - meanOP
    #    #kp['var'] = stddev
    if kp_variance == 'matrix' or kp_variance == 'matrixv2':
        #import ipdb; ipdb.set_trace()
        mean_sub = grid - mean.unsqueeze(-2).unsqueeze(-2)

        var = torch.matmul(mean_sub.unsqueeze(-1), mean_sub.unsqueeze(-2))
        var = var * heatmap.unsqueeze(-1)
        var = var.sum(dim=(2, 3))
        #var = var.permute(0, 2, 1, 3, 4)
        #import ipdb; ipdb.set_trace()
        if clip_variance:
            min_norm = torch.tensor(clip_variance).type(var.type())
            sg = smallest_singular(var).unsqueeze(-1).detach()
            var = torch.max(min_norm, sg) * var / (sg+1e-8)
        kp['var'] = var
    else:
        mean_sub = grid - mean.unsqueeze(-2).unsqueeze(-2)
        var = mean_sub ** 2
        var = var * heatmap
        var = var.sum(dim=2).sum(dim=2)
        var = var.mean(dim=-1, keepdim=True)
        var = var.unsqueeze(-1)
        #var = var.permute(0, 2, 1)
        kp['var'] = var
    kp['heatmap'] = heatmap

    return kp

def kp2gaussian(kp, spatial_size, kp_variance='matrix', dist_type='gauss'):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp['mean']

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())

    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape

    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)
    #import ipdb; ipdb.set_trace()

    mean_sub = (coordinate_grid - mean)
    if kp_variance == 'matrixv2':
        var = kp['var']
        inv_var = matrix_inverse(var)
        shape = inv_var.shape[:number_of_leading_dimensions] + (1, 1, 2, 2)
        inv_var = inv_var.view(*shape)
        under_exp = torch.matmul(torch.matmul(mean_sub.unsqueeze(-2), inv_var), mean_sub.unsqueeze(-1))
        under_exp = under_exp.squeeze(-1).squeeze(-1)**3
    elif kp_variance == 'matrix':
        var = kp['var']
        inv_var = matrix_inverse(var)
        shape = inv_var.shape[:number_of_leading_dimensions] + (1, 1, 2, 2)
        inv_var = inv_var.view(*shape)
        under_exp = torch.matmul(torch.matmul(mean_sub.unsqueeze(-2), inv_var), mean_sub.unsqueeze(-1))
        under_exp = under_exp.squeeze(-1).squeeze(-1)
    elif kp_variance == 'single':
        under_exp = (mean_sub ** 2).sum(-1) / kp['var']
    elif kp_variance == 'flattop':
        under_exp = (mean_sub ** 6).sum(-1) / kp['var']
    else:
        under_exp = (mean_sub ** 2).sum(-1) / kp_variance

    if dist_type == 'gauss':
        out = torch.exp(-0.5 * under_exp)
        #if mode == 'train':
        out = out + 1e-2
    elif dist_type == 'gauss_const':
        out = torch.exp(-0.5 * under_exp) + 1e-2
    elif dist_type == 'oneoverx':
        out = 1./(1.+ under_exp)

    return out

class StyleMLP(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, inp_dim=512, out_dims=[512], n_layers=3, noInpLabel=1, nc=1, use_bias = True, fully_conv=False, no_mlp = 0,
                 inp_kp = 0, use_satt=False, input_style=0, use_mlss_remap=False, use_noise_for_missing=False, n_kp = 0, heidelberg_version=0,
                 spat_sizes=None, use_kp_for_style=False, downsampmax=False
                 ):
        super(StyleMLP, self).__init__()
        mlpLayers = []
        self.n_outs = len(out_dims)
        self.inp_kp = inp_kp
        self.use_mlss_remap = use_mlss_remap
        w_dim = inp_dim
        ksz = 1 #if not inp_kp else 3
        pad = max(ksz-2,0)
        self.n_layers = n_layers
        self.use_noise_for_missing = use_noise_for_missing
        self.heidelberg_version= heidelberg_version
        self.spat_sizes = spat_sizes
        self.input_style = (input_style > 0)
        self.input_style_dim = input_style
        self.use_kp_for_style= use_kp_for_style
        self.downsampmax = downsampmax
        if use_noise_for_missing:
            self.noise_params = nn.Parameter(torch.cat([torch.zeros(1,n_kp),0.001*torch.ones(1,n_kp)],dim=0),requires_grad=True)
        else:
            self.noise_params = None
        if self.n_layers > 0:
            for i in range(n_layers):
                if not fully_conv:
                    mlpLayers.append(nn.Linear(inp_dim if i==0 else w_dim, w_dim))
                else:
                    mlpLayers.append(nn.Conv2d(inp_dim if i==0 else w_dim, w_dim, kernel_size=ksz, stride=1, padding=pad))
                mlpLayers.append(nn.LeakyReLU(0.2,inplace=True))
            self.mlp = nn.Sequential(*mlpLayers)

        self.use_satt = use_satt
        if self.use_satt:
            self.satt = Self_Attn(w_dim)

        if not fully_conv:
            self.output_mapping = nn.ModuleList([nn.Linear(w_dim, 2*out_dims[i]) for i in range(self.n_outs)])
        else:
            sty_inp_layer = [nn.Conv2d(w_dim, self.input_style_dim, kernel_size=1, stride=1, padding=0)] if self.input_style else []
            self.output_mapping = nn.ModuleList(sty_inp_layer+[nn.Conv2d((w_dim if not use_kp_for_style else n_kp), 2*out_dims[i], kernel_size=1, stride=1, padding=0) for i in range(self.n_outs)])
            self.n_outs = len(self.output_mapping)

        # initialze the biases, this is custom for the module
        with torch.no_grad():
            for i in range(self.n_outs):
                self.output_mapping[i].bias.zero_()
                #self.output_mapping[i].bias[:out_dims[i]] = 1

    def forward(self, x, kpsrc=None, kptarg=None):
        #if self.inp_kp:
        #    kp = torch.cat([kpsrc, kptarg],dim=1)
        #    x = torch.cat([x,F.adaptive_max_pool2d(kp, (x.shape[2], x.shape[3]))],dim=1)
        w = x if self.n_layers == 0 else self.mlp(x)
        if self.inp_kp:
            if self.use_mlss_remap:
                w_remap = mls_rigid_deformation_inv(w, kpsrc, kptarg, weight='allvarAndStr')
            else:
                w_remap = remap_feats_with_keypoints(w, kpsrc, kptarg, kp_type=self.inp_kp, noise_params = self.noise_params, heidelberg_version = self.heidelberg_version)
        else:
            w_remap = w
        if self.use_satt:
            w_remap = self.satt(w_remap)
        #import ipdb; ipdb.set_trace()
        if self.spat_sizes is not None:
            adaIn_src = w_remap if not self.use_kp_for_style else kptarg['img']
            if self.downsampmax:
                outs = [self.output_mapping[i](F.adaptive_max_pool2d(adaIn_src,(self.spat_sizes[i-self.input_style],self.spat_sizes[i-self.input_style])) if i >=self.input_style else w_remap) for i in range(self.n_outs)]
            else:
                outs = [self.output_mapping[i](F.interpolate(adaIn_src,(self.spat_sizes[i-self.input_style],self.spat_sizes[i-self.input_style]),mode='nearest') if i >=self.input_style else w_remap) for i in range(self.n_outs)]
        else:
            adaIn_src = w_remap if not self.use_kp_for_style else kptarg['img']
            outs = [self.output_mapping[i](adaIn_src) for i in range(self.n_outs)]
        return outs

    def get_projected_feat(self, x, kpsrc=None):
        #if self.inp_kp:
        #    kp = torch.cat([kpsrc, kptarg],dim=1)
        #    x = torch.cat([x,F.adaptive_max_pool2d(kp, (x.shape[2], x.shape[3]))],dim=1)
        w = x if self.n_layers == 0 else self.mlp(x)
        if self.inp_kp:
            if self.use_mlss_remap:
                w_remap = mls_rigid_deformation_inv(w, kpsrc, kptarg, weight='allvarAndStr')
            else:
                w_remap = map_feats_to_keypoints(w, kpsrc, kp_type=self.inp_kp, noise_params = self.noise_params, heidelberg_version = self.heidelberg_version)
        else:
            w_remap = w
        return w_remap

    def map_feat_to_target(self, kpfeat, kptarg, shape):
        #if self.inp_kp:
        #    kp = torch.cat([kpsrc, kptarg],dim=1)
        #    x = torch.cat([x,F.adaptive_max_pool2d(kp, (x.shape[2], x.shape[3]))],dim=1)
        w = kpfeat
        if self.inp_kp:
            if self.use_mlss_remap:
                w_remap = mls_rigid_deformation_inv(w, kpsrc, kptarg, weight='allvarAndStr')
            else:
                w_remap = remap_kpfeat_to_target(w, kptarg, shape, kp_type=self.inp_kp, heidelberg_version = self.heidelberg_version)
        else:
            w_remap = w
        if self.use_satt:
            w_remap = self.satt(w_remap)
        #import ipdb; ipdb.set_trace()
        if self.spat_sizes is not None:
            adaIn_src = w_remap if not self.use_kp_for_style else kptarg['img']
            if self.downsampmax:
                outs = [self.output_mapping[i](F.adaptive_max_pool2d(adaIn_src,(self.spat_sizes[i-self.input_style],self.spat_sizes[i-self.input_style])) if i >=self.input_style else w_remap ) for i in range(self.n_outs)]
            else:
                outs = [self.output_mapping[i](F.interpolate(adaIn_src,(self.spat_sizes[i-self.input_style],self.spat_sizes[i-self.input_style]), mode='nearest') if i >=self.input_style else w_remap ) for i in range(self.n_outs)]
        else:
            adaIn_src = w_remap if not self.use_kp_for_style else kptarg['img']
            outs = [self.output_mapping[i](adaIn_src) for i in range(self.n_outs)]
        return outs





class Canvas_Style(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, n_conv=5, out_dims=[512], use_vgg=False):
        super(Canvas_Style, self).__init__()

        self.inp_resize=64
        ds_layers = []
        self.n_outs = len(out_dims)
        # Image is 64 x 64
        #ds_layers.extend(get_conv_relu_block(nc if noInpLabel else nc+c_dim, conv_dim, 7, 1, 3, padtype='zero', slope=0.2))
        ds_layers.extend(get_conv_inorm_relu_block(4, conv_dim, 3, 2, 1, padtype='replication', slope=0.01))
        #ds_layers.append(nn.AvgPool2d(2,stride=2))
        ds_layers.extend(get_conv_inorm_relu_block(conv_dim, conv_dim, 3, 2, 1, padtype='replication', slope=0.01))
        #ds_layers.append(nn.AvgPool2d(2,stride=2))
        ds_layers.extend(get_conv_inorm_relu_block(conv_dim, conv_dim, 3, 2, 1, padtype='replication', slope=0.01))

        self.ds_layers = nn.Sequential(*ds_layers)
        # Down-Sampling
        w_dim = conv_dim
        self.output_mapping = nn.ModuleList([nn.Linear(w_dim, 2*out_dims[i]) for i in range(self.n_outs)])
        #-------------------------------------------
        # After downsampling spatial dim is 4 x 4
        # Feat dim is 512


    def forward(self, x):
        # replicate spatially and concatenate domain information
        bsz = x.size(0)
        x = F.interpolate(x, size=self.inp_resize, mode='bilinear')

        ds_out = F.adaptive_avg_pool2d(self.ds_layers(x),1).view(bsz,-1)

        #curr_downscale = 8 if self.lowres_mask == 0 else 4
        outs = [self.output_mapping[i](ds_out) for i in range(self.n_outs)]
        return outs

class PatchEncoder(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, out_dim=512, noInpLabel=0, nc=3, use_bias = True, style_spatdim = 4,
                 use_deform=False, encode_keypoints=False, n_kp = 15, use_kpstrength=False, binary_kpstrength = False,
                 use_bnorm=False, use_inorm_downsamp=False, use_skip_connection=False, use_imnetbackbone=False):
        super(PatchEncoder, self).__init__()

        self.noInpLabel = noInpLabel
        self.use_skip_connection = use_skip_connection
        ds_layers = []
        # Image is 128 x 128
        #ds_layers.extend(get_conv_relu_block(nc if noInpLabel else nc+c_dim, conv_dim, 7, 1, 3, padtype='zero', slope=0.2))
        self.addDSblock(ds_layers, nc if noInpLabel else nc+c_dim, conv_dim, 7, 1, 3, 0.2, 'zero', use_inorm_downsamp, use_skip_connection)

        # Down-Sampling
        curr_dim = conv_dim
        #-------------------------------------------
        # After downsampling spatial dim is 4 x 4
        # Feat dim is 512
        #-------------------------------------------
        for i in range(3):
            #def get_conv_relu_block(i, o, k, s, p, slope=0.1, padtype='zero', dilation=1):
            #ds_layers.extend(get_conv_relu_block(curr_dim, min(curr_dim*2,out_dim), 3, 2, 1, slope=0.2, padtype='zero'))
            self.addDSblock(ds_layers, curr_dim, min(curr_dim*2,out_dim), 3, 2, 1, 0.2,
                            'zero', use_inorm_downsamp, use_skip_connection)

            curr_dim =  min(curr_dim*2,out_dim)
            # 64, 32, 16#, 8, 4
        ds_dim  = curr_dim

        self.style_spatdim  = style_spatdim
        styleEncBranch = [] if not use_deform else [DeformableLayer(curr_dim, curr_dim)]
        for i in range(2):
            #def get_conv_relu_block(i, o, k, s, p, slope=0.1, padtype='zero', dilation=1):
            #ds_layers.extend(get_conv_relu_block(curr_dim, min(curr_dim*2,out_dim), 3, 2, 1, slope=0.2, padtype='zero'))
            s = 2 if (16//(2**i)) > style_spatdim else 1
            styleEncBranch.extend(get_conv_inorm_relu_block(curr_dim, min(curr_dim*2,out_dim), 3, s, 1, slope=0.2, padtype='zero'))
            curr_dim =  min(curr_dim*2,out_dim)
            # 64, 32, 16, 8, 4

        self.pool = (self.style_spatdim < 4)
        if self.pool:
            styleEncBranch.append(nn.AdaptiveMaxPool2d(1))
        self.downsamp = nn.Sequential(*ds_layers) if not use_skip_connection else nn.ModuleList(ds_layers)
        self.styleEncBranch = nn.Sequential(*styleEncBranch)

        self.encode_keypoints = encode_keypoints
        # encode_keypoints = 1 --> predict keypoints
        # encode_keypoints = 2 --> predict part segmentation
        if self.encode_keypoints:
            keypointBranch = []
            for i in range(3):
                if use_bnorm:
                    keypointBranch.append(ResidualBlockBnorm(ds_dim, padtype='zero'))
                else:
                    keypointBranch.append(ResidualBlock(ds_dim, padtype='zero'))
            keypointBranch.append(nn.Conv2d(ds_dim, n_kp, kernel_size=1, stride=1, padding=0))
            self.n_kp = n_kp
            self.keypointBranch = nn.Sequential(*keypointBranch)
            self.use_kpstrength = use_kpstrength
            self.binary_kpstrength = binary_kpstrength
            if use_kpstrength:
                self.kpStr_w = nn.Parameter(0.01*torch.randn(self.n_kp),requires_grad=True)
                self.kpStr_b = nn.Parameter(torch.zeros(self.n_kp),requires_grad=True)

    def addDSblock(self, ds_layers, inp_dim, out_dim, k, s, pad, slope, padtype, use_inorm_downsamp, use_skip_connection):
            if use_inorm_downsamp:
                ds = get_conv_inorm_relu_block(inp_dim, out_dim, k, s, pad, slope=slope, padtype=padtype)
            else:
                ds = get_conv_bnorm_relu_block(inp_dim, out_dim, k, s, pad, slope=slope, padtype=padtype)
            if use_skip_connection:
                ds_layers.append(nn.Sequential(*ds))
            else:
                ds_layers.extend(ds)

    def get_kp(self, feat):
        bsz = feat.size(0)
        kp_convout = self.keypointBranch(feat)
        if self.encode_keypoints==1:
            heatmap = F.softmax(kp_convout.view(bsz, self.n_kp, -1),dim=-1).view(bsz, self.n_kp, feat.size(2), feat.size(3))
        elif self.encode_keypoints==2:
            heatmap = F.softmax(kp_convout,dim=1)
            heatmap = (heatmap/(1e-20+heatmap.sum(dim=(2,3)).unsqueeze(-1).unsqueeze(-1)))
        #    heatmap = kp_convout
        #    kpStrength = None
        kpStrength = torch.sigmoid(self.kpStr_w * F.adaptive_max_pool2d(kp_convout,1).view(bsz,self.n_kp)  + self.kpStr_b) if self.use_kpstrength else None
        if self.binary_kpstrength and self.use_kpstrength:
            kpStrength = ((kpStrength>0.5).float()- kpStrength).detach() + kpStrength
        return heatmap, kpStrength

    def forward(self, x, c=None, get_kp=False):
        # replicate spatially and concatenate domain information
        #import ipdb; ipdb.set_trace()
        bsz = x.size(0)
        if self.noInpLabel:
            xcat = x
        else:
            c = c.unsqueeze(2).unsqueeze(3)
            c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
            xcat = torch.cat([x, c], dim=1)

        ds_out = self.downsamp(xcat)
        feat_out = self.styleEncBranch(ds_out).view(bsz,-1) if self.pool else self.styleEncBranch(ds_out)

        heatmap = None
        kpStrength = None
        if self.encode_keypoints and get_kp:
            heatmap, kpStrength = self.get_kp(ds_out)

        #curr_downscale = 8 if self.lowres_mask == 0 else 4
        return feat_out, heatmap, kpStrength

    def forward_kp(self, x, c=None):
        # replicate spatially and concatenate domain information
        if self.noInpLabel:
            xcat = x
        else:
            c = c.unsqueeze(2).unsqueeze(3)
            c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
            xcat = torch.cat([x, c], dim=1)

        ds_out = self.downsamp(xcat)
        heatmap, kpStrength = self.get_kp(ds_out)
        return heatmap, kpStrength


class PatchEncoder_V2(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, out_dim=512, noInpLabel=0, nc=3, use_bias = True, style_spatdim = 4,
                 use_deform=False, encode_keypoints=False, n_kp = 15, use_kpstrength=False, binary_kpstrength = False,
                 use_bnorm=False, use_inorm_downsamp=False, use_skip_connection=False, use_antialias=False,
                 use_imnetbackbone=False, normalize = True):
        super(PatchEncoder_V2, self).__init__()

        self.noInpLabel = noInpLabel
        self.use_skip_connection = use_skip_connection
        self.normalize_heatmap = normalize
        ds_layers = []
        # Image is 128 x 128
        #ds_layers.extend(get_conv_relu_block(nc if noInpLabel else nc+c_dim, conv_dim, 7, 1, 3, padtype='zero', slope=0.2))
        self.addDSblock(ds_layers, nc if noInpLabel else nc+c_dim, conv_dim, 7, 1, 3, 0.2, 'zero', use_inorm_downsamp, use_skip_connection, use_antialias)

        # Down-Sampling
        curr_dim = conv_dim
        self.use_imnetbackbone = use_imnetbackbone
        self.pnet = Vgg19() if use_imnetbackbone else None
        #-------------------------------------------
        # After downsampling spatial dim is 16 x 16
        # Feat dim is 512
        #-------------------------------------------
        if not use_imnetbackbone:
            ds_dims =[conv_dim]
            for i in range(5):
                #def get_conv_relu_block(i, o, k, s, p, slope=0.1, padtype='zero', dilation=1):
                #ds_layers.extend(get_conv_relu_block(curr_dim, min(curr_dim*2,out_dim), 3, 2, 1, slope=0.2, padtype='zero'))
                self.addDSblock(ds_layers, curr_dim, min(curr_dim*2,out_dim), 3, 2, 1, 0.2,
                                'zero', use_inorm_downsamp, use_skip_connection, use_antialias)

                curr_dim =  min(curr_dim*2,out_dim)
                # 64, 32, 16, 8, 4#
                ds_dims.append(curr_dim)
            self.downsamp = nn.Sequential(*ds_layers) if not use_skip_connection else nn.ModuleList(ds_layers)
        else:
            self.downsamp = None
            ds_dims = [v+c_dim for v in [64, 128, 256, 512, 512]]
            self.shift = torch.autograd.Variable(torch.Tensor([-.030, -.088, -.188]).view(1,3,1,1), requires_grad=False).cuda()
            self.scale = torch.autograd.Variable(torch.Tensor([.458, .448, .450]).view(1,3,1,1), requires_grad=False).cuda()

        self.style_spatdim  = style_spatdim
        styleEncBranch = nn.ModuleList()
        styleEncBranch.append(nn.Sequential(*get_conv_inorm_relu_block(n_kp+ds_dims[1], out_dim, 3, 1, 1, slope=0.2, padtype='zero')))
        # For downsampling
        styleEncBranch.append(nn.Sequential(*get_conv_inorm_relu_block(out_dim, out_dim, 3, 2, 1, slope=0.2, padtype='zero')))
        for i in range(2):
            #def get_conv_relu_block(i, o, k, s, p, slope=0.1, padtype='zero', dilation=1):
            #ds_layers.extend(get_conv_relu_block(curr_dim, min(curr_dim*2,out_dim), 3, 2, 1, slope=0.2, padtype='zero'))
            s = 1
            styleEncBranch.append(nn.Sequential(*get_conv_inorm_relu_block(out_dim, out_dim, 3, s, 1, slope=0.2, padtype='zero')))
            # 64, 32, 16, 8, 4
        styleEncBranch.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        styleEncBranch.append(nn.Sequential(*get_conv_inorm_relu_block(out_dim*2, out_dim, 3, 1, 1, slope=0.2, padtype='zero')))


        self.styleEncBranch = styleEncBranch
        self.encode_keypoints = encode_keypoints
        # encode_keypoints = 1 --> predict keypoints
        # encode_keypoints = 2 --> predict part segmentation
        if self.encode_keypoints:
            keypointBranch = []
            n_upsamp = 4 if not use_imnetbackbone else 3
            for i in range(n_upsamp):
                if use_imnetbackbone:
                    kpUpsamp = [ResidualProjectBlock(ds_dims[-1-i]*(1+(i>0)), [ds_dims[-2-i]//2, ds_dims[-2-i]//2, ds_dims[-2-i]], down_samp_factor = 1)]
                else:
                    if use_bnorm:
                        kpUpsamp =  get_conv_bnorm_relu_block(ds_dims[-1-i]*(1+(i>0)), ds_dims[-2-i], 3, 1, 1, slope=0.01, padtype='zero')
                    else:
                        kpUpsamp =  get_conv_inorm_relu_block(ds_dims[-1-i]*(1+(i>0)), ds_dims[-2-i], 3, 1, 1, slope=0.01, padtype='zero')
                keypointBranch.append(nn.Sequential(*(kpUpsamp+[nn.Upsample(scale_factor=2,mode='bilinear')])))
            #print(ds_dims, ds_dims[-1-i], n_upsamp)
            keypointBranch.append(nn.Conv2d(ds_dims[-1-i-use_imnetbackbone], n_kp, kernel_size=1, stride=1, padding=0))
            self.n_kp = n_kp
            self.keypointBranch = nn.ModuleList(keypointBranch)
            self.use_kpstrength = use_kpstrength
            self.binary_kpstrength = binary_kpstrength
            if use_kpstrength:
                self.kpStr_w = nn.Parameter(0.01*torch.randn(self.n_kp),requires_grad=True)
                self.kpStr_b = nn.Parameter(torch.zeros(self.n_kp),requires_grad=True)

    def addDSblock(self, ds_layers, inp_dim, out_dim, k, s, pad, slope, padtype, use_inorm_downsamp, use_skip_connection, use_antialias):
            if use_inorm_downsamp:
                ds = get_conv_inorm_relu_block(inp_dim, out_dim, k, s, pad, slope=slope, padtype=padtype)
            else:
                ds = get_conv_bnorm_relu_block(inp_dim, out_dim, k, s, pad, slope=slope, padtype=padtype)
            if use_skip_connection:
                ds_layers.append(nn.Sequential(*ds))
            else:
                ds_layers.extend(ds)

    #@profile
    def get_kp(self, feat):
        bsz = feat[-1].size(0)
        kp_convout = [feat[-1]]
        for li in range(len(self.keypointBranch)):
            kp_convout.append(self.keypointBranch[li](kp_convout[li]))
            kp_convout[-1] = torch.cat([kp_convout[-1], feat[-2-li]],dim=1) if li < (len(self.keypointBranch) -1 - self.use_imnetbackbone) else kp_convout[-1]

        if self.encode_keypoints==1:
            #import ipdb; ipdb.set_trace()
            heatmap = F.softmax(kp_convout[-1].view(bsz, self.n_kp, -1),dim=-1).view(bsz, self.n_kp, kp_convout[-1].size(2), kp_convout[-1].size(3))
        elif self.encode_keypoints==2:
            heatmap = F.softmax(kp_convout[-1],dim=1)
            if self.normalize_heatmap:
                heatmap = (heatmap/(1e-20+heatmap.sum(dim=(2,3)).unsqueeze(-1).unsqueeze(-1)))
        #    #heatmap = F.normalize(kp_convout[-1].view(bsz, self.n_kp, -1),p=1,dim=-1).view(bsz, self.n_kp, kp_convout[-1].size(2), kp_convout[-1].size(3))
        #elif self.encode_keypoints==2:
        #    heatmap = kp_convout[-1]
        kpStrength = torch.sigmoid(self.kpStr_w * F.adaptive_max_pool2d(kp_convout[-1],1).view(bsz,self.n_kp)  + self.kpStr_b) if self.use_kpstrength else None
        if self.binary_kpstrength and self.use_kpstrength:
            kpStrength = ((kpStrength>0.5).float()- kpStrength).detach() + kpStrength
        return heatmap, kpStrength

    #@profile
    def forward(self, x, c=None, get_kp=False):
        # replicate spatially and concatenate domain information
        #import ipdb; ipdb.set_trace()
        heatmap, kpStrength, ds_out = self.forward_kp(x, c, get_ds_out=True)

        # Decode style features using the generated heatmap
        style_inp = torch.cat([heatmap,ds_out[2]],dim=1)
        styf1 = self.styleEncBranch[0](style_inp)
        stydsf = self.styleEncBranch[3](self.styleEncBranch[2](self.styleEncBranch[1](styf1)))
        feat_out = self.styleEncBranch[-1](torch.cat([styf1,self.styleEncBranch[4](stydsf)],dim=1))

        #curr_downscale = 8 if self.lowres_mask == 0 else 4
        return feat_out, heatmap, kpStrength

    #@profile
    def forward_kp(self, x, c=None, get_ds_out=False):
        # replicate spatially and concatenate domain information
        bsz = x.size(0)
        if self.noInpLabel or self.use_imnetbackbone:
            xcat = x
        else:
            c = c.unsqueeze(2).unsqueeze(3)
            c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
            xcat = torch.cat([x, c], dim=1)

        ds_out = [xcat]
        if self.use_imnetbackbone:
            c = c.unsqueeze(2).unsqueeze(3)
            x = (x - self.shift.expand_as(x))/self.scale.expand_as(x)
            ds_out.extend([torch.cat([f,c.expand(bsz, c.size(1), f.size(2),f.size(3))],dim=1) for f in self.pnet(x)])
        else:
            for li in range(len(self.downsamp)):
                ds_out.append(self.downsamp[li](ds_out[li]))

        heatmap, kpStrength = self.get_kp(ds_out)

        if get_ds_out:
            return heatmap, kpStrength, ds_out
        else:
            return heatmap, kpStrength


class GeneratorPatchTransform(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, resLayers=4, up_sampling_type='bilinear',
                 n_upsamp_filt=2, additional_cond='none', noInpLabel=0, nc=1, use_bias = True,
                 use_const_tensor = 0, apply_noise=False, kp_input=0, no_residual=0, u_net = 0,
                 pred_targ_mask = 0, seperate_mask_path=0, input_style = 0):
        super(GeneratorPatchTransform, self).__init__()

        self.lowres_mask = int(0)
        self.additional_cond = additional_cond
        self.noInpLabel = noInpLabel
        self.input_style = input_style
        if use_const_tensor:
            self.const_tensor_dim=use_const_tensor
            self.const_tensor = nn.Parameter(torch.randn(1,self.const_tensor_dim, 128,128),requires_grad=True)
        else:
            self.const_tensor_dim = 0

        ds_layers = []
        # Image is 128 x 128
        nc = nc + kp_input
        nc = nc + self.const_tensor_dim if noInpLabel else nc+c_dim+self.const_tensor_dim
        ds_layers.extend(get_conv_inorm_relu_block(nc, conv_dim, 7, 1, 3, padtype='zero'))

        # Down-Sampling
        curr_dim = conv_dim
        extra_dim = 3 if self.additional_cond == 'image' else c_dim if self.additional_cond == 'label'else 0
        self.unet=u_net
        #-------------------------------------------
        # After downsampling spatial dim is 16 x 16
        # Feat dim is 512
        #-------------------------------------------
        for i in range(3 - self.lowres_mask):
            #def get_conv_inorm_relu_block(i, o, k, s, p, slope=0.1, padtype='zero', dilation=1):
            ds_layers.extend(get_conv_inorm_relu_block(curr_dim, curr_dim*2, 3, 2, 1, padtype='zero'))
            curr_dim = curr_dim * 2

        dilation=1
        ds_out_dim = curr_dim
        #-------------------------------------------
        # After residual spatial dim is 16 x 16
        # Feat dim is 512
        #-------------------------------------------
        self.res_layers = nn.ModuleList()
        self.n_reslayers = resLayers
        curr_dim = curr_dim
        for i in range(resLayers):
            self.res_layers.append(ResidualBlockNoNorm(dim_in=curr_dim, dilation=dilation, padtype='zero', apply_noise = apply_noise, use_bias=True, no_residual = no_residual))
            if i > 1:
                # This gives dilation as 1, 1, 2, 4, 8, 16
                dilation=dilation*2

        # Up-Sampling Differential layers
        self.up_sampling_convlayers = nn.ModuleList()
        if self.lowres_mask == 0:
            self.up_sampling_ReLU= nn.ModuleList()
        self.apply_noise = apply_noise
        if self.apply_noise:
            self.noiseModules = nn.ModuleList()

        for i in range(3-self.lowres_mask):
            inp_dim = curr_dim  + (0 if not self.unet else curr_dim)
            if self.lowres_mask == 0:
                if up_sampling_type== 't_conv':
                    self.up_sampling_convlayers.append(nn.ConvTranspose2d(inp_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=use_bias))
                elif up_sampling_type == 'nearest':
                    self.up_sampling_convlayers.append(nn.Upsample(scale_factor=2, mode='nearest'))
                    self.up_sampling_convlayers.append(nn.Conv2d(inp_dim, curr_dim//2, kernel_size=3, stride=1, padding=1, bias=use_bias))
                elif up_sampling_type == 'deform':
                    self.up_sampling_convlayers.append(AdaptiveScaleTconv(inp_dim, curr_dim//2, scale=2, n_filters=n_upsamp_filt))
                elif up_sampling_type == 'bilinear':
                    self.up_sampling_convlayers.append(AdaptiveScaleTconv(inp_dim, curr_dim//2, scale=2, use_deform=False, n_filters=n_upsamp_filt))
                self.up_sampling_ReLU.append(nn.LeakyReLU(0.2,inplace=True))

                if self.apply_noise:
                    self.noiseModules.append(NoiseInjection(curr_dim//2))

            else:
                # In this case just use more residual blocks to drop dimensions
                self.up_sampling_convlayers.append(nn.Sequential(*get_conv_inorm_relu_block(curr_dim+extra_dim, curr_dim//2, 3, 1, 1, padtype='zero')))
            curr_dim = curr_dim // 2

        pad=0
        self.pred_targ_mask = pred_targ_mask
        self.seperate_mask_path = seperate_mask_path
        mask_channel = pred_targ_mask if not seperate_mask_path else 0
        self.finalConv = nn.Conv2d(curr_dim, 3+ pred_targ_mask, kernel_size=1, stride=1, padding=pad)
        if self.pred_targ_mask and self.seperate_mask_path:
            inp_dim = ds_out_dim
            self.mask_decode_layers = nn.Sequential(*[nn.Conv2d(inp_dim, 20, kernel_size=3, stride=1, padding=1, bias=use_bias),
                                                     nn.LeakyReLU(0.2,inplace=True),
                                                     nn.Conv2d(20, 10, kernel_size=7, stride=1, padding=3, bias=use_bias),
                                                     nn.LeakyReLU(0.2,inplace=True),
                                                     nn.Upsample(scale_factor=4,mode='nearest'),
                                                     nn.Conv2d(10, 5, kernel_size=7, stride=1, padding=3, bias=use_bias),
                                                     nn.LeakyReLU(0.2,inplace=True),
                                                     nn.Upsample(scale_factor=2,mode='nearest'),
                                                     nn.Conv2d(5, 1, kernel_size=7, stride=1, padding=3, bias=use_bias),
                                                    ])

        self.downsamp = nn.ModuleList(ds_layers) if self.unet else nn.Sequential(*ds_layers)

    def prepInp(self, feat, x, c, curr_scale):
        if self.additional_cond == 'image':
            up_inp = torch.cat([feat,nn.functional.avg_pool2d(x,curr_scale)], dim=1)
        elif self.additional_cond == 'label':
            up_inp = torch.cat([feat,nn.functional.avg_pool2d(c,curr_scale)], dim=1)
        else:
            up_inp = feat
        return up_inp

    def forward(self, x, styles, c=None):

        ds_out = self.forward_inpEnc(x, c)
        #curr_downscale = 8 if self.lowres_mask == 0 else 4

        img_out, mask_out = self.forward_featToImg(ds_out, styles, c)

        return img_out, mask_out

    def forward_featToImg(self, feat, styles, c=None):
        # replicate spatially and concatenate domain information
        #curr_downscale = 8 if self.lowres_mask == 0 else 4
        res_out = [feat[-1]] if self.unet else [feat]
        if self.input_style:
            res_out[0] = res_out[0] + styles[0]
        sty_offset = (self.input_style > 0)

        for i in range(self.n_reslayers):
            conv_out = self.res_layers[i](res_out[i])
            #adaIn_out = adaptive_instance_norm_2d(conv_out, styles[i])
            #res_out.append(F.leaky_relu(adaIn_out))
            adaIn_out = adaptive_instance_norm_2d(F.leaky_relu(conv_out,0.2), styles[i+sty_offset])
            res_out.append(adaIn_out)

        #up_inp = [self.prepInp(bottle_out, x, c, curr_downscale)]
        #curr_downscale = curr_downscale//2 if self.lowres_mask == 0 else curr_downscale
        up_out = [res_out[-1]]
        for i in range(len(self.up_sampling_convlayers)):
            #self.up_sampling_convlayers(
            conv_inp = up_out[i] if not self.unet else torch.cat([up_out[i], feat[-1-i]], dim=1)
            if type(self.up_sampling_convlayers[i]) == AdaptiveScaleTconv:
                upsampout,_ = self.up_sampling_convlayers[i](conv_inp)
            else:
                upsampout = self.up_sampling_convlayers[i](conv_inp)
            #adaIn_out = adaptive_instance_norm_2d(upsampout, styles[self.n_reslayers+i])
            #up_out.append(self.up_sampling_ReLU[i](adaIn_out) if self.lowres_mask == 0 else upsampout)
            #curr_downscale = curr_downscale//2 if self.lowres_mask == 0 else curr_downscale
            if self.apply_noise:
                upsampout = self.noiseModules[i](upsampout)

            adaIn_out = adaptive_instance_norm_2d(self.up_sampling_ReLU[i](upsampout), styles[sty_offset+self.n_reslayers+i])
            up_out.append(adaIn_out)

        final_conv_out = self.finalConv(up_out[-1])
        img_out = torch.tanh(final_conv_out[:,:3,::])
        if self.pred_targ_mask:
            if not self.seperate_mask_path:
                mask_out = torch.sigmoid(final_conv_out[:,3:,::])
            else:
                mask_out = torch.sigmoid(self.mask_decode_layers(feat[-1]))
        else:
            mask_out = None
        #img_out = self.finalConv(up_out[-1])

        return img_out, mask_out

    def forward_inpEnc(self, x, c=None):
        # replicate spatially and concatenate domain information
        bsz = x.size(0)
        if self.noInpLabel:
            xcat = x
        else:
            c = c.unsqueeze(2).unsqueeze(3)
            c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
            xcat = torch.cat([x, c], dim=1)

        if self.const_tensor_dim:
            xcat = torch.cat([xcat, self.const_tensor.expand(bsz,self.const_tensor_dim, x.size(2),x.size(3))], dim=1)

        if self.unet:
            ds_out = [xcat]
            for i in range(len(self.downsamp)//3):
                ds_out.append(self.downsamp[3*i+2](self.downsamp[3*i+1](self.downsamp[3*i](ds_out[i]))))
        else:
            ds_out = self.downsamp(xcat)
        #curr_downscale = 8 if self.lowres_mask == 0 else 4

        return ds_out

class GeneratorPatchTransform_V2(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, noInpLabel=0, nc=1, use_bias = True,
                 use_const_tensor = 0, apply_noise=False, kp_input=0, no_residual=0,
                 pred_targ_mask = 0, res_layer_config = [512, 512, 512, 256, 128, 64], n_upsamp = 5,
                 input_style = 0, start_res=4, u_net=1, noAdaIn=False, downsampmax=False):
        super(GeneratorPatchTransform_V2, self).__init__()

        # Copy needed params
        self.noInpLabel = noInpLabel
        self.start_res = start_res
        self.input_style = input_style
        self.n_upsamp = n_upsamp
        self.unet= u_net
        self.downsampmax = downsampmax
        self.noAdaIn = noAdaIn
        self.no_residual = no_residual
        if use_const_tensor:
            self.const_tensor_dim=use_const_tensor
            self.const_tensor = nn.Parameter(torch.randn(1,self.const_tensor_dim, self.start_res, self.start_res),requires_grad=True)
        else:
            self.const_tensor_dim = 0
        # Image is 128 x 128
        nc = nc + kp_input
        nc = nc + self.const_tensor_dim if noInpLabel else nc+c_dim+self.const_tensor_dim
        self.n_reslayers = len(res_layer_config)


        self.inp_conv = snconv2d(nc, conv_dim, kernel_size=1, stride=1, use_sn=False)
        #-------------------------------------------
        # After residual spatial dim is 16 x 16
        # Feat dim is 512
        #-------------------------------------------
        conv_dim = conv_dim*(1+self.input_style) if self.input_style <=1 else (conv_dim + self.input_style)
        self.res_layers = nn.ModuleList()
        out_dims = [conv_dim] + res_layer_config#, 64, 64,  32,  32]
        dilation=1
        #resolut = [  4,   8,  16,  32, 64, 128]#, 64, 64, 128, 128]
        for i in range(self.n_reslayers):
            self.res_layers.append(ResidualBlockNoNorm(dim_in=out_dims[i]+(i>0)*conv_dim*self.unet, out_dim=out_dims[i+1], dilation=dilation, padtype='zero',
                                                       apply_noise = apply_noise, use_bias=True, no_residual = no_residual, use_sn=False))

        # Up-Sampling Differential layers
        pad=1
        self.pred_targ_mask = pred_targ_mask
        self.finalConv = snconv2d(out_dims[-1], 3+ pred_targ_mask, kernel_size=3, stride=1, padding=pad, use_sn=False)

    #@profile
    def forward(self, x, styles, c=None):

        ds_out = self.forward_inpEnc(x, c)
        #curr_downscale = 8 if self.lowres_mask == 0 else 4

        img_out, mask_out = self.forward_featToImg(ds_out, styles, c)

        return img_out, mask_out

    #@profile
    def forward_featToImg(self, feat, styles, c=None):
        # replicate spatially and concatenate domain information
        #curr_downscale = 8 if self.lowres_mask == 0 else 4
        sty_offset = (self.input_style > 0)
        if self.input_style:
            if self.downsampmax:
                feat = [torch.cat([f, F.adaptive_max_pool2d(styles[0], (f.size(2), f.size(3)))],dim=1) for f in feat]
            else:
                feat = [torch.cat([f, F.interpolate(styles[0], (f.size(2), f.size(3)), mode='nearest')],dim=1) for f in feat]
        res_out = feat[:1] #[feat[0] + styles[0]] if self.input_style else feat[:1]
        unet_feat = feat
        for i in range(self.n_reslayers):
            #------------------------------------------------------------------
            # Config 1: upsample --> conv --> relu --> instance norm
            #------------------------------------------------------------------
            #up_out = F.interpolate(res_out[i], scale_factor=2, mode='bilinear') if i >= (self.n_reslayers - self.n_upsamp) else res_out[i]
            #conv_out = self.res_layers[i](up_out)
            #if i < len(styles):
            #    adaIn_out = adaptive_instance_norm_2d(F.leaky_relu(conv_out,0.2), styles[i])
            #else:
            #    adaIn_out = F.instance_norm(F.leaky_relu(conv_out,0.2))
            #res_out.append(adaIn_out)
            #------------------------------------------------------------------
            # Config 2: upsample --> conv --> relu --> instance norm
            #------------------------------------------------------------------
            if i < len(styles) and (not self.noAdaIn):
                adaIn_out = F.leaky_relu(adaptive_instance_norm_2d(res_out[i], styles[i+sty_offset]),0.2 if self.no_residual else 0.001)
            else:
                adaIn_out = F.leaky_relu(F.instance_norm(res_out[i]), 0.2 if self.no_residual else 0.001)
            up_out = F.interpolate(adaIn_out, scale_factor=2, mode='bilinear') if i >= (self.n_reslayers - self.n_upsamp) else adaIn_out
            if self.unet and i >0:
                up_out = torch.cat([up_out, unet_feat[min(max(0,i-(self.n_reslayers - self.n_upsamp-1)),self.n_upsamp)]],dim=1)
            conv_out = self.res_layers[i](up_out)
            res_out.append(conv_out)

        #up_inp = [self.prepInp(bottle_out, x, c, curr_downscale)]
        #curr_downscale = curr_downscale//2 if self.lowres_mask == 0 else curr_downscale

        #import ipdb; ipdb.set_trace()
        final_conv_out = self.finalConv(res_out[-1])
        img_out = torch.tanh(final_conv_out[:,:3,::])
        if self.pred_targ_mask:
            mask_out = torch.sigmoid(final_conv_out[:,3:,::])
        else:
            mask_out = None
        #img_out = self.finalConv(up_out[-1])

        return img_out, mask_out

    #@profile
    def forward_inpEnc(self, x, c=None, styles=None):
        # replicate spatially and concatenate domain information
        bsz = x.size(0)
        if self.noInpLabel:
            xcat = x
        else:
            c = c.unsqueeze(2).unsqueeze(3)
            c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
            xcat = torch.cat([x, c], dim=1)
        if self.const_tensor_dim:
            xcat = torch.cat([xcat, self.const_tensor.expand(bsz,self.const_tensor_dim, x.size(2),x.size(3))], dim=1)

        conv_out = F.leaky_relu(self.inp_conv(xcat),0.2)
        if self.unet:
            if self.downsampmax:
                ds_out = [F.adaptive_max_pool2d(conv_out, (self.start_res*(2**i), self.start_res*(2**i))) for i in range(self.n_upsamp+1)]
            else:
                ds_out = [F.interpolate(conv_out, (self.start_res*(2**i), self.start_res*(2**i)), mode='nearest') for i in range(self.n_upsamp+1)]
        else:
            ds_out = [F.interpolate(conv_out, (self.start_res, self.start_res), mode='nearest')]

        #curr_downscale = 8 if self.lowres_mask == 0 else 4
        return ds_out

class Generator(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6, g_smooth_layers=0, binary_mask=0):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim))

        # Up-Sampling
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # replicate spatially and concatenate domain information
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)

class GeneratorDiffAndMask(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=3, g_smooth_layers=0, binary_mask=0):
        super(GeneratorDiffAndMask, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim))

        # Up-Sampling Differential layers
        self.up_sampling_convlayers = nn.ModuleList()
        self.up_sampling_inorm= nn.ModuleList()
        self.up_sampling_ReLU= nn.ModuleList()

        # Up-Sampling Mask layers
        self.up_sampling_convlayers_mask = nn.ModuleList()
        self.up_sampling_inorm_mask= nn.ModuleList()
        self.up_sampling_ReLU_mask = nn.ModuleList()
        for i in range(2):
            self.up_sampling_convlayers.append(nn.ConvTranspose2d(curr_dim+3, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            self.up_sampling_inorm.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            self.up_sampling_ReLU.append(nn.ReLU(inplace=False))

            ## Add the mask path
            self.up_sampling_convlayers_mask.append(nn.ConvTranspose2d(curr_dim+3, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            self.up_sampling_inorm_mask.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            self.up_sampling_ReLU_mask.append(nn.ReLU(inplace=False))
            curr_dim = curr_dim // 2

        self.final_Layer = nn.Conv2d(curr_dim+3, 3, kernel_size=7, stride=1, padding=3, bias=False)
        #layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        # Remove this non-linearity or use 2.0*tanh ?
        self.finalNonLin = nn.Tanh()

        self.final_Layer_mask = nn.Conv2d(curr_dim+3, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.finalNonLin_mask = nn.Sigmoid()


        self.hardtanh = nn.Hardtanh(min_val=-1, max_val=1)
        self.main = nn.Sequential(*layers)

    def forward(self, x, c, out_diff = False):
        # replicate spatially and concatenate domain information
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        xcat = torch.cat([x, c], dim=1)
        bottle_out = self.main(xcat)
        curr_downscale = 4
        up_inp = [torch.cat([bottle_out,nn.functional.avg_pool2d(x,curr_downscale)], dim=1)]
        up_inp_mask = [None]
        up_inp_mask[0] = up_inp[0]
        curr_downscale = curr_downscale//2
        up_out = []
        up_out_mask = []
        for i in range(len(self.up_sampling_convlayers)):
            #self.up_sampling_convlayers(x
            up_out.append(self.up_sampling_ReLU[i](self.up_sampling_inorm[i](self.up_sampling_convlayers[i](up_inp[i]))))
            up_inp.append(torch.cat([up_out[i],nn.functional.avg_pool2d(x,curr_downscale)], dim=1))

            # Compute the maks output
            up_out_mask.append(self.up_sampling_ReLU_mask[i](self.up_sampling_inorm_mask[i](self.up_sampling_convlayers_mask[i](up_inp_mask[i]))))
            up_inp_mask.append(torch.cat([up_out_mask[i],nn.functional.avg_pool2d(x,curr_downscale)], dim=1))
            curr_downscale = curr_downscale//2

        net_out = self.finalNonLin(self.final_Layer(up_inp[-1]))
        mask = self.finalNonLin_mask(self.final_Layer_mask(up_inp_mask[-1]))
        if out_diff:
            return ((1-mask)*x+mask*net_out), (net_out, mask)
        else:
            return ((1-mask)*x+mask*net_out)

class GeneratorDiffAndMask_V2(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=3, g_smooth_layers=0, binary_mask=0):
        super(GeneratorDiffAndMask_V2, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim))

        # Up-Sampling Differential layers
        self.up_sampling_convlayers = nn.ModuleList()
        self.up_sampling_inorm= nn.ModuleList()
        self.up_sampling_ReLU= nn.ModuleList()

        for i in range(2):
            self.up_sampling_convlayers.append(nn.ConvTranspose2d(curr_dim+3, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            self.up_sampling_inorm.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            self.up_sampling_ReLU.append(nn.ReLU(inplace=False))

            curr_dim = curr_dim // 2

        self.final_Layer = nn.Conv2d(curr_dim+3, 3, kernel_size=7, stride=1, padding=3, bias=False)
        #layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        # Remove this non-linearity or use 2.0*tanh ?
        self.finalNonLin = nn.Tanh()

        self.final_Layer_mask = nn.Conv2d(curr_dim+3, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.finalNonLin_mask = nn.Sigmoid()

        self.g_smooth_layers = g_smooth_layers
        if g_smooth_layers > 0:
            smooth_layers = []
            for i in range(g_smooth_layers):
                smooth_layers.append(nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False))
                smooth_layers.append(nn.Tanh())
            self.smooth_layers= nn.Sequential(*smooth_layers)

        self.hardtanh = nn.Hardtanh(min_val=-1, max_val=1)
        self.binary_mask = binary_mask
        self.main = nn.Sequential(*layers)

    def forward(self, x, c, out_diff = False):
        # replicate spatially and concatenate domain information
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        xcat = torch.cat([x, c], dim=1)
        bottle_out = self.main(xcat)
        curr_downscale = 4
        up_inp = [torch.cat([bottle_out,nn.functional.avg_pool2d(x,curr_downscale)], dim=1)]
        curr_downscale = curr_downscale//2
        up_out = []
        up_out_mask = []
        for i in range(len(self.up_sampling_convlayers)):
            #self.up_sampling_convlayers(x
            up_out.append(self.up_sampling_ReLU[i](self.up_sampling_inorm[i](self.up_sampling_convlayers[i](up_inp[i]))))
            up_inp.append(torch.cat([up_out[i],nn.functional.avg_pool2d(x,curr_downscale)], dim=1))

            curr_downscale = curr_downscale//2

        net_out = self.finalNonLin(self.final_Layer(up_inp[-1]))
        mask = self.finalNonLin_mask(2.0*self.final_Layer_mask(up_inp[-1]))

        if self.binary_mask:
            mask = ((mask>0.5).float()- mask).detach() + mask

        masked_image = ((1-mask)*x+(mask)*(2.0*net_out))

        if self.g_smooth_layers > 0:
            out_image = self.smooth_layers(masked_image)
        else:
            out_image = masked_image

        if out_diff:
            return out_image, (net_out, mask)
        else:
            return out_image

class GeneratorOnlyMask(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=5, g_smooth_layers=0, binary_mask=0):
        super(GeneratorOnlyMask, self).__init__()

        layers = []
        layers.extend(get_conv_inorm_relu_block(3+c_dim, conv_dim, 7, 1, 3, padtype='zero'))

        # Down-Sampling
        curr_dim = conv_dim

        for i in range(3):
            layers.extend(get_conv_inorm_relu_block(curr_dim, curr_dim*2, 4, 2, 1, padtype='zero'))
            curr_dim = curr_dim * 2

        dilation=1
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dilation=dilation, padtype='zero'))
            if i> 1:
                # This gives dilation as 1, 1, 2, 4, 8, 16
                dilation=dilation*2

        # Up-Sampling Differential layers
        self.up_sampling_convlayers = nn.ModuleList()
        self.up_sampling_inorm= nn.ModuleList()
        self.up_sampling_ReLU= nn.ModuleList()

        for i in range(3):
            self.up_sampling_convlayers.append(nn.ConvTranspose2d(curr_dim+3, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            self.up_sampling_inorm.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            self.up_sampling_ReLU.append(nn.ReLU(inplace=False))

            curr_dim = curr_dim // 2

        self.final_Layer_mask = nn.Conv2d(curr_dim+3, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.finalNonLin_mask = nn.Sigmoid()

        self.g_smooth_layers = g_smooth_layers
        if g_smooth_layers > 0:
            smooth_layers = []
            for i in range(g_smooth_layers):
                smooth_layers.append(nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False))
                smooth_layers.append(nn.Tanh())
            self.smooth_layers= nn.Sequential(*smooth_layers)

        self.binary_mask = binary_mask
        self.main = nn.Sequential(*layers)

    def forward(self, x, c, out_diff = False):
        # replicate spatially and concatenate domain information
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        xcat = torch.cat([x, c], dim=1)
        bottle_out = self.main(xcat)
        curr_downscale = 8
        up_inp = [torch.cat([bottle_out,nn.functional.avg_pool2d(x,curr_downscale)], dim=1)]
        curr_downscale = curr_downscale//2
        up_out = []
        up_out_mask = []
        for i in range(len(self.up_sampling_convlayers)):
            #self.up_sampling_convlayers(x
            up_out.append(self.up_sampling_ReLU[i](self.up_sampling_inorm[i](self.up_sampling_convlayers[i](up_inp[i]))))
            up_inp.append(torch.cat([up_out[i],nn.functional.avg_pool2d(x,curr_downscale)], dim=1))

            curr_downscale = curr_downscale//2

        mask = self.finalNonLin_mask(2.0*self.final_Layer_mask(up_inp[-1]))

        if self.binary_mask:
            mask = ((mask>0.5).float()- mask).detach() + mask

        masked_image = (1-mask)*x #+(mask)*(2.0*net_out))

        out_image = masked_image

        if out_diff:
            return out_image, mask
        else:
            return out_image

class GeneratorMaskAndFeat(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=5, g_smooth_layers=0, binary_mask=0, out_feat_dim=256, up_sampling_type='bilinear',
                 n_upsamp_filt=2, mask_size = 0, additional_cond='image', per_classMask=0, noInpLabel=0, mask_normalize = False, nc=3,
                 use_bias = False, use_bnorm = 0, cond_inp_pnet=0, cond_parallel_track = 0):
        super(GeneratorMaskAndFeat, self).__init__()

        self.lowres_mask = int(mask_size <= 32)
        self.per_classMask = per_classMask
        self.additional_cond = additional_cond
        self.noInpLabel = noInpLabel
        self.mask_normalize = mask_normalize
        layers = []
        # Image is 128 x 128
        layers.extend(get_conv_inorm_relu_block(nc if noInpLabel else nc+c_dim, conv_dim, 7, 1, 3, padtype='zero'))

        # Down-Sampling
        curr_dim = conv_dim
        extra_dim = 3 if self.additional_cond == 'image' else c_dim if self.additional_cond == 'label'else 0

        #-------------------------------------------
        # After downsampling spatial dim is 16 x 16
        # Feat dim is 512
        #-------------------------------------------
        for i in range(3 - self.lowres_mask):
            layers.extend(get_conv_inorm_relu_block(curr_dim, curr_dim*2, 4, 2, 1, padtype='zero'))
            curr_dim = curr_dim * 2

        dilation=1
        #-------------------------------------------
        # After residual spatial dim is 16 x 16
        # Feat dim is 512
        #-------------------------------------------
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dilation=dilation, padtype='zero'))
            if i> 1:
                # This gives dilation as 1, 1, 2, 4, 8, 16
                dilation=dilation*2

        # Up-Sampling Differential layers
        self.up_sampling_convlayers = nn.ModuleList()
        if self.lowres_mask == 0:
            self.up_sampling_inorm= nn.ModuleList()
            self.up_sampling_ReLU= nn.ModuleList()

        self.out_feat_dim = out_feat_dim
        if out_feat_dim > 0:
            featGenLayers = []
            #-------------------------------------------
            # After featGen layers spatial dim is 1 x 1
            # Feat dim is 512
            #-------------------------------------------
            for i in xrange(3):
                featGenLayers.extend(get_conv_inorm_relu_block(curr_dim, curr_dim, 3, 1, 1, padtype='zero'))
                featGenLayers.append(nn.MaxPool2d(2) if i<2 else nn.MaxPool2d(4))

            self.featGenConv = nn.Sequential(*featGenLayers)
            self.featGenLin = nn.Linear(curr_dim, out_feat_dim)

        for i in range(3-self.lowres_mask):
            if self.lowres_mask == 0:
                if up_sampling_type== 't_conv':
                    self.up_sampling_convlayers.append(nn.ConvTranspose2d(curr_dim+extra_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=use_bias))
                elif up_sampling_type == 'nearest':
                    self.up_sampling_convlayers.append(nn.Upsample(scale_factor=2, mode='nearest'))
                    self.up_sampling_convlayers.append(nn.Conv2d(curr_dim+extra_dim, curr_dim//2, kernel_size=3, stride=1, padding=1, bias=use_bias))
                elif up_sampling_type == 'deform':
                    self.up_sampling_convlayers.append(AdaptiveScaleTconv(curr_dim+extra_dim, curr_dim//2, scale=2, n_filters=n_upsamp_filt))
                elif up_sampling_type == 'bilinear':
                    self.up_sampling_convlayers.append(AdaptiveScaleTconv(curr_dim+extra_dim, curr_dim//2, scale=2, use_deform=False, n_filters=n_upsamp_filt))
                self.up_sampling_inorm.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
                self.up_sampling_ReLU.append(nn.ReLU(inplace=False))
            else:
                # In this case just use more residual blocks to drop dimensions
                self.up_sampling_convlayers.append(nn.Sequential(*get_conv_inorm_relu_block(curr_dim+extra_dim, curr_dim//2, 3, 1, 1, padtype='zero')))


            curr_dim = curr_dim // 2

        self.final_Layer_mask = nn.Conv2d(curr_dim+extra_dim, c_dim+1 if per_classMask else 1, kernel_size=7, stride=1, padding=3, bias=True if mask_normalize else use_bias)
        self.finalNonLin_mask = nn.Sigmoid()

        self.g_smooth_layers = g_smooth_layers
        if g_smooth_layers > 0:
            smooth_layers = []
            for i in range(g_smooth_layers):
                smooth_layers.append(nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False))
                smooth_layers.append(nn.Tanh())
            self.smooth_layers= nn.Sequential(*smooth_layers)

        self.binary_mask = binary_mask
        self.main = nn.Sequential(*layers)

    def prepInp(self, feat, x, c, curr_scale):
        if self.additional_cond == 'image':
            up_inp = torch.cat([feat,nn.functional.avg_pool2d(x,curr_scale)], dim=1)
        elif self.additional_cond == 'label':
            up_inp = torch.cat([feat,nn.functional.avg_pool2d(c,curr_scale)], dim=1)
        else:
            up_inp = feat
        return up_inp

    def forward(self, x, c, out_diff = False, binary_mask=False, mask_threshold = 0.3):
        # replicate spatially and concatenate domain information
        bsz = x.size(0)
        if self.per_classMask:
            maxC,cIdx = c.max(dim=1)
            cIdx[maxC==0] = c.size(1) + 1 if self.mask_normalize else c.size(1)

        if self.noInpLabel:
            xcat = x
        else:
            c = c.unsqueeze(2).unsqueeze(3)
            c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
            xcat = torch.cat([x, c], dim=1)
        bottle_out = self.main(xcat)
        curr_downscale = 8 if self.lowres_mask == 0 else 4
        up_inp = [self.prepInp(bottle_out, x, c, curr_downscale)]
        curr_downscale = curr_downscale//2 if self.lowres_mask == 0 else curr_downscale
        up_out = []
        for i in range(len(self.up_sampling_convlayers)):
            #self.up_sampling_convlayers(x
            if type(self.up_sampling_convlayers[i]) == AdaptiveScaleTconv:
                upsampout,_ = self.up_sampling_convlayers[i](up_inp[i])
            else:
                upsampout = self.up_sampling_convlayers[i](up_inp[i])
            up_out.append(self.up_sampling_ReLU[i](self.up_sampling_inorm[i](upsampout)) if self.lowres_mask == 0 else upsampout)
            up_inp.append(self.prepInp(up_out[i], x, c, curr_downscale))

            curr_downscale = curr_downscale//2 if self.lowres_mask == 0 else curr_downscale

        allmasks = self.final_Layer_mask(up_inp[-1])
        if self.mask_normalize:
            allmasks = torch.cat([F.softmax(allmasks, dim=1), torch.zeros_like(allmasks[:,0:1,::]).detach()], dim=1)
        chosenMask = allmasks if (self.per_classMask==0) else allmasks[torch.arange(cIdx.size(0)).long().cuda(),cIdx,::].view(bsz,1,allmasks.size(2), allmasks.size(3))
        if not self.mask_normalize:
            mask = self.finalNonLin_mask(2.0*chosenMask)
        else:
            mask = chosenMask

        if self.out_feat_dim > 0:
            out_feat = self.featGenLin(self.featGenConv(bottle_out).view(bsz,-1))
        else:
            out_feat = None

        if self.binary_mask or binary_mask:
            if self.mask_normalize:
                maxV,_ = allmasks.max(dim=1)
                mask = (torch.ge(mask, maxV.view(mask.size())).float()- mask).detach() + mask
            else:
                mask = ((mask>=mask_threshold).float()- mask).detach() + mask


        #masked_image = (1-mask)*x #+(mask)*(2.0*net_out))

        #out_image = masked_image

        return None, mask, out_feat, allmasks

class GeneratorMaskAndFeat_ImNetBackbone(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=5, g_smooth_layers=0, binary_mask=0, out_feat_dim=256, up_sampling_type='bilinear',
                 n_upsamp_filt=2, mask_size = 0, additional_cond='image', per_classMask=0, noInpLabel=0, mask_normalize = False, nc=3, use_bias = False,
                 net_type='vgg19', use_bnorm = 0, cond_inp_pnet=0):
        super(GeneratorMaskAndFeat_ImNetBackbone, self).__init__()

        self.pnet = Vgg19() if net_type == 'vgg19' else None
        self.per_classMask = per_classMask
        self.additional_cond = additional_cond
        self.noInpLabel = noInpLabel
        self.mask_normalize = mask_normalize
        self.nc = nc
        self.out_feat_dim = out_feat_dim
        self.binary_mask = binary_mask
        # Down-Sampling
        curr_dim = conv_dim
        extra_dim = 3 if self.additional_cond == 'image' else c_dim if self.additional_cond == 'label'else 0
        layers = nn.ModuleList()
        if nc > 3:
            extra_dim = extra_dim + 1
            self.appendGtInp = True
        else:
            self.appendGtInp = False

        ResBlock  = ResidualBlockBnorm if use_bnorm==1 else ResidualBlock if use_bnorm==2 else ResidualBlockNoNorm
        #===========================================================
        # Three blocks of layers:
        # Feature absorb layer --> Residual block --> Upsampling
        #===========================================================
        # First block This takes input features of 512x8x8 dims
        # Upsample to 16x16
        layers.append(nn.Conv2d(512+extra_dim, 512, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU(0.1))
        layers.append(ResBlock(dim_in=512,dilation=1, padtype='zero'))
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        #-----------------------------------------------------------
        # Second Block - This takes input features of 512x16x16 from Layer 1 and 512x16x16 from VGG
        # Upsample to 32x32
        layers.append(nn.Conv2d(1024+extra_dim, 512, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU(0.1))
        layers.append(ResBlock(dim_in=512,dilation=1, padtype='zero'))
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        #-----------------------------------------------------------
        # Third layer
        # This takes input features of 256x32x32 from Layer 1 and 256x32x32 from VGG
        layers.append(nn.Conv2d(512+256+extra_dim, 512, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU(0.1))
        layers.append(ResBlock(dim_in=512,dilation=1, padtype='zero'))

        self.layers = layers

        self.final_Layer_mask = nn.Conv2d(512+extra_dim, c_dim+1 if per_classMask else 1, kernel_size=7, stride=1, padding=3, bias=True if mask_normalize else use_bias)
        self.finalNonLin_mask = nn.Sigmoid()

        self.shift = torch.autograd.Variable(torch.Tensor([-.030, -.088, -.188]).view(1,3,1,1), requires_grad=False).cuda()
        self.scale = torch.autograd.Variable(torch.Tensor([.458, .448, .450]).view(1,3,1,1), requires_grad=False).cuda()

    def prepInp(self, feat, img, c, gtmask):
        if self.additional_cond == 'image':
            up_inp = torch.cat([feat,nn.functional.adaptive_avg_pool2d(img,feat.size(-1))], dim=1)
        elif self.additional_cond == 'label':
            up_inp = torch.cat([feat,c.expand(c.size(0), c.size(1), feat.size(2), feat.size(3))], dim=1)
        else:
            up_inp = feat
        if self.appendGtInp:
            up_inp = torch.cat([up_inp, nn.functional.adaptive_max_pool2d(gtmask,up_inp.size(-1))], dim=1)
        return up_inp

    def forward(self, x, c, out_diff = False, binary_mask=False, mask_threshold = 0.3):
        # replicate spatially and concatenate domain information
        bsz = x.size(0)
        img = x[:,:3,::]
        gtmask = x[:,3:,::]  if self.appendGtInp else None
        if self.per_classMask:
            maxC,cIdx = c.max(dim=1)
            cIdx[maxC==0] = c.size(1) + 1 if self.mask_normalize else c.size(1)

        c = c.unsqueeze(2).unsqueeze(3)
        img = (img - self.shift.expand_as(img))/self.scale.expand_as(img)

        vgg_out = self.pnet(img)
        up_inp = [self.prepInp(vgg_out[-1], img, c, gtmask)]
        for i in range(len(self.layers)):
            #self.up_sampling_convlayers(img
            upsampout = self.layers[i](up_inp[-1])
            up_inp.append(upsampout)
            if i%4 == 3:
                up_inp.append(self.prepInp(torch.cat([up_inp[-1],vgg_out[-1-(i+1)//4]],dim=1), img, c, gtmask))
        up_inp.append(self.prepInp(up_inp[-1], img, c, gtmask))

        allmasks = self.final_Layer_mask(up_inp[-1])
        if self.mask_normalize:
            allmasks = torch.cat([F.softmax(allmasks, dim=1), torch.zeros_like(allmasks[:,0:1,::]).detach()], dim=1)
        chosenMask = allmasks if (self.per_classMask==0) else allmasks[torch.arange(cIdx.size(0)).long().cuda(),cIdx,::].view(bsz,1,allmasks.size(2), allmasks.size(3))
        if not self.mask_normalize:
            mask = self.finalNonLin_mask(2.0*chosenMask)
        else:
            mask = chosenMask

        if self.out_feat_dim > 0:
            out_feat = self.featGenLin(self.featGenConv(bottle_out).view(bsz,-1))
        else:
            out_feat = None

        if self.binary_mask or binary_mask:
            if self.mask_normalize:
                maxV,_ = allmasks.max(dim=1)
                mask = (torch.ge(mask, maxV.view(mask.size())).float()- mask).detach() + mask
            else:
                mask = ((mask>=mask_threshold).float()- mask).detach() + mask


        #masked_image = (1-mask)*x #+(mask)*(2.0*net_out))

        #out_image = masked_image

        return None, mask, out_feat, allmasks

class GeneratorMaskAndFeat_ImNetBackbone_V2(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=5, g_smooth_layers=0, binary_mask=0, out_feat_dim=256, up_sampling_type='bilinear',
                 n_upsamp_filt=2, mask_size = 0, additional_cond='image', per_classMask=0, noInpLabel=0, mask_normalize = False, nc=3, use_bias = False,
                 net_type='vgg19', use_bnorm = 0, cond_inp_pnet=0, cond_parallel_track= 0):
        super(GeneratorMaskAndFeat_ImNetBackbone_V2, self).__init__()

        self.pnet = Vgg19(final_feat_size=mask_size) if net_type == 'vgg19' else None
        self.mask_size = mask_size
        self.per_classMask = per_classMask
        self.additional_cond = additional_cond
        self.noInpLabel = noInpLabel
        self.mask_normalize = mask_normalize
        self.nc = nc
        self.out_feat_dim = out_feat_dim
        self.cond_inp_pnet = cond_inp_pnet
        self.cond_parallel_track = cond_parallel_track
        # Down-Sampling
        curr_dim = conv_dim
        extra_dim = 3 if self.additional_cond == 'image' else c_dim if self.additional_cond == 'label'else 0
        layers = nn.ModuleList()
        if nc > 3:
            extra_dim = extra_dim# + 1
            self.appendGtInp = False #True
        else:
            self.appendGtInp = False

        ResBlock  = ResidualBlockBnorm if use_bnorm==1 else ResidualBlock if use_bnorm==2 else ResidualBlockNoNorm
        #===========================================================
        # Three blocks of layers:
        # Feature absorb layer --> Residual block --> Upsampling
        #===========================================================
        # First block This takes input features of 512x32x32 dims
        # Upsample to 16x16
        start_dim = 512
        gt_cond_dim =  0 if cond_inp_pnet else int(nc>3)*self.cond_parallel_track if self.cond_parallel_track else int(nc>3)
        if self.cond_parallel_track:
            cond_parallel_layers = [] #nn.ModuleList()
            cond_parallel_layers.append(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1))
            cond_parallel_layers.append(nn.LeakyReLU(0.1, inplace=True))
            cond_parallel_layers.append(nn.Conv2d(64, self.cond_parallel_track, kernel_size=3, stride=1, padding=1))
            cond_parallel_layers.append(nn.LeakyReLU(0.1, inplace=True))
            self.cond_parallel_layers = nn.Sequential(*cond_parallel_layers)

        layers.append(nn.Conv2d(512+extra_dim + gt_cond_dim, start_dim, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        layers.append(ResBlock(dim_in=start_dim,dilation=1, padtype='zero'))
        #layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        #-----------------------------------------------------------
        # Second Block - This takes input features of 512x16x16 from Layer 1 and 512x16x16 from VGG
        # Upsample to 32x32
        layers.append(nn.Conv2d(start_dim+extra_dim, start_dim//2, kernel_size=3, stride=1, padding=1))
        start_dim = start_dim // 2
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        layers.append(ResBlock(dim_in=start_dim,dilation=1, padtype='zero'))
        #-----------------------------------------------------------
        # Third layer
        # This takes input features of 256x32x32 from Layer 1 and 256x32x32 from VGG
        layers.append(nn.Conv2d(start_dim+extra_dim, start_dim//2, kernel_size=3, stride=1, padding=1))
        start_dim = start_dim // 2
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        layers.append(ResBlock(dim_in=start_dim,dilation=1, padtype='zero'))

        self.layers = layers

        self.final_Layer_mask = nn.Conv2d(start_dim+extra_dim, c_dim+1 if per_classMask else 1, kernel_size=7, stride=1, padding=3, bias=True if mask_normalize else use_bias)
        self.finalNonLin_mask = nn.Sigmoid()

        self.binary_mask = binary_mask
        self.shift = torch.autograd.Variable(torch.Tensor([-.030, -.088, -.188]).view(1,3,1,1), requires_grad=False).cuda()
        self.scale = torch.autograd.Variable(torch.Tensor([.458, .448, .450]).view(1,3,1,1), requires_grad=False).cuda()

    def prepInp(self, feat, img, c, gtmask):
        if self.additional_cond == 'image':
            up_inp = torch.cat([feat,nn.functional.adaptive_avg_pool2d(img,feat.size(-1))], dim=1)
        elif self.additional_cond == 'label':
            up_inp = torch.cat([feat,c.expand(c.size(0), c.size(1), feat.size(2), feat.size(3))], dim=1)
        else:
            up_inp = feat
        if self.appendGtInp:
            up_inp = torch.cat([up_inp, nn.functional.adaptive_max_pool2d(gtmask,up_inp.size(-1))], dim=1)
        return up_inp

    def forward(self, x, c, out_diff = False, binary_mask=False, mask_threshold = 0.3):
        # replicate spatially and concatenate domain information
        bsz = x.size(0)
        gtmask = x[:,3:,::] if x.size(1) > 3 else None
        img = x[:,:3,::]
        img = (img - self.shift.expand_as(img))/self.scale.expand_as(img)
        if self.cond_inp_pnet:
            img = img*gtmask
        if self.cond_parallel_track and gtmask is not None:
            gtfeat = self.cond_parallel_layers(gtmask)
        else:
            gtfeat = gtmask

        if self.per_classMask:
            maxC,cIdx = c.max(dim=1)
            cIdx[maxC==0] = c.size(1) + 1 if self.mask_normalize else c.size(1)

        c = c.unsqueeze(2).unsqueeze(3)

        vgg_out = self.pnet(img)
        #up_inp = [self.prepInp(vgg_out[-1], img, c, gtmask)]
        if (gtfeat is not None) and (not self.cond_inp_pnet):
            up_inp = [torch.cat([self.prepInp(vgg_out[-1], img, c, gtfeat), nn.functional.adaptive_max_pool2d(gtfeat,vgg_out[-1].size(-1))],dim=1)]
        else:
            up_inp = [self.prepInp(vgg_out[-1], img, c, gtfeat)]

        for i in range(len(self.layers)):
            #self.up_sampling_convlayers(img
            upsampout = self.layers[i](up_inp[-1])
            up_inp.append(upsampout)
            if i%3 == 2:
                up_inp.append(self.prepInp(upsampout, img, c, gtfeat))

        allmasks = self.final_Layer_mask(up_inp[-1])
        if self.mask_normalize:
            allmasks = torch.cat([F.softmax(allmasks, dim=1), torch.zeros_like(allmasks[:,0:1,::]).detach()], dim=1)
        chosenMask = allmasks if (self.per_classMask==0) else allmasks[torch.arange(cIdx.size(0)).long().cuda(),cIdx,::].view(bsz,1,allmasks.size(2), allmasks.size(3))
        if not self.mask_normalize:
            mask = self.finalNonLin_mask(2.0*chosenMask)
        else:
            mask = chosenMask

        if self.out_feat_dim > 0:
            out_feat = self.featGenLin(self.featGenConv(bottle_out).view(bsz,-1))
        else:
            out_feat = None

        if self.binary_mask or binary_mask:
            if self.mask_normalize:
                maxV,_ = allmasks.max(dim=1)
                mask = (torch.ge(mask, maxV.view(mask.size())).float()- mask).detach() + mask
            else:
                mask = ((mask>=mask_threshold).float()- mask).detach() + mask


        #masked_image = (1-mask)*x #+(mask)*(2.0*net_out))

        #out_image = masked_image

        return None, mask, out_feat, allmasks


class GeneratorBoxReconst(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, feat_dim=128, repeat_num=6, g_downsamp_layers=2, dil_start =0,
                 up_sampling_type='t_conv', padtype='zero', nc=3, n_upsamp_filt=1, gen_full_image=0):
        super(GeneratorBoxReconst, self).__init__()

        downsamp_layers = []
        layers = []
        downsamp_layers.extend(get_conv_inorm_relu_block(nc+1, conv_dim, 7, 1, 3, padtype=padtype))
        self.g_downsamp_layers = g_downsamp_layers
        self.gen_full_image = gen_full_image

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(g_downsamp_layers):
            downsamp_layers.extend(get_conv_inorm_relu_block(curr_dim, curr_dim*2, 4, 2, 1, padtype=padtype))
            curr_dim = curr_dim * 2

        # Bottleneck
        # Here- input the target features
        dilation=1
        if feat_dim > 0:
            layers.extend(get_conv_inorm_relu_block(curr_dim+feat_dim, curr_dim, 3, 1, 1, padtype=padtype, dilation=dilation))
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dilation=dilation, padtype=padtype))
            if i> dil_start:
                # This gives dilation as 1, 1, 2, 4, 8, 16
                dilation=dilation*2

        # Up-Sampling
        for i in range(g_downsamp_layers):
            if up_sampling_type== 't_conv':
                layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            elif up_sampling_type == 'nearest':
                layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
                layers.append(nn.Conv2d(curr_dim, curr_dim//2, kernel_size=3, stride=1, padding=1, bias=False))
            elif up_sampling_type == 'deform':
                layers.append(AdaptiveScaleTconv(curr_dim+(self.gen_full_image * curr_dim//2), curr_dim//2, scale=2, n_filters=n_upsamp_filt))
            elif up_sampling_type == 'bilinear':
                layers.append(AdaptiveScaleTconv(curr_dim+(self.gen_full_image * curr_dim//2), curr_dim//2, scale=2, use_deform=False))

            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.LeakyReLU(0.1,inplace=True))
            curr_dim = curr_dim // 2

        pad=3
        if padtype=='reflection':
            layers.append(nn.ReflectionPad2d(pad)); pad=0
        layers.append(nn.Conv2d(curr_dim, nc, kernel_size=7, stride=1, padding=pad, bias=False))
        # Remove this non-linearity or use 2.0*tanh ?
        layers.append(nn.Tanh())
        self.hardtanh = nn.Hardtanh(min_val=-1, max_val=1)
        self.downsample = nn.Sequential(*downsamp_layers)
        #self.generate = nn.Sequential(*layers)
        self.generate = nn.ModuleList(layers)

    def forward(self, x, feat, out_diff = False):
        w, h = x.size(2), x.size(3)
        # This is just to makes sure that when we pass it through the downsampler we don't lose some width and height
        xI = F.pad(x,(0,(8-h%8)%8,0,(8 - w%8)%8),mode='replicate')
        #print(xI.device, [p.device for p in self.parameters()][0])
        if self.gen_full_image:
            dowOut = [xI]
            for i in xrange(self.g_downsamp_layers+1):
                dowOut.append(self.downsample[3*i+2](self.downsample[3*i+1](self.downsample[3*i](dowOut[-1]))))
            downsamp_out = dowOut[-1]
        else:
            downsamp_out = self.downsample(xI)

        # replicate spatially and concatenate domain information
        if feat is not None:
            feat = feat.unsqueeze(2).unsqueeze(3)
            feat = feat.expand(feat.size(0), feat.size(1), downsamp_out.size(2), downsamp_out.size(3))

            genInp = torch.cat([downsamp_out, feat], dim=1)
        else:
            genInp = downsamp_out
        #net_out = self.generate(genInp)
        outs = [genInp]
        feat_out = []
        d_count = -2
        for i,l in enumerate(self.generate):
            if type(l) is not AdaptiveScaleTconv:
                outs.append(l(outs[i]))
            else:
                deform_out, deform_params = l(outs[i], extra_inp = None if not self.gen_full_image else dowOut[d_count])
                d_count = d_count-1
                outs.append(deform_out)
                feat_out.append(deform_params)
        outImg = outs[-1][:,:,:w,:h]
        if not out_diff:
            return outImg
        else:
            return outImg, feat_out


class Discriminator(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6, init_stride=2, classify_branch=1, max_filters=None, nc=3,
                 use_norm=0, d_kernel_size = 4, patch_size = 2, use_tv_inp = 0, use_maxpool=False, use_selfatt=0, use_sn=0):
        super(Discriminator, self).__init__()

        layers = []
        self.use_tv_inp = use_tv_inp
        if self.use_tv_inp:
            self.tvWeight = torch.zeros((1,3,3,3))
            self.tvWeight[0,:,1,1] = -2.0
            self.tvWeight[0,:,1,2] = 1.0; self.tvWeight[0,:,2,1] = 1.0;
            self.tvWeight = Variable(self.tvWeight,requires_grad=False).cuda()
        # Start training
        self.nc=nc + use_tv_inp
        # UGLY HACK
        dkz = d_kernel_size if d_kernel_size > 1 else 4
        pad = 1 if dkz == 3 else 2
        if dkz == 3:
            #layers.append(nn.ReplicationPad2d(pad))
            layers.append(snconv2d(self.nc, conv_dim, kernel_size=3, stride=1, use_sn=use_sn))
            layers.append(nn.LeakyReLU(0.02, inplace=True))
            #layers.append(nn.ReplicationPad2d(pad))
            layers.append(snconv2d(conv_dim, conv_dim, kernel_size=dkz, stride=init_stride, use_sn=use_sn))
        else:
            #layers.append(nn.ReplicationPad2d(pad))
            layers.append(snconv2d(self.nc, conv_dim, kernel_size=dkz, stride=init_stride, use_sn=use_sn))
        if use_norm==1:
            layers.append(nn.BatchNorm2d(conv_dim))
        elif use_norm==2:
            layers.append(nn.InstanceNorm2d(conv_dim))
        layers.append(nn.LeakyReLU(0.02, inplace=True))

        curr_dim = conv_dim
        assert(patch_size <= 64)
        n_downSamp = int(np.log2(image_size// patch_size)) +2 - init_stride
        # 64 - 32 - 16 -
        for i in range(1, repeat_num):
            out_dim =  curr_dim*2 if max_filters is None else min(curr_dim*2, max_filters)
            stride = 1 if ((i >= n_downSamp) or use_maxpool) else 2
            #layers.append(nn.ReplicationPad2d(pad))
            layers.append(snconv2d(curr_dim, out_dim, kernel_size=dkz, stride=stride, use_sn=use_sn))
            if use_norm==1:
                layers.append(nn.BatchNorm2d(out_dim))
            elif use_norm==2:
                layers.append(nn.InstanceNorm2d(out_dim))
            if i == 3 and use_selfatt > 0:
                layers.append(Self_Attn(out_dim))
            if i == 4 and use_selfatt > 1:
                layers.append(Self_Attn(out_dim))
            if use_maxpool and (i < n_downSamp):
                layers.append(nn.MaxPool2d(2))
            layers.append(nn.LeakyReLU(0.02, inplace=True))
            curr_dim = out_dim

        k_size = int(image_size / np.power(2, repeat_num)) + 2- init_stride
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Sequential(*[snconv2d(curr_dim, 1 , kernel_size=1 if patch_size<dkz else dkz, stride=1,bias=False, use_sn=use_sn)])
        self.classify_branch = classify_branch
        if classify_branch==1:
            self.conv2 = snconv2d(curr_dim, c_dim, kernel_size=k_size, bias=False, use_sn=use_sn)
        elif classify_branch == 2:
            # This is the projection discriminator!
            #self.embLayer = nn.utils.weight_norm(nn.Linear(c_dim, curr_dim, bias=False))
            self.embLayer = snlinear(c_dim, curr_dim, bias=False, use_sn=use_sn)

    def forward(self, x, label=None, get_feat=False):
        #import ipdb; ipdb.set_trace()
        if self.use_tv_inp:
            tvImg = torch.abs(F.conv2d(F.pad(x,(1,1,1,1),mode='replicate'),self.tvWeight))
            x = torch.cat([x,tvImg],dim=1)
        sz = x.size()
        h_out = []
        if not get_feat:
            h = self.main(x)
        else:
            h = x
            for i in range(len(self.main)):
                h = self.main[i](h)
                if type(self.main[i]) == nn.Conv2d and (i>=2):
                    h_out.append(h)

        out_real = self.conv1(h)
        if self.classify_branch==1:
            out_aux = self.conv2(h)
            return out_real.view(sz[0],-1), out_aux.squeeze(), h_out
        elif self.classify_branch==2:
            lab_emb = self.embLayer(label)
            #out_aux = (lab_emb.unsqueeze(-1).unsqueeze(-1) * F.max_pool2d(h,4,stride=1)).sum(dim=1)
            out_aux = (lab_emb.unsqueeze(-1).unsqueeze(-1) * F.adaptive_max_pool2d(h,1)).sum(dim=1)
            return (out_real.squeeze() + out_aux), None, h_out
            #return (F.avg_pool2d(out_real,2).squeeze()).view(-1,1)
        else:
            return out_real.squeeze(), None, h_out

class DiscriminatorSmallPatch(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=1, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=k_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_real = self.conv1(h)
        out_aux = self.conv2(h)
        return out_real.squeeze(), out_aux.squeeze()

class DiscriminatorGAP(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=3, init_stride=1,  max_filters=None, nc=3, use_bnorm=False):
        super(DiscriminatorGAP, self).__init__()

        layers = []
        self.nc=nc
        self.c_dim = c_dim
        layers.append(nn.Conv2d(nc, conv_dim, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(conv_dim))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        layers.append(nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1))
        layers.append(nn.MaxPool2d(2))
        if use_bnorm:
            layers.append(nn.BatchNorm2d(conv_dim))
        layers.append(nn.LeakyReLU(0.1, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            out_dim =  curr_dim*2 if max_filters is None else min(curr_dim*2, max_filters)
            layers.append(nn.Conv2d(curr_dim, out_dim, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_dim))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
            layers.append(ResidualBlockBnorm(dim_in=out_dim, dilation=1, padtype='zero'))
            if (i < 4):
                # We want to have 8x8 resolution vefore GAP input
                layers.append(nn.MaxPool2d(2))
            curr_dim = out_dim

        self.main = nn.Sequential(*layers)
        self.globalPool = nn.AdaptiveAvgPool2d(1)
        self.classifyFC = nn.Linear(curr_dim, c_dim, bias=False)

    def forward(self, x, label=None):
        bsz = x.size(0)
        sz = x.size()
        h = self.main(x)
        out_aux = self.classifyFC(self.globalPool(h).view(bsz, -1))
        return None, out_aux.view(bsz,self.c_dim)

class DiscriminatorGAP_ImageNet(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, c_dim = 5, net_type='vgg19', max_filters=None, global_pool='mean',use_bias=False, class_ftune = 0):
        super(DiscriminatorGAP_ImageNet, self).__init__()

        layers = []
        nFilt = 512 if max_filters is None else max_filters
        self.pnet = Vgg19(only_last=True) if net_type == 'vgg19' else None
        if class_ftune > 0.:
            pAll = list(self.pnet.named_parameters())
            # Multiply by two for weight and bias
            for pn in pAll[::-1][:2*class_ftune]:
                pn[1].requires_grad = True
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        layers.append(nn.Conv2d(512, nFilt, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(nFilt))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        layers.append(nn.Conv2d(nFilt, nFilt, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(nFilt))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        self.layers = nn.Sequential(*layers)
        self.globalPool = nn.AdaptiveAvgPool2d(1) if global_pool == 'mean' else nn.AdaptiveMaxPool2d(1)
        self.classifyFC = nn.Linear(nFilt, c_dim, bias=use_bias)
        self.shift = torch.autograd.Variable(torch.Tensor([-.030, -.088, -.188]).view(1,3,1,1), requires_grad=False).cuda()
        self.scale = torch.autograd.Variable(torch.Tensor([.458, .448, .450]).view(1,3,1,1), requires_grad=False).cuda()
        self.c_dim  = c_dim

    def forward(self, x, label=None, get_feat = False):
        bsz = x.size(0)
        sz = x.size()
        x = (x - self.shift.expand_as(x))/self.scale.expand_as(x)
        vOut = self.pnet(x)
        h = self.layers(vOut)
        out_aux = self.classifyFC(self.globalPool(h).view(bsz, -1))
        if get_feat:
            return None, out_aux.view(bsz,self.c_dim), h
        else:
            return None, out_aux.view(bsz,self.c_dim)

class DiscriminatorGAP_ImageNet_Weldon(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, c_dim = 5, net_type='vgg19', max_filters=None, global_pool='mean', topk=3, mink=3, use_bias=False):
        super(DiscriminatorGAP_ImageNet_Weldon, self).__init__()

        layers = []
        self.topk = topk
        self.mink = mink
        nFilt = 512 if max_filters is None else max_filters
        self.pnet = Vgg19(only_last=True) if net_type == 'vgg19' else None
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        layers.append(nn.Conv2d(512, nFilt, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(nFilt))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        layers.append(nn.Conv2d(nFilt, nFilt, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(nFilt))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        self.layers = nn.Sequential(*layers)
        #self.AggrConv = nn.conv2d(nFilt, c_dim, kernel_size=1, stride=1, bias=False)
        self.classifyConv = nn.Conv2d(nFilt, c_dim, kernel_size=1, stride=1, bias=use_bias)
        self.globalPool = nn.AdaptiveAvgPool2d(1) if global_pool == 'mean' else nn.AdaptiveMaxPool2d(1)
        self.shift = torch.autograd.Variable(torch.Tensor([-.030, -.088, -.188]).view(1,3,1,1), requires_grad=False).cuda()
        self.scale = torch.autograd.Variable(torch.Tensor([.458, .448, .450]).view(1,3,1,1), requires_grad=False).cuda()
        self.c_dim  = c_dim

    def forward(self, x, label=None, get_feat = False):
        bsz = x.size(0)
        sz = x.size()
        x = (x - self.shift.expand_as(x))/self.scale.expand_as(x)
        vOut = self.pnet(x)
        h = self.layers(vOut)
        classify_out = self.classifyConv(h)
        if self.topk > 0:
            topk_vals, topk_idx = classify_out.view(bsz,self.c_dim,-1).topk(self.topk)
            out_aux = topk_vals.sum(dim=-1)
            if self.mink > 0:
                mink_vals, mink_idx = classify_out.view(bsz,self.c_dim,-1).topk(self.mink, largest=False)
                out_aux = out_aux + mink_vals.sum(dim=-1)
        else:
            out_aux = self.globalPool(classify_out).view(bsz,-1)
        if get_feat:
            return None, out_aux.view(bsz,self.c_dim), h
        else:
            return None, out_aux.view(bsz,self.c_dim)

class DiscriminatorBBOX(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=64, conv_dim=64, c_dim=5, repeat_num=6):
        super(DiscriminatorBBOX, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=k_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_aux = self.conv2(h)
        return out_aux.squeeze()

class DiscriminatorGlobalLocal(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, bbox_size = 64, conv_dim=64, c_dim=5, repeat_num_global=6, repeat_num_local=5, nc=3):
        super(DiscriminatorGlobalLocal, self).__init__()

        maxFilt = 512 if image_size==128 else 128
        globalLayers = []
        globalLayers.append(nn.Conv2d(nc, conv_dim, kernel_size=4, stride=2, padding=1,bias=False))
        globalLayers.append(nn.LeakyReLU(0.2, inplace=True))

        localLayers = []
        localLayers.append(nn.Conv2d(nc, conv_dim, kernel_size=4, stride=2, padding=1, bias=False))
        localLayers.append(nn.LeakyReLU(0.2, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num_global):
            globalLayers.append(nn.Conv2d(curr_dim, min(curr_dim*2,maxFilt), kernel_size=4, stride=2, padding=1, bias=False))
            globalLayers.append(nn.LeakyReLU(0.2, inplace=True))
            curr_dim = min(curr_dim * 2, maxFilt)

        curr_dim = conv_dim
        for i in range(1, repeat_num_local):
            localLayers.append(nn.Conv2d(curr_dim, min(curr_dim * 2, maxFilt), kernel_size=4, stride=2, padding=1, bias=False))
            localLayers.append(nn.LeakyReLU(0.2, inplace=True))
            curr_dim = min(curr_dim * 2, maxFilt)

        k_size_local = int(bbox_size/ np.power(2, repeat_num_local))
        k_size_global = int(image_size/ np.power(2, repeat_num_global))

        self.mainGlobal = nn.Sequential(*globalLayers)
        self.mainLocal = nn.Sequential(*localLayers)

        # FC 1 for doing real/fake
        self.fc1 = nn.Linear(curr_dim*(k_size_local**2+k_size_global**2), 1, bias=False)

        # FC 2 for doing classification only on local patch
        if c_dim > 0:
            self.fc2 = nn.Linear(curr_dim*(k_size_local**2), c_dim, bias=False)
        else:
            self.fc2 = None

    def forward(self, x, boxImg, classify=False):
        bsz = x.size(0)
        h_global = self.mainGlobal(x)
        h_local = self.mainLocal(boxImg)
        h_append = torch.cat([h_global.view(bsz,-1), h_local.view(bsz,-1)], dim=-1)
        out_rf = self.fc1(h_append)
        out_cls = self.fc2(h_local.view(bsz,-1)) if classify and (self.fc2 is not None) else None
        return out_rf.squeeze(), out_cls, h_append

class DiscriminatorGlobalLocal_SN(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, bbox_size = 64, conv_dim=64, c_dim=5, repeat_num_global=6, repeat_num_local=5, nc=3):
        super(DiscriminatorGlobalLocal_SN, self).__init__()

        maxFilt = 512 if image_size==128 else 128
        globalLayers = []
        globalLayers.append(SpectralNorm(nn.Conv2d(nc, conv_dim, kernel_size=4, stride=2, padding=1,bias=False)))
        globalLayers.append(nn.LeakyReLU(0.2, inplace=True))

        localLayers = []
        localLayers.append(SpectralNorm(nn.Conv2d(nc, conv_dim, kernel_size=4, stride=2, padding=1, bias=False)))
        localLayers.append(nn.LeakyReLU(0.2, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num_global):
            globalLayers.append(SpectralNorm(nn.Conv2d(curr_dim, min(curr_dim*2,maxFilt), kernel_size=4, stride=2, padding=1, bias=False)))
            globalLayers.append(nn.LeakyReLU(0.2, inplace=True))
            curr_dim = min(curr_dim * 2, maxFilt)

        curr_dim = conv_dim
        for i in range(1, repeat_num_local):
            localLayers.append(SpectralNorm(nn.Conv2d(curr_dim, min(curr_dim * 2, maxFilt), kernel_size=4, stride=2, padding=1, bias=False)))
            localLayers.append(nn.LeakyReLU(0.2, inplace=True))
            curr_dim = min(curr_dim * 2, maxFilt)

        k_size_local = int(bbox_size/ np.power(2, repeat_num_local))
        k_size_global = int(image_size/ np.power(2, repeat_num_global))

        self.mainGlobal = nn.Sequential(*globalLayers)
        self.mainLocal = nn.Sequential(*localLayers)

        # FC 1 for doing real/fake
        self.fc1 = SpectralNorm(nn.Linear(curr_dim*(k_size_local**2+k_size_global**2), 1, bias=False))

        # FC 2 for doing classification only on local patch
        self.fc2 = SpectralNorm(nn.Linear(curr_dim*(k_size_local**2), c_dim, bias=False))

    def forward(self, x, boxImg, classify=False):
        bsz = x.size(0)
        h_global = self.mainGlobal(x)
        h_local = self.mainLocal(boxImg)
        h_append = torch.cat([h_global.view(bsz,-1), h_local.view(bsz,-1)], dim=-1)
        out_rf = self.fc1(h_append)
        out_cls = self.fc2(h_local.view(bsz,-1)) if classify else None
        return out_rf.squeeze(), out_cls, h_append


##-------------------------------------------------------
## Implementing perceptual loss using VGG19
##-------------------------------------------------------

class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)

def normalize_tensor(in_feat,eps=1e-10):
    # norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1)).view(in_feat.size()[0],1,in_feat.size()[2],in_feat.size()[3]).repeat(1,in_feat.size()[1],1,1)
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1)).view(in_feat.size()[0],1,in_feat.size()[2],in_feat.size()[3])
    return in_feat/(norm_factor.expand_as(in_feat)+eps)

class squeezenet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(squeezenet, self).__init__()
        pretrained_features = models.squeezenet1_1(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        self.slice7 = torch.nn.Sequential()
        self.N_slices = 7
        for x in range(2):
            self.slice1.add_module(str(x), pretrained_features[x])
        for x in range(2,5):
            self.slice2.add_module(str(x), pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), pretrained_features[x])
        for x in range(10, 11):
            self.slice5.add_module(str(x), pretrained_features[x])
        for x in range(11, 12):
            self.slice6.add_module(str(x), pretrained_features[x])
        for x in range(12, 13):
            self.slice7.add_module(str(x), pretrained_features[x])


        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        h = self.slice6(h)
        h_relu6 = h
        h = self.slice7(h)
        h_relu7 = h
        out = [h_relu1,h_relu2,h_relu3,h_relu4,h_relu5,h_relu6,h_relu7]

        return out


class VGGLoss(nn.Module):
    def __init__(self, network = 'vgg', use_perceptual=True, imagenet_norm = False, use_style_loss=0):
        super(VGGLoss, self).__init__()
        self.criterion = nn.L1Loss()

        self.use_style_loss = use_style_loss

        if network == 'vgg':
            self.chns = [64,128,256,512,512]
        else:
            self.chns = [64,128,256,384,384,512,512]

        if use_perceptual:
            self.use_perceptual = True
            self.lin0 = NetLinLayer(self.chns[0],use_dropout=False)
            self.lin1 = NetLinLayer(self.chns[1],use_dropout=False)
            self.lin2 = NetLinLayer(self.chns[2],use_dropout=False)
            self.lin3 = NetLinLayer(self.chns[3],use_dropout=False)
            self.lin4 = NetLinLayer(self.chns[4],use_dropout=False)
            self.lin0.cuda()
            self.lin1.cuda()
            self.lin2.cuda()
            self.lin3.cuda()
            self.lin4.cuda()

        # Do this since the tensors have already been normalized to have mean and variance [0.5,0.5,0.5]
        self.imagenet_norm = imagenet_norm
        if not self.imagenet_norm:
            self.shift = torch.autograd.Variable(torch.Tensor([-.030, -.088, -.188]).view(1,3,1,1)).cuda()
            self.scale = torch.autograd.Variable(torch.Tensor([.458, .448, .450]).view(1,3,1,1)).cuda()

        self.net_type = network
        if network == 'vgg':
            self.pnet = Vgg19().cuda()
            self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        else:
            self.pnet = squeezenet().cuda()
            self.weights = [1.0]*7
            if use_perceptual:
                self.lin5 = NetLinLayer(self.chns[5],use_dropout=False)
                self.lin6 = NetLinLayer(self.chns[6],use_dropout=False)
                self.lin5.cuda()
                self.lin6.cuda()
        if self.use_perceptual:
            self.load_state_dict(torch.load('/BS/rshetty-wrk/work/code/controlled-generation/trained_models/perceptualSim/'+network+'.pth'), strict=False)
        for param in self.parameters():
            param.requires_grad = False

    def gram(self, x):
        a, b, c, d = x.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = x.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    def forward(self, x, y):
        x, y = x.expand(x.size(0), 3, x.size(2), x.size(3)), y.expand(y.size(0), 3, y.size(2), y.size(3))
        if not self.imagenet_norm:
            x = (x - self.shift.expand_as(x))/self.scale.expand_as(x)
            y = (y - self.shift.expand_as(y))/self.scale.expand_as(y)

        x_vgg, y_vgg = self.pnet(x), self.pnet(y)
        loss = 0
        if self.use_perceptual:
            normed_x = [normalize_tensor(x_vgg[kk]) for (kk, out0) in enumerate(x_vgg)]
            normed_y = [normalize_tensor(y_vgg[kk]) for (kk, out0) in enumerate(y_vgg)]
            diffs = [(normed_x[kk]-normed_y[kk].detach())**2 for (kk,out0) in enumerate(x_vgg)]
            loss = self.lin0.model(diffs[0]).mean()
            loss = loss + self.lin1.model(diffs[1]).mean()
            loss = loss + self.lin2.model(diffs[2]).mean()
            loss = loss + self.lin3.model(diffs[3]).mean()
            loss = loss + self.lin4.model(diffs[4]).mean()
            if(self.net_type=='squeeze'):
                loss = loss + self.lin5.model(diffs[5]).mean()
                loss = loss + self.lin6.model(diffs[6]).mean()
            if self.use_style_loss:
                style_loss = 0.
                for kk in xrange(3, len(x_vgg)):
                    style_loss += self.criterion(self.gram(x_vgg[kk]), self.gram(y_vgg[kk]))
                loss += self.use_style_loss * style_loss
        else:
            for i in range(len(x_vgg)):
                loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

