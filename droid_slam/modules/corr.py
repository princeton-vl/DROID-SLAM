import torch
import torch.nn.functional as F

import droid_backends

class CorrSampler(torch.autograd.Function):

    @staticmethod
    def forward(ctx, volume, coords, radius):
        ctx.save_for_backward(volume,coords)
        ctx.radius = radius
        corr, = droid_backends.corr_index_forward(volume, coords, radius)
        return corr

    @staticmethod
    def backward(ctx, grad_output):
        volume, coords = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_volume, = droid_backends.corr_index_backward(volume, coords, grad_output, ctx.radius)
        return grad_volume, None, None


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=3):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, num, h1, w1, h2, w2 = corr.shape
        corr = corr.reshape(batch*num*h1*w1, 1, h2, w2)
        
        for i in range(self.num_levels):
            self.corr_pyramid.append(
                corr.view(batch*num, h1, w1, h2//2**i, w2//2**i))
            corr = F.avg_pool2d(corr, 2, stride=2)
            
    def __call__(self, coords):
        out_pyramid = []
        batch, num, ht, wd, _ = coords.shape
        coords = coords.permute(0,1,4,2,3)
        coords = coords.contiguous().view(batch*num, 2, ht, wd)
        
        for i in range(self.num_levels):
            corr = CorrSampler.apply(self.corr_pyramid[i], coords/2**i, self.radius)
            out_pyramid.append(corr.view(batch, num, -1, ht, wd))

        return torch.cat(out_pyramid, dim=2)

    def cat(self, other):
        for i in range(self.num_levels):
            self.corr_pyramid[i] = torch.cat([self.corr_pyramid[i], other.corr_pyramid[i]], 0)
        return self

    def __getitem__(self, index):
        for i in range(self.num_levels):
            self.corr_pyramid[i] = self.corr_pyramid[i][index]
        return self


    @staticmethod
    def corr(fmap1, fmap2):
        """ all-pairs correlation """
        batch, num, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.reshape(batch*num, dim, ht*wd) / 4.0
        fmap2 = fmap2.reshape(batch*num, dim, ht*wd) / 4.0
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        return corr.view(batch, num, ht, wd, ht, wd)


class CorrLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fmap1, fmap2, coords, ii, jj, r):
        ctx.r = r
        ctx.save_for_backward(fmap1, fmap2, coords, ii, jj)
        corr, = droid_backends.altcorr_forward(fmap1, fmap2, coords, ii, jj, ctx.r)
        return corr

    @staticmethod
    def backward(ctx, grad_corr):
        fmap1, fmap2, coords = ctx.saved_tensors
        fmap1_grad, fmap2_grad, coords_grad, ii, jj = \
            droid_backends.altcorr_backward(fmap1, fmap2, coords, ii, jj, grad_corr, ctx.r)
        return fmap1_grad, fmap2_grad, coords_grad, None

class AltCorrBlock:
    def __init__(self, fmaps, num_levels=4, radius=3):
        self.num_levels = num_levels
        self.radius = radius

        B, N, C, H, W = fmaps.shape
        fmaps = fmaps.view(B*N, C, H, W)
        
        self.pyramid = []
        for i in range(self.num_levels):
            sz = (B, N, C, H//2**i, W//2**i)
            self.pyramid.append(fmaps.view(*sz))
            fmaps = F.avg_pool2d(fmaps, 2, stride=2)


    def __call__(self, coords, ii, jj):

        coords = coords.permute(0, 1, 4, 2, 3).contiguous()

        corr_list = []
        for i in range(self.num_levels):
            corr = CorrLayer.apply(
                self.pyramid[0], self.pyramid[i], coords / 2**i, ii, jj, self.radius
            )

            corr_list.append(corr.flatten(2, 3))

        corr = torch.stack(corr_list, dim=2).flatten(2, 3)
        return corr
