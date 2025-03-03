import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from NAFNet import NAFBlock

class PUNNet(nn.Module):
    def __init__(self, factor, nch_in, nch_out, dilation, nch_ker, nblk):
        super().__init__()
        self.factor = factor
        self.dilation = dilation
        
        ly = []
        ly += [ nn.Conv2d(nch_in, nch_ker, kernel_size=1) ]
        self.head = nn.Sequential(*ly)
        
        ly = []
        ly += [ nn.Conv2d(nch_ker,  nch_ker,    kernel_size=1) ]
        ly += [ nn.Conv2d(nch_ker,    nch_ker//2, kernel_size=1) ]
        ly += [ nn.Conv2d(nch_ker//2, nch_out,    kernel_size=1) ]
        self.tail = nn.Sequential(*ly)
        
        self.masked_conv = nn.Sequential(CentralMaskedConv2d(nch_ker,nch_ker, kernel_size=2*dilation-1, stride=1, padding=dilation-1),
                                  nn.Conv2d(nch_ker, nch_ker, kernel_size=1),
                                  nn.Conv2d(nch_ker, nch_ker, kernel_size=1)
                                  )
    
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        chan = nch_ker
        self.encoders.append(
            nn.Sequential(NAFBlock(chan, dilation))
        )
        self.downs.append(
            nn.Sequential(nn.Conv2d(chan, chan//2, kernel_size=1, stride=1),
                          Downsample(dilation)
            )
        )
        
        chan = chan * 2
        
        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan, dilation) for _ in range(nblk)]
            )
            
        self.ups.append(
            nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1),
                Upsample(dilation)
            )
        )
        chan = chan // 2
        self.decoders.append(
            nn.Sequential(
                NAFBlock(chan, dilation))
        )
        
    def forward(self, x, refine=False):
        
        b, c, h, w = x.shape
        fa = self.dilation**2
        
        if self.factor > 1:
            x = pixel_shuffle_down_sampling(x, self.factor, pad=0)
        else:
            x = x
        
        if h % fa != 0:
            p = 1
            x = F.pad(x, (p, p, p, p), 'reflect')
        else:
            x = x
        
        x = self.head(x)
        
        x = self.masked_conv(x)

        encs = []

        for i, (encoder, down) in enumerate(zip(self.encoders, self.downs)):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for i, (decoder, up, enc_skip) in enumerate(zip(self.decoders, self.ups, encs[::-1])):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        
        x = self.tail(x)
        
        if h % fa != 0:
            x = x[:,:,p:-p,p:-p]
        else:
            x = x
        
        if self.factor > 1:
            x = pixel_shuffle_up_sampling(x, self.factor, pad=0)
        else:
            x = x
        
        return x

class CentralMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH//2, kH//2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)
    
class Downsample(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.dilation=dilation

    def forward(self, x):
        B,C,H,W = x.shape
        x = rearrange(x, 'b c (hd h) (wd w) -> b (c hd wd) h w', h=self.dilation**2, w=self.dilation**2)
        x = rearrange(x, 'b c (hn hh) (wn ww) -> b c (hn wn) hh ww', hh=self.dilation, ww=self.dilation)
        x = rearrange(x, 'b (c hd wd) cc hh ww-> b (c cc) (hd hh) (wd ww)', hd=H//(self.dilation**2), wd=W//(self.dilation**2))
        return x
    
class Upsample(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.dilation=dilation

    def forward(self, x):
        B,C,H,W = x.shape
        x = rearrange(x, 'b (c cc) (hd hh) (wd ww) -> b (c hd wd) cc hh ww', cc = self.dilation**2, hh=self.dilation, ww=self.dilation)
        x = rearrange(x, 'b c (hn wn) hh ww -> b c (hn hh) (wn ww)', hn=self.dilation, wn=self.dilation)
        x = rearrange(x, 'b (c hd wd) h w -> b c (hd h) (wd w)', hd=H//self.dilation, wd=W//self.dilation)
        return x
    

def pixel_shuffle_down_sampling(x:torch.Tensor, f:int, pad:int=0, pad_value:float=0.):
    '''
    pixel-shuffle down-sampling (PD) from "When AWGN-denoiser meets real-world noise." (AAAI 2019)
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad between each down-sampled images
        pad_value (float) : padding value
    Return:
        pd_x (Tensor) : down-shuffled image tensor with pad or not
    '''
    # single image tensor
    if len(x.shape) == 3:
        c,w,h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,3,2,4).reshape(c, w+2*f*pad, h+2*f*pad)
    # batched image tensor
    else:
        b,c,w,h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(b,c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,2,4,3,5).reshape(b,c,w+2*f*pad, h+2*f*pad)

def pixel_shuffle_up_sampling(x:torch.Tensor, f:int, pad:int=0):
    '''
    inverse of pixel-shuffle down-sampling (PD)
    see more details about PD in pixel_shuffle_down_sampling()
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad will be removed
    '''
    # single image tensor
    if len(x.shape) == 3:
        c,w,h = x.shape
        before_shuffle = x.view(c,f,w//f,f,h//f).permute(0,1,3,2,4).reshape(c*f*f,w//f,h//f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)
    # batched image tensor
    else:
        b,c,w,h = x.shape
        before_shuffle = x.view(b,c,f,w//f,f,h//f).permute(0,1,2,4,3,5).reshape(b,c*f*f,w//f,h//f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)