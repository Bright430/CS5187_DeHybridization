import os,sys,cv2,numpy as np
sys.path.insert(0,'.')
import arcface, common

def psf_otf(shape,sigma):
    h,w=shape; fy=np.fft.fftfreq(h)[:,None]; fx=np.fft.fftfreq(w)[None,:]
    return np.exp(-2*np.pi**2*sigma**2*(fx**2+fy**2))

def richardson_lucy(y,H,iters=30):
    # one-sided deconvolution with Gaussian PSF; y assumed nonneg after shifting
    y0=np.clip(y,0,None)+1e-6
    x=np.full_like(y0,y0.mean())
    Hc=np.conj(H)
    for _ in range(iters):
        blur=np.real(np.fft.ifft2(np.fft.fft2(x)*H))
        ratio=y0/(blur+1e-6)
        x=x*np.real(np.fft.ifft2(np.fft.fft2(ratio)*Hc))
        x=np.clip(x,0,None)
    return np.clip(x,0,1)

sig=6.0; Htf=psf_otf((512,512),sig)
raw=[]; rl=[]; fused=[]
for k in range(30):
    Himg=cv2.imread(f'data/hybrid/{k:02d}_s6.png',0).astype(np.float32)/255
    E=cv2.imread(f'data/truth/{k:02d}_E.png',0).astype(np.float32)/255
    M=cv2.imread(f'data/truth/{k:02d}_M.png',0).astype(np.float32)/255
    eE,eM=arcface.embed_many([E,M]); W=arcface.warp_matrix(Himg)
    L,B=common.band_split(Himg,sig)
    # normalize L to [0,1] before RL (L is already blurred M in [0,1])
    x=richardson_lucy(L,Htf,iters=30)
    fu=np.clip(x+0.7*B,0,1)
    for name,img,arr in (('raw',L,raw),('rl',x,rl),('fused',fu,fused)):
        e=arcface.embed_with(img,W)
        arr.append((float(e@eM),float(e@eE)))
for name,a in (('raw LP',raw),('RL decon (30 it)',rl),('RL + 0.7HP fused',fused)):
    x=np.array(a); print(f'{name}: cos to M {x[:,0].mean():.3f} +- {x[:,0].std(ddof=1):.3f}; cos to E {x[:,1].mean():.3f}')
