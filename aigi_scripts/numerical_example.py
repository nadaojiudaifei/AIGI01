#!/usr/bin/env python3
"""Scalar arithmetic for the algorithm guide. NOT a trained image reconstruction."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import torch
from aigi.math_ops import importance,quantization_scale,local_snr,time_field,guidance_weights,guided_velocity,waterfill_target
s=torch.tensor(.6);w=importance(s);m=quantization_scale(torch.tensor(2.),w,torch.tensor(0.))
ybar=torch.tensor(2.8)/m
mu,sigma=.2,.3
n=torch.round(ybar-mu);qhat=n+mu+.02
snr=torch.tensor((mu**2+sigma**2)*12)
t=time_field(snr,w,0.,.5)
zp,zy=guidance_weights(t,w,snr)
v=guided_velocity(torch.tensor(.1),torch.tensor(.3),torch.tensor(.4),zp,zy)
z=.8-t*v+.05
ws=torch.tensor([.1,.424,.829,1.]).reshape(1,1,2,2)
cs=torch.tensor([.2,.5,1.2,.8]).reshape(1,1,2,2)
r=waterfill_target(ws,cs,torch.tensor([2.]))
print(json.dumps(dict(note='Illustrative scalar inputs; not neural-network predictions or compression results',
                     S=float(s),w=float(w),m=float(m),ybar=float(ybar),symbol=int(n),qhat=float(qhat),
                     snr=float(snr),t=float(t),zeta_p=float(zp),zeta_y=float(zy),velocity=float(v),
                     generated_latent=float(z),waterfill_bits=r.flatten().tolist(),budget=float(r.sum())),indent=2))
