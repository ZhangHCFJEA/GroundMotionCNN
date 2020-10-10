# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 02:06:41 2020

@author: ZhangHC
"""

from keras.models import load_model
import os
#from obspy.core import read, UTCDateTime
import h5py
#from obspy.clients.fdsn import Client
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import sys
sys.path.append('H:/python')
from util import save_h5, MMI_Worden2012

path = 'C:/Users/ZhangHC'
model = 'model_NCEDC_3s-4CNN-0.09.h5'
mode2 = 'model_SCEDC_3s-4CNN-0.12.h5'
m = load_model(os.path.join(path, model))
m2 = load_model(os.path.join(path, mode2))

dataPath = 'H:\MLData\groundMotion\Parkfield'
file = 'NCEDC_wave4.hdf5'
f = h5py.File(os.path.join(dataPath, file), 'r')

x, y, nam, evt, net = f['velData'], f['MMI_PGA'], f['station'], f['eventID'], f['network']
pga, dist, mag, pd = f['PGA_BSSA14'], f['epicentral distance'], f['Mag'], f['Pd']

_train_x, test_x, _train_Y, test_Y, _train_pga, test_pga, _train_dist, test_dist, _train_mag, test_mag = train_test_split(np.array(x[:, :, :]), 
                                                                                                                          np.array(y), np.array(pga), np.array(dist), np.array(mag),
                                                                                                                          test_size=0.25, random_state=7)

index1 = [-3.463, 0.729, -1.374] #log(Pd)=a+bM+clog(R)

test_x = test_x[:, :, :int(3*100)]
test_x = np.array(test_x).transpose(0, 2, 1)
test_X = test_x/np.std(test_x)

pred_y = m.predict(test_X) #PK
res = [] 
for n in range(len(test_Y)):
    res.append((test_Y[n]-pred_y[n].tolist()).tolist())
mn1, sd1 = np.mean(res), np.std(res)

fig = plt.figure(2, figsize=(7, 7), dpi=60)
plt.scatter(test_Y, pred_y, s=20, marker='o', facecolor='b', edgecolor='w', label='N='+str(len(y)))
plt.plot([-3, 7], [-3, 7], color='orange', linestyle= '--', label='1:1')
plt.plot([-3, 6], [-2, 7], 'g--', label='±1')
plt.plot([-2, 7], [-3, 6], 'g--')
plt.plot([-2.5, 7], [-3, 6.5], 'c--', label='±0.5')
plt.plot([-3, 6.5], [-2.5, 7], 'c--')
plt.xlim(-2, 7)
plt.ylim(-2, 7)
plt.xlabel('Record') 
plt.ylabel('Predict')
plt.grid()
plt.legend()


#
figSavePath = 'G:/Work Report/论文/CNN/figures'
fig = plt.figure(3, figsize=(9, 7), dpi=200)
ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2) 
ax2.scatter(np.arange(len(res)), res, marker='o', edgecolors='black', s=20,
            facecolor='orange', label='N='+str(len(res)))
ax2.plot([0, len(test_Y)], [sd1+mn1, sd1+mn1], 'r--', linewidth=3)
ax2.plot([0, len(test_Y)], [-sd1+mn1, -sd1+mn1], 'r--', linewidth=3)
ax2.plot([0, len(test_Y)], [mn1, mn1], color='green', linestyle='--', linewidth=3)
ax2.text(-2, 2.15, '(a)', fontsize=14)
#ax2.legend(fontsize=12, shadow=True)
ax2.set_xlabel('No.', fontsize=12)
ax2.set_ylabel('MMI residual (observe-predict)', fontsize=12)
ax2.set_ylim([-2, 2])
n, bins, patches = ax3.hist(np.array(res), facecolor="blue", edgecolor="black", alpha=0.6, 
                            bins=list(np.arange(-2, 2, 0.2)), orientation='horizontal')    

y = norm.pdf(bins, mn1, sd1)
ax3.plot(y*max(n), bins, 'r--')
ax3.text(0.3*max(n), 1.0, str(round(mn1, 3))+'±'+str(round(sd1, 2)), fontsize=12)
ax3.set_ylim([-2, 2])
plt.tight_layout()
plt.savefig(os.path.join(figSavePath, 'pk2pk.png'), dpi=200)
plt.savefig(os.path.join(figSavePath, 'pk2pk.eps'), dpi=200)

figSavePath = 'H:/python/CNN_figure'
np.savetxt(os.path.join(figSavePath, 'pk2pk.txt'), res)


###########
pred_y2 = m2.predict(test_X) #PK
res2 = [] 
for n in range(len(test_Y)):
    res2.append((test_Y[n]-pred_y2[n].tolist()).tolist())
mn2, sd2 = np.mean(res2), np.std(res2)

#
fig = plt.figure(4, figsize=(9, 7), dpi=200)
ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2) 
ax2.scatter(np.arange(len(res2)), res2, marker='o', edgecolors='black', s=20,
            facecolor='orange', label='N='+str(len(res)))
ax2.plot([0, len(test_Y)], [sd2+mn2, sd2+mn2], 'r--', linewidth=3)
ax2.plot([0, len(test_Y)], [-sd2+mn2, -sd2+mn2], 'r--', linewidth=3)
ax2.plot([0, len(test_Y)], [mn2, mn2], color='green', linestyle='--', linewidth=3)
ax2.text(-2, 2.15, '(a)', fontsize=14)
#ax2.legend(fontsize=12, shadow=True)
ax2.set_xlabel('No.', fontsize=12)
ax2.set_ylabel('MMI residual (observe-predict)', fontsize=12)
ax2.set_ylim([-2, 2])
n, bins, patches = ax3.hist(np.array(res2), facecolor="blue", edgecolor="black", alpha=0.6, 
                            bins=list(np.arange(-2, 2, 0.2)), orientation='horizontal')    

y = norm.pdf(bins, mn2, sd2)
ax3.plot(y*max(n), bins, 'r--')
ax3.text(0.3*max(n), 1.0, str(round(mn2, 3))+'±'+str(round(sd2, 2)), fontsize=12)
ax3.set_ylim([-2, 2])
plt.tight_layout()
plt.savefig(os.path.join(figSavePath, 'la2pk.png'), dpi=200)
plt.savefig(os.path.join(figSavePath, 'la2pk.eps'), dpi=200)

figSavePath = 'H:/python/CNN_figure'
np.savetxt(os.path.join(figSavePath, 'la2pk.txt'), res2)


###################
atten = []
for n in range(len(pga)):
    atten.append(MMI_Worden2012(pga[n]*100, dist[n], mag[n], 'PGA'))

res3 = [] 
for n in range(len(test_Y)):
    res3.append((test_Y[n]-atten[n].tolist()).tolist())
mn3, sd3 = np.mean(res3), np.std(res3)

#
fig = plt.figure(5, figsize=(9, 7), dpi=200)
ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2) 
ax2.scatter(np.arange(len(res3)), res3, marker='o', edgecolors='black', s=20,
            facecolor='orange', label='N='+str(len(res)))
ax2.plot([0, len(test_Y)], [sd3+mn3, sd3+mn3], 'r--', linewidth=3)
ax2.plot([0, len(test_Y)], [-sd3+mn3, -sd3+mn3], 'r--', linewidth=3)
ax2.plot([0, len(test_Y)], [mn3, mn3], color='green', linestyle='--', linewidth=3)
ax2.text(-2, 4.15, '(a)', fontsize=14)
#ax2.legend(fontsize=12, shadow=True)
ax2.set_xlabel('No.', fontsize=12)
ax2.set_ylabel('MMI residual (observe-predict)', fontsize=12)
ax2.set_ylim([-4, 4])
n, bins, patches = ax3.hist(np.array(res3), facecolor="blue", edgecolor="black", alpha=0.6, 
                            bins=list(np.arange(-4, 4, 0.2)), orientation='horizontal')    

y = norm.pdf(bins, mn3, sd3)
y = y/max(y)
ax3.plot(y*max(n), bins, 'r--')
ax3.text(0.3*max(n), 3.4, str(round(mn3, 3))+'±'+str(round(sd3, 2)), fontsize=12)
ax3.set_ylim([-4, 4])
plt.tight_layout()
plt.savefig(os.path.join(figSavePath, 'pk_atten.png'), dpi=200)
plt.savefig(os.path.join(figSavePath, 'pk_atten.eps'), dpi=200)

figSavePath = 'H:/python/CNN_figure'
np.savetxt(os.path.join(figSavePath, 'pk_atten.txt'), res3)





