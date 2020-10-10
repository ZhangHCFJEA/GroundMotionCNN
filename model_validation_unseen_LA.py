# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 01:03:19 2020

@author: ZhangHC
"""

# from tensorflow_core.python.keras.models import load_model
from keras.models import load_model
import os
import h5py
# from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

path = 'C:/Users/ZhangHC'
#model1 = 'model_NCEDC_3s-4CNN-0.09.h5'
#model2 = 'model_NCEDC_5s-4CNN-0.09.h5'
model1 = 'model_SCEDC_3s-4CNN-0.12.h5'
# model2 = 'model_SCEDC_5s-4CNN-0.11.h5'

m1 = load_model(os.path.join(path, model1))
#m2 = load_model(os.path.join(path, model2))

file1 = 'SCEDC_new.hdf5'
dataPath = 'H:/MLData/groundMotion/SCEDC/new'
f1 = h5py.File(os.path.join(dataPath, file1), 'r')

# Test 1
X = f1['velData']
Y = f1['MMI_PGA']

#
test_X1 = X[:, :, :int(3*100.)]
test_Y = Y
test_X1 = np.array(test_X1).transpose(0, 2, 1)
test_X1 = test_X1/np.std(test_X1)
pred_y1 = m1.predict(test_X1)

#
gap1 = []
for n in range(len(test_Y)):
    gap1.append(test_Y[n]-pred_y1[n])
mean1 = np.mean(gap1)
sd1 = np.std(gap1)
print(mean1, sd1)


figSavePath = 'H:/python/CNN_figure'
#
fig = plt.figure(1, figsize=(7, 7), dpi=60)
plt.hist([test_Y, np.reshape(pred_y1, len(pred_y1), )], bins=[-2, -1,  0,  1,  2,  3,  4,  5,  6], label=['record', 'predict'], alpha=0.7)    
plt.legend()
plt.grid()
plt.xlabel('MMI') 
plt.ylabel('Counts')
plt.title('Histogram of MMI')
plt.show()
plt.savefig(os.path.join(figSavePath, 'freshTest_SCEDC.png'), dpi=60)

fig = plt.figure(2, figsize=(7, 7), dpi=60)
plt.scatter(test_Y, pred_y1, s=20, marker='o', facecolor='b', edgecolor='w', label='N='+str(len(test_Y)))
# plt.scatter(test_Y, pred_y1, s=20, marker='o', facecolor='b', edgecolor='w', label='N='+str(len(test_Y)))
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
plt.savefig(os.path.join(figSavePath, 'freshTestDis_SCEDC.png'), dpi=60)


fig = plt.figure(3, figsize=(7, 7), dpi=60)
ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2) 
ax2.scatter(np.arange(len(gap1)), gap1, marker='o', edgecolors='black', s=20,
            facecolor='orange', label='N='+str(len(gap1)))

ax2.plot([0, len(test_Y)], [sd1+mean1, sd1+mean1], 'r--', linewidth=3)
ax2.plot([0, len(test_Y)], [-sd1+mean1, -sd1+mean1], 'r--', linewidth=3)
ax2.plot([0, len(test_Y)], [mean1, mean1], color='green', linestyle='--', linewidth=3)
ax2.text(0, 2.2, '(b)', fontsize=12)
# ax2.legend(fontsize=12, shadow=True)
ax2.set_xlabel('No.', fontsize=12)
ax2.set_ylabel('MMI residual (observe-predict)', fontsize=12)
ax2.set_ylim([-2, 2])
n, bins, patches = ax3.hist(np.array(gap1), facecolor="blue", edgecolor="black", alpha=0.6, 
                            bins=list(np.arange(-2, 2, 0.2)), orientation='horizontal')    

y = norm.pdf(bins, mean1, sd1)
y = y/max(y)
ax3.plot(y*max(n), bins, 'r--')
ax3.text(0.3*max(n), 1.0, str(round(mean1, 3))+'±'+str(round(sd1, 2)), fontsize=12)
ax3.set_ylim([-2, 2])
plt.tight_layout()
plt.savefig(os.path.join(figSavePath, 'freshTestRes_SCEDC.png'), dpi=200)
plt.savefig(os.path.join(figSavePath, 'freshTestRes_SCEDC.eps'), dpi=200)



np.savetxt(os.path.join(figSavePath, 'fresh_SCEDC.txt'), gap1)



