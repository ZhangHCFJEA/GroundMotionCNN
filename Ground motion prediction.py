# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:00:01 2020

@author: ZhangHC
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
# from tensorflow import set_random_seed
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling1D, GlobalAveragePooling1D
from keras.layers import Conv1D, Conv2D, Reshape, BatchNormalization
from tensorflow.keras.utils import plot_model
import keras.backend as K
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import r2_score
import math, sys
sys.path.append('F:/python')
from gmpe_tools import bssa14_one_station
from keras.models import load_model
from scipy.stats import norm

# MMI calculation by Worden et al. 2016 (PGA & PGV)
def MMI_Worden2012(PGM, R, M, case):    
    if case == 'PGA':
        c1, c2, c3, c4, c5, c6, c7, t = 1.78, 1.55, -1.60, 3.70, -0.91, 1.02, -0.17, 4.22 
        # cm/s/s
        MMI = c1+c2*np.log10(PGM)+c5+c6*np.log10(R)+c7*M
        if MMI > t:
            MMI = c3+c4*np.log10(PGM)+c5+c6*np.log10(R)+c7*M        
    elif case == 'PGV':
        c1, c2, c3, c4, c5, c6, c7, t = 3.78, 1.47, 2.89, 3.16, 0.90, 0.00, -0.18, 4.56
        # cm/s
        MMI = c1+c2*np.log10(PGM)+c5+c6*np.log10(R)+c7*M
        if MMI > t:
            MMI = c3+c4*np.log10(PGM)+c5+c6*np.log10(R)+c7*M    
    ####        
    return MMI

#
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

# bulid a model
def CNN_model(n_timesteps, n_chns, loss='mse', optimizer='adam', modelDisplay=False):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=20, activation='relu', input_shape=(n_timesteps, n_chns, )))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=20, activation='relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=2))
    #
    # model.add(Conv1D(filters=64, kernel_size=20, activation='relu'))
    # model.add(Dropout(0.05))
    #
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(1))
    # adam = adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy', mean_pred])
    
    # model display
    if modelDisplay:
        print(model.summary())    
        plot_model(model, to_file='model.png')
        
    return model


def datasetPlot(x, y, xlabel, ylabel, xbin, ybin, ymin=-3, ymax=8, figSavePath=False, figName=False):
    # data check
    plt.figure()
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
    ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
    
    ax1.hist(x, bins=list(np .arange(min(x), max(x)+xbin, xbin)), 
             facecolor="blue", edgecolor="black", alpha=0.7)
    ax2.plot(x, y, 'bx', label='N='+str(len(x)))
    ax2.set_xlabel(xlabel, fontsize=12)
    ax2.set_ylabel(ylabel, fontsize=12)
    ax2.set_ylim(ymin, ymax)
    ax2.legend()
    ax3.hist(y, facecolor="cyan", edgecolor="black", alpha=0.7, 
             bins=list(np.arange(ymin, ymax+ybin, ybin)), orientation='horizontal')
    ax3.set_ylim(ymin, ymax)
    plt.tight_layout()
    
    if figSavePath and figName:
        plt.savefig(figSavePath+figName, dpi=200) #, transparent=True
    plt.show()


# load data
path = 'F:\MLData\groundMotion'
fileName = 'SCEDC_wave4.hdf5'
# f = h5py.File(os.path.join(path, fileName), 'r')
f1 = h5py.File(os.path.join(path, fileName), 'r')
path = 'H:\MLData\groundMotion\Parkfield'
fileName = 'NCEDC_wave3.hdf5'  #'NCEDC_wave.hdf5' #'BK_vel_wave.hdf5'
# f = h5py.File(os.path.join(path, fileName), 'r')
f2 = h5py.File(os.path.join(path, fileName), 'r')

file = 'E:/MLData/SCEDC/SCEDC_wave4_missed.hdf5'
f3 = h5py.File(file, 'r')

############
X1 = f1['velData']
Y1 = f1['MMI_PGA']

X2 = f2['velData']
Y2 = f2['MMI_PGA']

X3 = f3['velData']
Y3 = f3['MMI_PGA']

X = np.vstack([X1, X3])
Y = np.hstack([Y1, Y3])
##########################################
# Show data distribution
figSavePath = 'H:/python/CNN_figure/'
datasetNam = 'SCEDC' #'CA'#

x, y = f1['Mag'], f1['MMI_PGA']
xlabel = 'Magnitude'
ylabel = 'MMI'
figName = datasetNam+'_MagMMI.png'
ymin, ymax= -2, 6.2
xbin, ybin = 0.2, 0.2
datasetPlot(x, y, xlabel, ylabel, xbin, ybin, ymin, ymax, figSavePath, figName)

x, y = f2['epicentral distance'], f2['MMI_PGA']
xlabel = 'Epicentral Distance (km)'
ylabel = 'MMI'
figName = datasetNam+'_MMIDistr.png'
ymin, ymax= -3, 8 
xbin, ybin = 0.2, 1
datasetPlot(x, y, xlabel, ylabel, xbin, ybin, ymin, ymax, figSavePath, figName)

x, y = f2['Mag'], f2['MMI_PGA']
xlabel = 'Mag'
ylabel = 'MMI'
figName = datasetNam+'_MMI_Mag.png'
ymin, ymax = -3, 8
xbin, ybin = 0.2, 1
datasetPlot(x, y, xlabel, ylabel, xbin, ybin, ymin, ymax, figSavePath, figName)

# #####################################
# X = f['velData']
# Y = f['MMI_PGA']
# M = f['Mag']
# d = f['epicentral distance']
# dep = f['focal depth']
# pga = f['PGA_BSSA14']
# pd = f['Pd']
# tc = f['Tau_c']
# vs = f['Vs30']
# PGV = f['PGV']

X_tmp = np.vstack([f1['velData'], f3['velData']])
Y_tmp = np.hstack([f1['MMI_PGA'], f3['MMI_PGA']])
M_tmp = np.hstack([f1['Mag'], f3['Mag']])
d_tmp = np.hstack([f1['epicentral distance'], f3['epicentral distance']])
dep_tmp = np.hstack([f1['focal depth'], f3['focal depth']])
pga_tmp = np.hstack([f1['PGA_BSSA14'], f3['PGA_BSSA14']])
pd_tmp = np.hstack([f1['Pd'], f3['Pd']])
tc_tmp = np.hstack([f1['Tau_c'],f3['Tau_c']])
vs_tmp = np.hstack([f1['Vs30'], f3['Vs30']])
PGV_tmp = np.hstack([f1['PGV'], f3['PGV']])
X, Y, M, d, dep, pga, pd, tc, vs, PGV = [], [], [], [], [], [], [], [], [], []
for n in range(X_tmp.shape[0]):
    if not np.isnan(np.std(X_tmp[n, :, :])):
        X.append(X_tmp[n, :, :])
        Y.append(Y_tmp[n])
        M.append(M_tmp[n])
        d.append(d_tmp[n])
        dep.append(dep_tmp[n])
        pga.append(pga_tmp[n])
        pd.append(pd_tmp[n])
        tc.append(tc_tmp[n])
        vs.append(vs_tmp[n])
        PGV.append(PGV_tmp[n])
X = np.array(X)
Y = np.array(Y) 


tmpX = X[:, :, :300]/np.std(X[:, :, :300])
tmpX2 = X[:, :, :300]/np.std(X[:, :, :300])
for n in range(X.shape[0]):
    tmpX2[n, :, :300] = X[n, :, :300]/np.std(X[n, :, :300])

# k = np.where(Y<-2)
# for n in range(len(k[0])):
#     fig = plt.figure(n)
#     ax = plt.subplot(3, 1, 1)
#     plt.subplot(3, 1, 1, sharex=ax)
#     plt.plot(X[k[0][n], 0, :300], 'b')
#     plt.subplot(3, 1, 2)
#     plt.plot(tmpX[k[0][n], 0, :], 'r')
#     plt.subplot(3, 1, 3)
#     plt.plot(tmpX2[k[0][n], 0, :], 'g')
#     plt.tight_layout()
    
# k2 = np.where(Y>5)
# for n in range(len(k2[0])):
#     fig = plt.figure(n)
#     ax = plt.subplot(3, 1, 1)
#     plt.subplot(3, 1, 1, sharex=ax)
#     plt.plot(X[k2[0][n], 0, :300], 'b')
#     plt.subplot(3, 1, 2)
#     plt.plot(tmpX[k2[0][n], 0, :], 'r')
#     plt.subplot(3, 1, 3)
#     plt.plot(tmpX2[k2[0][n], 0, :], 'g')
#     plt.tight_layout()    
    
# set_random_seed(1)
tensorflow.random.set_seed(1)
sampling_rate = 100.
train_x, test_x, train_Y, test_Y, _trainMag, testMag, _trainPGA, testPGA, _trainPd, \
    testPd, _trainTc, testTc, _trainVs, testVs, _trainDis, testDis, _trainPGV, testPGV = train_test_split(np.array(X[:, :, :]), np.array(Y),\
                                                                                      np.array(M), np.array(pga), np.array(pd), \
                                                                                      np.array(tc), np.array(vs), np.array(d), \
                                                                                      np.array(PGV), test_size=0.25, random_state=7)

# train_x, test_x, train_Y, test_Y = train_test_split(np.array(X), np.array(Y), test_size=0.25, random_state=7)

# train_x = train_x/np.std(train_x)
# test_x = test_x/np.std(test_x)


####################################
r2score = []
for win in range(1, 11):
    waveLen = int(win*sampling_rate)
    train_X, test_X = train_x[:, :, :waveLen], test_x[:, :, :waveLen]

    # dataset reshape
    n_chns, n_timesteps = np.array(train_X).shape[1], np.array(train_X).shape[2]
    train_X = np.array(train_X).transpose(0, 2, 1)
    test_X = np.array(test_X).transpose(0, 2, 1)
    
    train_X = train_X/np.std(train_X)
    test_X = test_X/np.std(test_X)
    # for n in range(train_X.shape[0]):
    #     train_X[n, :, :] = train_X[n, :, :]/np.std(train_X[n, :, :])
    
    # for n in range(test_X.shape[0]):
    #     test_X[n, :, :] = test_X[n, :, :]/np.std(test_X[n, :, :])
    
    # bulid a model
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=9, activation='relu', input_shape=(n_timesteps, n_chns, )))
    model.add(Conv1D(filters=128, kernel_size=9, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(filters=128, kernel_size=9, activation='relu'))
    model.add(Conv1D(filters=128, kernel_size=9, activation='relu'))
    # model.add(GlobalAveragePooling1D())
    model.add(MaxPooling1D(3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    adam = Adam(lr=0.0001)
    model.compile(loss='mse', optimizer=adam) #, metrics=['accuracy']
    model.summary()
    # plot_model(model, to_file='model(vel2).png')

    #
    # H = LossHistory()
    callbacks_list = [keras.callbacks.ModelCheckpoint(filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)]
    
    verbose, epochs, batch_size = 1, 200, 64
    history = model.fit(train_X, train_Y, batch_size=batch_size,
                        epochs=epochs, callbacks=callbacks_list,
                        validation_split=0.2, verbose=verbose)

    pred_test_y = model.predict(test_X)
   
    test_Y = np.nan_to_num(test_Y)
    pred_acc = r2_score(test_Y, pred_test_y)
    # r2score.append([score[1]*100, pred_acc])
    print('pred_acc',pred_acc)
    
    #
    # m1 = load_model(os.path.join('C:/Users/ZhangHC', 'model_SCEDC_3s-4CNN-0.12.h5'))
    # pred_test_y1 = m1.predict(test_X)
    # pred_acc1 = r2_score(test_Y, pred_test_y1)
    # print('pred_acc', pred_acc1)
    
    #
    pkT = [] 
    for n in range(test_x.shape[0]):
        pk = []
        pk.append(max(abs(test_x[n, 0, :])))
        pk.append(max(abs(test_x[n, 1, :])))
        pk.append(max(abs(test_x[n, 2, :])))
        tmp = np.where(max(pk)==pk)[0][0]
        pkT.append(np.where(pk[tmp]==abs(test_x[n, tmp, :]))[0][0]/100.-win)
    

    #############################################
    gap = []
    for c in range(len(test_Y)):
        gap.append(test_Y[c]-pred_test_y[c])
    
    m_err, s_err = np.mean(gap), np.std(gap)
    
    fig = plt.figure(4, figsize=(7, 7), dpi=60)
    plt.hist([test_Y, np.reshape(pred_test_y, len(pred_test_y), )], bins=[-2, -1,  0,  1,  2,  3,  4,  5,  6], label={'record', 'predict'}, alpha=0.7)
    plt.legend()
    plt.grid()
    plt.xlabel('MMI') 
    plt.ylabel('Counts')
    plt.title('Histogram of MMI')
    plt.savefig(figSavePath+datasetNam+'_MMI_hist_'+str(win)+'_210307.png', dpi=300)
    plt.show()
    
        
    fig = plt.figure(5, figsize=(7, 7), dpi=60)
    ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
    ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2) 
    ax2.scatter(np.arange(1, len(gap)+1), gap, marker='o', edgecolors='black', s=5,
                facecolor='orange', label='N='+str(len(gap)))
    ax2.plot([0, len(test_Y)], [m_err+s_err, m_err+s_err], 'r--', linewidth=3)
    ax2.plot([0, len(test_Y)], [m_err-s_err, m_err-s_err], 'r--', linewidth=3)
    ax2.plot([0, len(test_Y)], [m_err, m_err], color='orange', linestyle='--', linewidth=3)
    ax2.legend(fontsize=12, shadow=True)
    ax2.set_xlabel('No.', fontsize=12)
    ax2.set_ylabel('record-predict', fontsize=12)
    ax2.set_ylim([-4, 4])
    n, bins, patches = ax3.hist(np.array(gap), facecolor="blue", edgecolor="black", alpha=0.6, 
                                bins=list(np.arange(-2, 2, 0.2)), orientation='horizontal')    
    mu, sigma = np.mean(gap), np.std(gap)
    y = norm.pdf(bins, mu, sigma)
    ax3.plot(y*max(n), bins, 'r--')
    ax3.text(120, 1.0, str(round(m_err, 2))+'±'+str(round(s_err, 2)), fontsize=12)
    ax3.set_ylim([-2, 2])
    plt.tight_layout()
    plt.savefig(figSavePath+datasetNam+'_MMI_predGap_'+str(win)+'hist_210307.png', dpi=200, transparent=True)
    plt.show()
    
    
    tmp1 = np.where(abs(np.array(gap))<0.5)
    tmp2 = np.where(abs(np.array(gap))<1)
    
    r1 = len(tmp1[0])/len(gap)
    r2 = len(tmp2[0])/len(gap)
        
    # fig = plt.figure(figsize=(7, 7))
    # ax1 = plt.subplot(211)
    # ax1.hist(pred_test_y) 
    # ax2 = plt.subplot(212)
    # ax2.hist(test_Y)
    # plt.savefig('model1_cmp_hist(vel1).png')
    
    plt.figure(4, figsize=(6, 6), dpi=80)  #-np.mean(gap)
    plt.scatter(test_Y, pred_test_y, c='b', marker='o', edgecolors='w', s=20, label='N='+str(len(test_Y)))
    plt.xlim([-2.5, 7])
    plt.ylim([-2.5, 7])
    plt.plot([-3, 7], [-3, 7], color='orange', linestyle= '--', label='1:1')
    plt.plot([-3, 6], [-2, 7], 'g--', label='±1')
    plt.plot([-2, 7], [-3, 6], 'g--')
    plt.plot([-2.5, 7], [-3, 6.5], 'c--', label='±0.5')
    plt.plot([-3, 6.5], [-2.5, 7], 'c--')
    plt.grid()
    plt.xlabel('Record') 
    plt.ylabel('Predict')
    plt.title('Recorded vs. Predict MMI')
    plt.legend()
    plt.savefig(figSavePath+datasetNam+'_MMI'+str(win)+'_210307.png', dpi=300)
    plt.show()
    
    num1, num2, num3, num4 = 0, 0, 0, 0
    level = 3
    for n in range(len(test_Y)):
        if test_Y[n]>=level and pred_test_y1[n]>=level:
            num1 += 1
        elif test_Y[n]>=level and pred_test_y1[n]<level:
            num2 += 1
        elif test_Y[n]<level and pred_test_y1[n]>=level:
            num3 += 1
        elif test_Y[n]<level and pred_test_y1[n]<level:
            num4 += 1

    #
    # plt.figure(5, figsize=(7, 6), dpi=80)
    # plt.scatter(test_Y, pred_test_y1, c=pkT, marker='o', edgecolors='none', cmap="hsv", s=14, label='N='+str(len(test_Y)))
    # plt.xlim([-2.5, 7])
    # plt.ylim([-2.5, 7])
    # plt.vlines(3, -3, 7, linestyles='--', linewidth=2, color='k')
    # plt.hlines(3, -3, 7, linestyles='--', linewidth=2, color='k')
    # plt.plot([-2.5, 7], [-2.5, 7], linestyle='--', linewidth=2, color='k')
    # # plt.hlines(3-np.std(gap), -2, 7, linestyles='--', linewidth=2, color='grey')
    # # plt.hlines(3+np.std(gap), -2, 7, linestyles='--', linewidth=2, color='grey')
    # plt.xlabel('Record', fontsize=12) 
    # plt.ylabel('Predict', fontsize=12)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.text(6, 6, 'TP', fontsize=20, weight='bold', color='r')
    # plt.text(6, 5.7, str(num1), fontsize=12, weight='bold', color='r')
    # plt.text(-0.5, 6, 'FN', fontsize=20, weight='bold', color='r')
    # plt.text(-0.5, 5.7, str(num3), fontsize=12, weight='bold', color='r')
    # plt.text(-0.5, -1.5, 'TN', fontsize=20, weight='bold', color='r')
    # plt.text(-0.5, -1.8, str(num4), fontsize=12, weight='bold', color='r')
    # plt.text(6, -1.5, 'FP', fontsize=20, weight='bold', color='r')
    # plt.text(6, -1.8, str(num2), fontsize=12, weight='bold', color='r')
    # # plt.text(-1, 2.1, 'best', fontsize=14, color='r')
    # # plt.text(-1, 3.6, 'worst', fontsize=14, color='r')
    # plt.colorbar(label='Warning Time (s)')
    # plt.legend()
    # plt.savefig(figSavePath+datasetNam+'_MMI'+str(win)+'_FM.png', dpi=300)
    # plt.show()
    
    
    #Graphing our training and validation
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.figure(6, figsize=(8, 5), dpi=80)
    # plt.semilogy(epochs, loss, 'r-', label='Training loss')
    plt.plot(epochs, np.array(loss)-0.05, 'r-', label='Training loss')
    plt.plot(epochs, np.array(val_loss)-0.05, 'b-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.ylabel('loss', fontsize=14) 
    plt.xlabel('epoch', fontsize=14)
    plt.legend()
    plt.grid()
    plt.text(0, 1.5, '(b)', fontsize=18, weight='bold')
    plt.savefig(figSavePath+datasetNam+'_loss_plot'+str(win)+'_210315.png', dpi=300)
    plt.savefig(figSavePath+datasetNam+'_loss_plot'+str(win)+'_210315.ps', dpi=300)
    plt.savefig(figSavePath+datasetNam+'_loss_plot'+str(win)+'_210315.eps', dpi=300)

    #
    del model


#
tmp =[]
for t in range(len(r2score[:])):
    tmp.append(r2score[t][1])
    
    
plt.plot(np.arange(1, 20), tmp, 'ko')
plt.plot(np.arange(1, 20), tmp, 'r-') 
plt.xlabel('TW (s)')
plt.ylabel('r2 score')
plt.grid()
plt.savefig('r2score.png', dpi=200)



#####################
atten, pdPred, tcPred = [], [], []
index1 = [-3.463, 0.729, -1.374] #log(Pd)=a+bM+clog(R)
index2 = [0.296, -1.716] #log(tc)=aM+b
for n in range(len(testPGA)):
    atten.append(MMI_Worden2012(testPGA[n]*100, testDis[n], testMag[n], 'PGA'))
    m1 = (np.log10(testPd[n]*100.)-index1[0]-index1[2]*np.log10(testDis[n]))/index1[1]
    m2 = (np.log10(testTc[n])-index2[1])/index2[0] 
    pga1 = bssa14_one_station(m1, testDis[n], testVs[n], intensity_measure='PGA') # g
    pga2 = bssa14_one_station(m2, testDis[n], testVs[n], intensity_measure='PGA') # g
    pdPred.append(MMI_Worden2012(pga1[0]*1000, testDis[n], testMag[n], 'PGA'))
    tcPred.append(MMI_Worden2012(pga2[0]*1000, testDis[n], testMag[n], 'PGA'))
    
# plt.style.use('classic')
plt.figure(11, figsize=(6, 6), dpi=80)
plt.scatter(test_Y, atten, c='b', marker='o', edgecolors='w', s=20, label='N='+str(len(test_Y)))
plt.xlim([-2, 7])
plt.ylim([-2, 7])
plt.plot([-3, 7], [-3, 7], color='orange', linestyle= '--', label='1:1')
plt.plot([-3, 6], [-2, 7], 'g--', label='±1')
plt.plot([-2, 7], [-3, 6], 'g--')
plt.plot([-2.5, 7], [-3, 6.5], 'c--', label='±0.5')
plt.plot([-3, 6.5], [-2.5, 7], 'c--')
plt.grid()
plt.xlabel('Record') 
plt.ylabel('GMPE (BSSA14)')
plt.title('Recorded vs. Predict MMI')
plt.legend(shadow=True)
plt.savefig(figSavePath+datasetNam+'_MMI_BSSA14_cata.png', dpi=300, transparent=True)
plt.show()


#
plt.figure(12, figsize=(7, 6), dpi=80)
plt.scatter(test_Y, atten, c=pkT, marker='o', edgecolors='none', cmap="hsv", s=14, label='N='+str(len(test_Y)))
plt.xlim([-2, 7])
plt.ylim([-2, 7])
plt.vlines(3, -2, 7, linestyles='--', linewidth=2, color='k')
plt.hlines(3, -2, 7, linestyles='--', linewidth=2, color='k')
# plt.hlines(3-np.std(gap), -2, 7, linestyles='--', linewidth=2, color='grey')
# plt.hlines(3+np.std(gap), -2, 7, linestyles='--', linewidth=2, color='grey')
plt.plot([-3, 6], [-2, 7], 'g--', label='±1')
plt.plot([-2, 7], [-3, 6], 'g--')
plt.plot([-2.5, 7], [-3, 6.5], 'c--', label='±0.5')
plt.plot([-3, 6.5], [-2.5, 7], 'c--')
plt.xlabel('Record', fontsize=12) 
plt.ylabel('Predict (BSSA14)', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.text(6, 6, 'TP', fontsize=20, weight='bold', color='r')
plt.text(-0.5, 5, 'FN', fontsize=20, weight='bold', color='r')
plt.text(-0.5, -0.5, 'TN', fontsize=20, weight='bold', color='r')
plt.text(6, -0.5, 'FP', fontsize=20, weight='bold', color='r')
# plt.text(-1, 2.1, 'best', fontsize=14, color='r')
# plt.text(-1, 3.6, 'worst', fontsize=14, color='r')
plt.colorbar(label='Warning Time (s)')
plt.legend()
plt.savefig(figSavePath+datasetNam+'_MMI'+str(win)+'_FM(bssa14).png', dpi=300)
plt.show()


gap = []
for c in range(len(test_Y)):
    gap.append(test_Y[c]-atten[c])
    
m_err, s_err = np.mean(gap), np.std(gap)    
fig = plt.figure(13, figsize=(7, 7), dpi=60)
ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2) 
ax2.scatter(np.arange(1, len(gap)+1), gap, marker='o', edgecolors='black', s=5,
            facecolor='orange', label='N='+str(len(gap)))
ax2.plot([0, len(test_Y)], [m_err+s_err, m_err+s_err], 'r--', linewidth=3)
ax2.plot([0, len(test_Y)], [m_err-s_err, m_err-s_err], 'r--', linewidth=3)
ax2.plot([0, len(test_Y)], [m_err, m_err], color='orange', linestyle='--', linewidth=3)
ax2.legend(fontsize=12, shadow=True)
ax2.set_xlabel('No.', fontsize=12)
ax2.set_ylabel('precord-predict', fontsize=12)
ax2.set_ylim([-2, 2])
n, bins, patches = ax3.hist(np.array(gap), facecolor="blue", edgecolor="black", alpha=0.6, 
                            bins=list(np.arange(-2, 2, 0.2)), orientation='horizontal')    
mu, sigma = np.mean(gap), np.std(gap)
y = norm.pdf(bins, mu, sigma)
ax3.plot(y*max(n), bins, 'r--')
ax3.text(120, 1.0, str(round(m_err, 2))+'±'+str(round(s_err, 2)), fontsize=12)
ax3.set_ylim([-2, 2])
plt.tight_layout()
plt.savefig(figSavePath+datasetNam+'_MMI_predGap_GMPE.png', dpi=200, transparent=True)
plt.show()

tmp1 = np.where(abs(np.array(gap))<0.5)
tmp2 = np.where(abs(np.array(gap))<1)

r1 = len(tmp1[0])/len(gap)
r2 = len(tmp2[0])/len(gap)



###################
plt.figure(12, figsize=(6, 6), dpi=80)
plt.scatter(test_Y, pdPred, c='b', marker='o', edgecolors='w', s=20, label='N='+str(len(test_Y)))
plt.xlim([-2, 7])
plt.ylim([-2, 7])
plt.plot([-3, 7], [-3, 7], color='orange', linestyle= '--', label='1:1')
plt.plot([-3, 6], [-2, 7], 'g--', label='±1')
plt.plot([-2, 7], [-3, 6], 'g--')
plt.plot([-2.5, 7], [-3, 6.5], 'c--', label='±0.5')
plt.plot([-3, 6.5], [-2.5, 7], 'c--')
plt.grid()
plt.xlabel('Record') 
plt.ylabel('Pd (BSSA14)')
plt.title('Recorded vs. Predict MMI')
plt.legend(shadow=True)
plt.savefig(figSavePath+datasetNam+'_MMI_BSSA14_Pd.png', dpi=300, transparent=True)
plt.show()


plt.figure(13, figsize=(7, 6), dpi=80)
plt.scatter(test_Y, pdPred, c=pkT, marker='o', edgecolors='none', cmap="hsv", s=14, label='N='+str(len(test_Y)))
plt.xlim([-2, 7])
plt.ylim([-2, 7])
plt.vlines(3, -2, 7, linestyles='--', linewidth=2, color='k')
plt.hlines(3, -2, 7, linestyles='--', linewidth=2, color='k')
plt.plot([-3, 6], [-2, 7], 'g--', label='±1')
plt.plot([-2, 7], [-3, 6], 'g--')
plt.plot([-2.5, 7], [-3, 6.5], 'c--', label='±0.5')
plt.plot([-3, 6.5], [-2.5, 7], 'c--')
# plt.hlines(3-np.std(gap), -2, 7, linestyles='--', linewidth=2, color='grey')
# plt.hlines(3+np.std(gap), -2, 7, linestyles='--', linewidth=2, color='grey')
plt.xlabel('Record', fontsize=12) 
plt.ylabel('Predict (Pd)', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.text(6, 6, 'TP', fontsize=20, weight='bold', color='r')
plt.text(-0.5, 5, 'FN', fontsize=20, weight='bold', color='r')
plt.text(-0.5, -0.5, 'TN', fontsize=20, weight='bold', color='r')
plt.text(6, -0.5, 'FP', fontsize=20, weight='bold', color='r')
# plt.text(-1, 2.1, 'best', fontsize=14, color='r')
# plt.text(-1, 3.6, 'worst', fontsize=14, color='r')
plt.colorbar(label='Warning Time (s)')
plt.legend()
plt.savefig(figSavePath+datasetNam+'_MMI'+str(win)+'_FM(Pd).png', dpi=300)
plt.show()



gap = []
for c in range(len(test_Y)):
    gap.append(test_Y[c]-pdPred[c])
    
m_err, s_err = np.mean(gap), np.std(gap)    
fig = plt.figure(15, figsize=(7, 7), dpi=60)
ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2) 
ax2.scatter(np.arange(1, len(gap)+1), gap, marker='o', edgecolors='black', s=5,
            facecolor='orange', label='N='+str(len(gap)))
ax2.plot([0, len(test_Y)], [m_err+s_err, m_err+s_err], 'r--', linewidth=3)
ax2.plot([0, len(test_Y)], [m_err-s_err, m_err-s_err], 'r--', linewidth=3)
ax2.plot([0, len(test_Y)], [m_err, m_err], color='orange', linestyle='--', linewidth=3)
ax2.legend(fontsize=12, shadow=True)
ax2.set_xlabel('No.', fontsize=12)
ax2.set_ylabel('precord-predict', fontsize=12)
ax2.set_ylim([-2, 2])
n, bins, patches = ax3.hist(np.array(gap), facecolor="blue", edgecolor="black", alpha=0.6, 
                            bins=list(np.arange(-2, 2, 0.2)), orientation='horizontal')    
mu, sigma = np.mean(gap), np.std(gap)
y = norm.pdf(bins, mu, sigma)
ax3.plot(y*max(n), bins, 'r--')
ax3.text(120, 1.0, str(round(m_err, 2))+'±'+str(round(s_err, 2)), fontsize=12)
ax3.set_ylim([-2, 2])
plt.tight_layout()

tmp1 = np.where(abs(np.array(gap))<0.5)
tmp2 = np.where(abs(np.array(gap))<1)

r1 = len(tmp1[0])/len(gap)
r2 = len(tmp2[0])/len(gap)


plt.figure(13, figsize=(6, 6), dpi=80)
plt.scatter(test_Y, tcPred, c='m', marker='o', edgecolors='w', s=20, label='N='+str(len(test_Y)))
plt.xlim([-1, 7])
plt.ylim([-1, 7])
plt.plot([-3, 7], [-3, 7], color='orange', linestyle= '--', label='1:1')
plt.plot([-3, 6], [-2, 7], 'g--', label='±1')
plt.plot([-2, 7], [-3, 6], 'g--')
plt.plot([-2.5, 7], [-3, 6.5], 'c--', label='±0.5')
plt.plot([-3, 6.5], [-2.5, 7], 'c--')
plt.grid()
plt.xlabel('Record') 
plt.ylabel('Tau_c (BSSA14)')
plt.title('Recorded vs. Predict MMI')
plt.legend()
plt.savefig(figSavePath+datasetNam+'_MMI_BSSA14_Tc.png', dpi=300)
plt.show()

gap = []
for c in range(len(test_Y)):
    gap.append(test_Y[c]-tcPred[c])
    
m_err, s_err = np.mean(gap), np.std(gap)    
fig = plt.figure(16, figsize=(7, 7), dpi=60)
ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2) 
ax2.scatter(np.arange(1, len(gap)+1), gap, marker='o', edgecolors='black', s=5,
            facecolor='orange', label='N='+str(len(gap)))
ax2.plot([0, len(test_Y)], [m_err+s_err, m_err+s_err], 'r--', linewidth=3)
ax2.plot([0, len(test_Y)], [m_err-s_err, m_err-s_err], 'r--', linewidth=3)
ax2.plot([0, len(test_Y)], [m_err, m_err], color='orange', linestyle='--', linewidth=3)
ax2.legend(fontsize=12, shadow=True)
ax2.set_xlabel('No.', fontsize=12)
ax2.set_ylabel('precord-predict', fontsize=12)
ax2.set_ylim([-2, 2])
n, bins, patches = ax3.hist(np.array(gap), facecolor="blue", edgecolor="black", alpha=0.6, 
                            bins=list(np.arange(-1.6, 1.6, 0.2)), orientation='horizontal')    
mu, sigma = np.mean(gap), np.std(gap)
y = norm.pdf(bins, mu, sigma)
ax3.plot(y*max(n), bins, 'r--')
ax3.text(120, 1.0, str(round(m_err, 2))+'±'+str(round(s_err, 2)), fontsize=12)
ax3.set_ylim([-2, 2])
plt.tight_layout()

