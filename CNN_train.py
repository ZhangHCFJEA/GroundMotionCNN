# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 05:18:43 2020

@author: ZhangHC
"""


import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow import set_random_seed
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling1D, GlobalAveragePooling1D
from keras.layers import Conv1D, Conv2D, Reshape
from keras.utils import plot_model
import keras.backend as K
from keras.optimizers import adam, sgd
import matplotlib.pyplot as pl
from sklearn.metrics import r2_score
import math, sys
sys.path.append('H:/python')
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
path = 'E:\MLData\groundMotion'
fileName = 'LA.hdf5'
f = h5py.File(os.path.join(path, fileName), 'r')


############
X = f['velData']
Y = f['MMI_PGA']

##########################################
# Show data distribution
figSavePath = 'H:/python/CNN_figure/'
datasetNam = 'CA'#'SCEDC'

x, y = f['epicentral distance'], f['Mag']
xlabel = 'Epicentral Distance (km)'
ylabel = 'Magnitude'
figName = datasetNam+'_hypoDistr.png'
ymin, ymax= 1.8, 7
xbin, ybin = 10, 0.5
datasetPlot(x, y, xlabel, ylabel, xbin, ybin, ymin, ymax, figSavePath, figName)

# #####################################
X_tmp = f['velData']
Y_tmp = f['MMI_PGA']
M_tmp = f['Mag']
d_tmp = f['epicentral distance']
dep_tmp = f['focal depth']
pga_tmp = f['PGA_BSSA14']
pd_tmp = f['Pd']
tc_tmp = f['Tau_c']
vs_tmp = f['Vs30']
PGV_tmp = f['PGV']
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

#
set_random_seed(1)
sampling_rate = 100.
train_x, test_x, train_Y, test_Y, _trainMag, testMag, _trainPGA, testPGA, _trainPd, \
    testPd, _trainTc, testTc, _trainVs, testVs, _trainDis, testDis, _trainPGV, testPGV = train_test_split(np.array(X[:, :, :]), np.array(Y),\
                                                                                      np.array(M), np.array(pga), np.array(pd), \
                                                                                      np.array(tc), np.array(vs), np.array(d), \
                                                                                      np.array(PGV), test_size=0.25, random_state=7)

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
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    adam = adam(lr=0.0001)
    model.compile(loss='mse', optimizer=adam) #, metrics=['accuracy']
    model.summary()
    plot_model(model, to_file='model.png')

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
    plt.hist([test_Y, np.reshape(pred_test_y, len(pred_test_y), )], 
             bins=[-2, -1,  0,  1,  2,  3,  4,  5,  6], 
             label={'record', 'predict'}, alpha=0.7)
    plt.legend()
    plt.grid()
    plt.xlabel('MMI') 
    plt.ylabel('Counts')
    plt.title('Histogram of MMI')
    plt.savefig(figSavePath+datasetNam+'_MMI_hist_'+str(win)+'.png', dpi=300)
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
    ax2.set_ylim([-2, 2])
    n, bins, patches = ax3.hist(np.array(gap), facecolor="blue", edgecolor="black", alpha=0.6, 
                                bins=list(np.arange(-2, 2, 0.2)), orientation='horizontal')    
    mu, sigma = np.mean(gap), np.std(gap)
    y = norm.pdf(bins, mu, sigma)
    ax3.plot(y*max(n), bins, 'r--')
    ax3.text(120, 1.0, str(round(m_err, 2))+'±'+str(round(s_err, 2)), fontsize=12)
    ax3.set_ylim([-2, 2])
    plt.tight_layout()
    plt.savefig(figSavePath+datasetNam+'_MMI_predGap_'+str(win)+'hist.png', dpi=200, transparent=True)
    plt.show()
    
    #
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
    plt.savefig(figSavePath+datasetNam+'_MMI'+str(win)+'.png', dpi=300)
    plt.show()
    
    num1, num2, num3, num4 = 0, 0, 0, 0
    level = 3
    for n in range(len(test_Y)):
        if test_Y[n]>=level and pred_test_y[n]>=level:
            num1 += 1
        elif test_Y[n]>=level and pred_test_y[n]<level:
            num2 += 1
        elif test_Y[n]<level and pred_test_y[n]>=level:
            num3 += 1
        elif test_Y[n]<level and pred_test_y[n]<level:
            num4 += 1

    #
    plt.figure(5, figsize=(7, 6), dpi=80)
    plt.scatter(test_Y, pred_test_y, c=pkT, marker='o', edgecolors='none', cmap="hsv", s=14, label='N='+str(len(test_Y)))
    plt.xlim([-2.5, 7])
    plt.ylim([-2.5, 7])
    plt.vlines(3, -3, 7, linestyles='--', linewidth=2, color='k')
    plt.hlines(3, -3, 7, linestyles='--', linewidth=2, color='k')
    plt.plot([-2.5, 7], [-2.5, 7], linestyle='--', linewidth=2, color='k')
    plt.xlabel('Record', fontsize=12) 
    plt.ylabel('Predict', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.text(6, 6, 'TP', fontsize=20, weight='bold', color='r')
    plt.text(6, 5.7, str(num1), fontsize=12, weight='bold', color='r')
    plt.text(-0.5, 6, 'FN', fontsize=20, weight='bold', color='r')
    plt.text(-0.5, 5.7, str(num3), fontsize=12, weight='bold', color='r')
    plt.text(-0.5, -1.5, 'TN', fontsize=20, weight='bold', color='r')
    plt.text(-0.5, -1.8, str(num4), fontsize=12, weight='bold', color='r')
    plt.text(6, -1.5, 'FP', fontsize=20, weight='bold', color='r')
    plt.text(6, -1.8, str(num2), fontsize=12, weight='bold', color='r')
    # plt.text(-1, 2.1, 'best', fontsize=14, color='r')
    # plt.text(-1, 3.6, 'worst', fontsize=14, color='r')
    plt.colorbar(label='Warning Time (s)')
    plt.legend()
    plt.savefig(figSavePath+datasetNam+'_MMI'+str(win)+'_FM.png', dpi=300)
    plt.show()
    
    
    #Graphing our training and validation
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.figure(6, figsize=(8, 5), dpi=80)
    # plt.semilogy(epochs, loss, 'r-', label='Training loss')
    plt.plot(epochs, loss, 'r-', label='Training loss')
    plt.plot(epochs, val_loss, 'b-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.ylabel('loss') 
    plt.xlabel('epoch')
    # plt.ylim(0, 2)
    plt.legend()
    plt.grid()
    plt.savefig(figSavePath+datasetNam+'_loss_plot'+str(win)+'.png', dpi=300)

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
