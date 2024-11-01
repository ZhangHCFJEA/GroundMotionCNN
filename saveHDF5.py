# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 01:36:31 2020

@author: ZhangHC
"""

import sys
sys.path.append('H:/python/MudPy-master/src/python')
from obspy.core import read, Stream, UTCDateTime
import numpy as np
import pandas as pd
import h5py
from mudpy.forward import highpass
from obspy.geodetics.base import gps2dist_azimuth as dist_azi
import os
from obspy.clients.fdsn import Client

# initial a query client
client = Client(base_url='NCEDC')

#
def MMI_Worden2012(PGM, R, M, case):    
    if case == 'PGA':
        c1, c2, c3, c4, c5, c6, c7, t = 1.78, 1.55, -1.60, 3.70, -0.91, 1.02, -0.17, 4.22 
        #
        MMI = c1+c2*np.log10(PGM)+c5+c6*np.log10(R)+c7*M
        if MMI > t:
            MMI = c3+c4*np.log10(PGM)+c5+c6*np.log10(R)+c7*M        
    elif case == 'PGV':
        c1, c2, c3, c4, c5, c6, c7, t = 3.78, 1.47, 2.89, 3.16, 0.90, 0.00, -0.18, 4.56
        #
        MMI = c1+c2*np.log10(PGM)+c5+c6*np.log10(R)+c7*M
        if MMI > t:
            MMI = c3+c4*np.log10(PGM)+c5+c6*np.log10(R)+c7*M
    
    ####        
    return MMI

#
def save_h5(h5f,data,target):
    shape_list = list(data.shape)
    # create target datasets if not exist
    if not h5f.__contains__(target):
        shape_list[0] = None
        dataset = h5f.create_dataset(target, data=data, maxshape=tuple(shape_list), 
                                     chunks=True)
        return
    else:
        dataset = h5f[target]
    #
    len_old = dataset.shape[0]
    len_new = len_old+data.shape[0]
    #
    shape_list[0] = len_new
    dataset.resize(tuple(shape_list))
    dataset[len_old:len_new] = data
    
#  
earthquakes = pd.read_csv('H:/python/NSEDC.csv', dtype=str)
ot_yr = earthquakes.Date.values
ot_hr = earthquakes.Time.values
eventID = earthquakes.EventID.values
lat = earthquakes.Latitude.values
lon = earthquakes.Longitude.values
dep = earthquakes.Depth.values
mag = earthquakes.Magnitude.values
filePath = 'H:/MLData/groundMotion/Parkfield/'
net = 'BK'
sta = 'TSCN' #PKD,RAMR
stLoc = [35.543978, -120.348062] # TSCN
# stLoc = [35.688123, -120.400916] #TCHL
saveLen = 20

#
hf1 = h5py.File(os.path.join(filePath, sta, 'vel_wave2.hdf5'), 'a') #, 'hdf/vel3.hdf5'
hf2 = h5py.File(os.path.join(filePath, sta, 'acc_wave2.hdf5'), 'a')

#
for idx in range(len(earthquakes)-1, 0, -1):
    M = float(mag[idx])
    evtID = eventID[idx]
    if M>2.0:
        ot = str(ot_yr[idx])+'T'+str(ot_hr[idx])
        fileName = ot[0:4]+ot[5:7]+ot[8:10]+ot[11:13]+ot[14:16]+ot[17:]

        #
        try:
            f = open(os.path.join(filePath, 'phase', fileName+'.phase'))
            pha = f.read()
            tmp = pha.split('],')       
            f.close()
            picks = []
            for s in range(len(tmp)):
                bk = []
                for l in range(len(tmp[s][:])):
                    if tmp[s][l]==',':
                        bk.append(l)
                #
                year = int(tmp[s][bk[2]-4:bk[2]])
                mon = int(tmp[s][bk[3]-2:bk[3]])
                day = int(tmp[s][bk[4]-2:bk[4]])
                hr = int(tmp[s][bk[5]-2:bk[5]])
                mint = int(tmp[s][bk[6]-2:bk[6]])
                
                if len(bk)==8:
                    sec = int(tmp[s][bk[7]-2:bk[7]])
                    msec = int(tmp[s][bk[7]+1:bk[7]+6])
                    picks.append(UTCDateTime(year, mon, day, hr, mint, sec, msec))
                else:
                    sec = int(tmp[s][bk[6]+1:-1])
                    picks.append(UTCDateTime(year, mon, day, hr, mint, sec))
                    
                if tmp[s][bk[0]-3:bk[0]-1]==net and tmp[s][bk[0]+3:bk[1]-1]==sta:
                    arr = UTCDateTime(year, mon, day, hr, mint, sec, msec)
                
            #
            minArr = min(picks)
            #
            dist, az, baz = dist_azi(float(lat[idx]), float(lon[idx]), stLoc[0], stLoc[1])
            R = ((dist/1000)**2+(float(dep[idx]))**2)**0.5
            smp = 100.
            V, V2, pga, pgv, mmi = [], [], [], [], []
            #
            if 'arr' in locals():           
                try:
                    # vel = read(os.path.join(filePath, sta, fileName+'_vel.mseed'))
                    
                    vel = client.get_waveforms(net, sta, '*', 'HH?', UTCDateTime(ot)-1*60., 
                                                UTCDateTime(ot)+3*60., attach_response=True)
                    vel.remove_response(output="VEL") 
                    
                    # if sampling rate is not 100 Hz, then resample.
                    if vel[0].stats.sampling_rate < smp:
                        vel.interpolate(sampling_rate=smp)
                    elif vel[0].stats.sampling_rate > smp:
                        vel.resample(smp)
                    vel.detrend('linear')
    
                    # trim trace since P-onset
                    st_c = vel.copy()
                    st_c.trim(starttime=arr, endtime=arr+saveLen, fill_value=0)
                    # trim trace since the P arrival
                    st_c2 = vel.copy()
                    st_c2.trim(starttime=UTCDateTime(ot), endtime=UTCDateTime(ot)+saveLen, fill_value=0)
                    # diff vel to acc
                    st_c3 = vel.copy()
                    st_c3.differentiate()
                    st_c3.detrend('linear')
                    #
                    sta_c = st_c3.copy()
                    sta_c.trim(starttime=arr, endtime=arr+saveLen, fill_value=0)
                    sta_c2 = st_c3.copy()
                    sta_c2.trim(starttime=UTCDateTime(ot), endtime=UTCDateTime(ot)+saveLen, fill_value=0)
    
                    # 
                    V = np.vstack((st_c[0].data, st_c[1].data, st_c[2].data))
                    V2 = np.vstack((st_c2[0].data, st_c2[1].data, st_c2[2].data))
                    A = np.vstack((sta_c[0].data, sta_c[1].data, sta_c[2].data))
                    A2 = np.vstack((sta_c2[0].data, sta_c2[1].data, sta_c2[2].data))
                    
                    # PGA & PGV
                    pgv = max(max(abs(st_c[0].data)), max(abs(st_c[1].data)), max(abs(st_c[2].data)))
                    pga = max(max(abs(sta_c[0].data)), max(abs(sta_c[1].data)), max(abs(sta_c[2].data)))
                    
                    # MMI
                    mmi_pga = MMI_Worden2012(pga*100, R, M, 'PGA')
                    mmi_pgv = MMI_Worden2012(pgv*100, R, M, 'PGV') 
                    
                    #
                    save_h5(hf1, data=np.array([V]), target='velData')
                    save_h5(hf1, data=np.array([V2]), target='velData2')
                    save_h5(hf1, data=np.array([A]), target='accData')
                    save_h5(hf1, data=np.array([A2]), target='accData2')
                    save_h5(hf1, data=np.array([arr-UTCDateTime(ot)]), target='arrival') 
                    save_h5(hf1, data=np.array([minArr-UTCDateTime(ot)]), target='first arrival') 
                    save_h5(hf1, data=np.array([pgv]), target='PGV') 
                    save_h5(hf1, data=np.array([pga]), target='PGA')  
                    save_h5(hf1, data=np.array([mmi_pga]), target='MMI_PGA')  
                    save_h5(hf1, data=np.array([mmi_pgv]), target='MMI_PGV')  
                    save_h5(hf1, data=np.array([M]), target='Mag') 
                    save_h5(hf1, data=np.array([dist]), target='epicentral distance') 
                    save_h5(hf1, data=np.array([evtID.encode('utf-8')]), target='eventID') 
                    save_h5(hf1, data=np.array([float(dep[idx])]), target='focal depth') 
                    
                except Exception as e:
                    print(e)
    #                break
                
                #
                A, A2, pga, pgv, mmi = [], [], [], [], []
                try:
                    # acc = read(os.path.join(filePath, sta, fileName+'_acc.mseed'))
                    
                    acc = client.get_waveforms(net, sta, '*', 'HN?', UTCDateTime(ot)-1*60., 
                                                UTCDateTime(ot)+3*60., attach_response=True)
                    acc.remove_response(output="ACC") 
                    acc.detrend('linear')
                    
                    if acc[0].stats.sampling_rate < smp:
                        acc.interpolate(sampling_rate=smp)
                    elif acc[0].stats.sampling_rate > smp:
                        acc.resample(smp)
                    acc.detrend('linear')
                    #
                    st_c = acc.copy()
                    st_c.trim(starttime=arr, endtime=arr+saveLen, fill_value=0)
                    st_c2 = acc.copy()
                    st_c2.trim(starttime=UTCDateTime(ot), endtime=UTCDateTime(ot)+saveLen, fill_value=0)
                    # integrate acc to vel in order to calculate MMI.
                    st_c3 = acc.copy()
                    st_c3.integrate()
                    hpf = 10
                    st_c3[0].data = highpass(st_c3[0].data,1/hpf,1./st_c3[0].stats.delta,4,zerophase=False)
                    st_c3[1].data = highpass(st_c3[1].data,1/hpf,1./st_c3[1].stats.delta,4,zerophase=False)
                    st_c3[2].data = highpass(st_c3[2].data,1/hpf,1./st_c3[2].stats.delta,4,zerophase=False)
                    stv_c = st_c3.copy()
                    stv_c.trim(starttime=arr, endtime=arr+saveLen, fill_value=0)
                    stv_c2 = st_c3.copy()
                    stv_c2.trim(starttime=UTCDateTime(ot), endtime=UTCDateTime(ot)+saveLen, fill_value=0)
    
                    #
                    A = np.hstack((st_c[0].data, st_c[1].data, st_c[2].data))
                    A2 = np.hstack((st_c2[0].data, st_c2[1].data, st_c2[2].data))
                    V = np.hstack((stv_c[0].data, stv_c[1].data, stv_c[2].data))
                    V2 = np.hstack((stv_c2[0].data, stv_c2[1].data, stv_c2[2].data))
                    
                    #
                    pga = max(max(abs(st_c[0].data)), max(abs(st_c[1].data)), max(abs(st_c[2].data)))
                    pgv = max(max(abs(stv_c[0].data)), max(abs(stv_c[1].data)), max(abs(stv_c[2].data)))
                    
                    # MMI
                    mmi_pga = MMI_Worden2012(pga*100, R, M, 'PGA')
                    mmi_pgv = MMI_Worden2012(pgv*100, R, M, 'PGV')                
                                   
                    #
                    save_h5(hf2, data=np.array([V]), target='velData')
                    save_h5(hf2, data=np.array([V2]), target='velData2')
                    save_h5(hf2, data=np.array([A]), target='accData')
                    save_h5(hf2, data=np.array([A2]), target='accData2')
                    save_h5(hf1, data=np.array([arr-UTCDateTime(ot)]), target='arrival') 
                    save_h5(hf1, data=np.array([minArr-UTCDateTime(ot)]), target='first arrival') 
                    save_h5(hf2, data=np.array([pgv]), target='PGV') 
                    save_h5(hf2, data=np.array([pga]), target='PGA')  
                    save_h5(hf2, data=np.array([mmi_pga]), target='MMI_PGA')  
                    save_h5(hf2, data=np.array([mmi_pgv]), target='MMI_PGV')  
                    save_h5(hf2, data=np.array([M]), target='Mag') 
                    save_h5(hf2, data=np.array([dist]), target='epicentral distance') 
                    save_h5(hf2, data=np.array([evtID.encode('utf-8')]), target='eventID') 
                    save_h5(hf2, data=np.array([float(dep[idx])]), target='focal depth') 
                except Exception as e:
                    print(e)
            
            del arr
        except Exception as e:
            print(e)
    
    #
    print(idx)
    
hf1.close()
hf2.close()



#
import h5py
import os

filePath = 'H:/MLData/groundMotion/Parkfield'
sta = 'TCHL' #PKD,RAMR
hf1 = h5py.File(os.path.join(filePath, sta, 'vel_wave.hdf5'), 'r')

print(hf1.keys())
print(hf1['Mag'])
