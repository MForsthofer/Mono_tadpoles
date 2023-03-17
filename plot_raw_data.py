# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 11:19:52 2020

resamples the whole signal first, butterworth filters it, and then cuts and exports individual cycles based 
on threshold crossing

@author: forsthofer
"""

from scipy import interpolate
import scipy.signal as signal
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import scipy.stats as st
import pandas as pd
import scipy.io as sp
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft

#set parameters for analysis
resamp_rate = 200 #resampling rate
stimtype = 1 #0 for hexapod, 1 for visual
filtfreq = 4 #cutoff for the low-pass butterworth filter
export_start = 42.25#from which time point to export
export_end = 82.25#to which time point to export
export = 1 #should the script export a trace? 0 for no, 1 for yes

def resample(stim_position, eye_position, resamp_rate):
    '''resamples time, stimulus and eye position signal to a set rate'''
    t = np.arange(eye_position.index[0], eye_position.index[-1], 1/resamp_rate) #interpolate time from beginning to end of eye recording at the set resample rate
    fstim = interpolate.interp1d(stim_position.index,stim_position,fill_value='extrapolate'); # get an interpolation function of the original stimulus data to the original time
    vispos = fstim(t) #apply the function to the resampled time
    fleye = interpolate.interp1d(eye_position.index,eye_position['Left Eye'],fill_value='extrapolate') #do the same for the left eye
    leyepos = np.array(fleye(t)) # apply the function
    freye = interpolate.interp1d(eye_position.index,eye_position['Right Eye'],fill_value='extrapolate')#and the riht eye
    reyepos = np.array(freye(t)) #apply the function
    eyepos = np.array([leyepos,reyepos], dtype='float32') #put resampled left and right eye traces back into one array
    return t, vispos, eyepos.transpose()

def butfilt(trace, filtfreq, filt_order):
    resamp_rate = 200
    #filter order determines how 'smooth' your filter output will be
    #cutoff frequency
    #filter frequency, normalized to the 'usable frequency range', which like in the 
    #FFT script, is up to half the sampling rate. say we want to cutoff at 5 Hz, and 
    #our sampling rate is 200. We can thus analyze frequencies up to 100 Hz, and 5 Hz 
    #is 5/(200/2) = 0.05 or 5% of that. Wn is the name that all the guides and examples 
    #use so i stuck with that, we could replace it with normalized_filtfreq?
    Wn = filtfreq/(resamp_rate/2) 
    #calculate numerator and denominator of the filter. Basically these are the filter 
    #settings. I'll explain more once i've read up on those words. 
    b, a = signal.butter(filt_order, Wn, btype='lowpass', output='ba', )
    trace_f = signal.filtfilt(b, a, trace)
    return trace_f

#load in your data in matlab format
Tk().withdraw()
filename = askopenfilename()
imported_data = sp.loadmat(filename)

#since the data structure is very convoluted, get the stimulus and eye positions
#into easily accessible variables (pandas dataframes)
stimchans = ['Visual', 'Hexapod']
eyechans = ['Left Eye', 'Right Eye']
stim_position = pd.DataFrame(data=-imported_data['Ch1'][0][0][8][:,0:2], index=imported_data['Ch1'][0][0][6].flatten(), columns=stimchans)
eye_position = pd.DataFrame(data=imported_data['Ch2'][0][0][8], index=imported_data['Ch2'][0][0][6].flatten(), columns=eyechans)
#get time and stim positions as arrays as we now need to work with the time array for interpolation

if stimtype==0:
    stimtime, stimpos, eyepos = resample(stim_position['Hexapod'], eye_position, resamp_rate)
elif stimtype ==1:
    stimtime, stimpos, eyepos = resample(stim_position['Visual'], eye_position, resamp_rate)

eyepos[:,0] = butfilt(eyepos[:,0], filtfreq, 4)
eyepos[:,1] = butfilt(eyepos[:,1], filtfreq, 4)

#make an FFT and plot it for the selected time period
eyes = eyepos[np.where(np.logical_and(stimtime> export_start, stimtime<export_end)),:][0]
#eyes = stimpos[np.where(np.logical_and(stimtime> export_start, stimtime<export_end))]
N = len(eyes)
T = np.mean(np.diff(stimtime)) #get the time difference between all the resampled time points as a vector. Then take the mean to get it down to one value 

#calculate the FFT for the left and the right eye signal, unfiltered


for current_eye in range(2):
    #get one eye (left or right) and normalize to mean of eye position in the next line
    eyetrace = eyes[:,current_eye] #get one eye out of the array containing both eye traces
    eyetrace = eyetrace-np.nanmean(eyetrace) #set the eye trace to move around zero
        
    #calculate the FFT
    eyetracef = fft(eyetrace) #make an fft from the eyetrace. will have the same amount of datapoints as the original trace. will be mirrored
    tf = np.linspace(0.0, 1.0/(2.0*T), N//2) #set the frequency domain (y axis). Needs to be half your sampling rate (google Nyquist Frequency). So if you sample with 100 Hz, this goes from 0 to 50 Hz. 
                                                #is as long as half your FFT (in datapoints)
    eyetracef2 = np.abs(eyetracef[0:N//2]) #'unmirror' the spectrum. It's symmetrical around the middle, so take the first half and take it x2. 
                                            # since in practise there are no negative values for frequency content, take the absolute of the spectrogram afterwards. 
    plt.plot(tf, 2.0/N * eyetracef2, linewidth=1) #plot the spectrogram to your generated y axis. normalize your frequency content to double the sampling interval (because we only take half of the spectrograms, meaning half the data points)
                                                    #1/N would be the sampling interval, 1/N*2 is double the sampling interval and is the same as 2/N
plt.xlim(0, 15)
plt.xlabel('Frequency (Hz)')
plt.ylabel('|P(f)|')
#filter eye position


if export == 1:
    dataexport = pd.DataFrame(data=eyepos, index=stimtime.transpose(), columns=('Left Eye', 'Right Eye'))
    dataexport['Stimulus'] = stimpos
    dataexport = dataexport[export_start:export_end]
    eyeposavg = dataexport.iloc[:,0:2].mean(axis=1)
    dataexport = pd.concat([dataexport, eyeposavg], axis=1)
    dataexport.rename(columns={0:'Both Eyes Average'}, inplace=True)
    dataexport.to_excel(filename[filename.rfind('/')+1:filename.find('.smr')]+'_exampletrace.xlsx')

f, ax = plt.subplots(3,1, sharex=True)
ax[0].plot(stimtime, stimpos)
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Position (°)')
ax[0].set_xlim(0, np.max(stimtime))
ax[0].set_title('Stimulus')

ax[1].plot(stimtime, eyepos[:,0])
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Position (°)')
ax[1].set_xlim(0, np.max(stimtime))
ax[1].set_title('Left Eye')

ax[2].plot(stimtime, eyepos[:,0])
ax[2].set_xlabel('Time (s)')
ax[2].set_ylabel('Position (°)')
ax[2].set_xlim(0, np.max(stimtime))
ax[2].set_title('Right Eye')

f.tight_layout()

# new_data = pd.DataFrame(data=eyepos, index=stimtime, columns=('Left Eye', 'Right Eye'))
# new_data['Stim position'] = stimpos
# new_data['Stim time'] = stimtime
# new_data.to_excel(filename[filename.rfind('/')+1:filename.find('.')]+('.xlsx'), index = False)

