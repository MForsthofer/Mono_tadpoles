# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 11:19:52 2020
Updates:
    25.05.2021
        - Added 1st and 2nd half cycle gain analysis
        - added amplitude analysis
        - added experimental integration
    
    26.06.2021
        - added patch note section. 
        - added a line at the cycle selection that lets you skip your last cycle. comment in and out at will. 
            don't forget that you did. 
            
    18.01.2022
        - changed first cycle analysis to only include first half of the first cycle. 
            before, there could be issues where the script found the amplitude across
            the whole cycle instead of just the first direction. 
        
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
import os.path 

#set parameters for analysis
resamp_rate = 200 #resampling rate
threshold = -10 #check for the stimulus to cross this value for segmentation
cycle_length = 10 #cycle length in seconds
phase = -0.25 #how much to shift phase for cutting. 0 is rising threshold, 1 is a full cycle
filtfreq = 4 #cutoff frequency for the lowpass butterworth filter
stimtype = 1 #type of stimulation. 0 for hexapod, 1 for visual
filt_order = 4 #filter order. determines the strenght of the frequency cutoff for the filter, higher means stronger. 4 is pretty standart, but 2 is fine too
section_cutout = 1
firstcycles = np.array([0]) #select which cycles to analyze separately

def data_subsection(stimtime, stimpos, eyetime, eyepos):
    '''Takes the eye and stimulus traces, plots them, and lets you select a part of it to 
    analyze further in the script by clicking before and after it in the plot and hitting enter'''

    #plot traces for stim, left and right eye
    f, ax = plt.subplots(3,1)
    mngr = plt.get_current_fig_manager()
    # to put it into the upper left corner for example:
    mngr.window.setGeometry(1,31,840, 840)
    ax[0].plot(stimtime, stimpos)
    ax[0].set_title('Stimulus position')
    ax[0].set_ylabel('Position (°)')
    ax[1].plot(eyetime, eyepos[:,0])
    ax[1].set_title('Left eye position')
    ax[1].set_ylabel('Position (°)')
    ax[2].plot(eyetime, eyepos[:,1])
    ax[2].set_title('Right eye position')
    ax[2].set_ylabel('Position (°)')
    ax[2].set_xlabel('Time (s)')
    
    #lets the user click in the plot. The x coordinate of the second to last click
    #will be the start of the part to analyze further, the last click the end. hit 
    #enter once you are happy with your selection. 
    x_y_click_coordinates = plt.ginput(-1, 0)
    plt.close()
    start_time = x_y_click_coordinates[-2][0]
    end_time = x_y_click_coordinates[-1][0]

    #cuts out the stimulus between your first and last click based on time. 
    eyepos2 = eyepos[(eyetime>start_time) & (eyetime<end_time),:]
    eyetime2 = eyetime[(eyetime>start_time) & (eyetime<end_time)]
    stimpos2 = stimpos[(stimtime>start_time) & (stimtime<end_time)]
    stimtime2 = stimtime[(stimtime>start_time) & (stimtime<end_time)]
    return (stimtime2, stimpos2, eyetime2, eyepos2)

def findtime(stimtime, stimpos, pre_cross, threshold):
    '''finds the time of neg to pos crossing of a threshold between two points, given the stimulus time and position, the threshold and the index before the crossing'''
    m = (stimpos[pre_cross+1]-stimpos[pre_cross])/(stimtime[pre_cross+1]-stimtime[pre_cross])
    crosstime = (threshold-stimpos[pre_cross])/m+stimtime[pre_cross]
    return crosstime

def findposition(eyetime, eyetrace, xcross):
    '''finds the position at a given time point by linear interpolation'''
    t_pre = np.where(eyetime<xcross)[-1][-1]
    m1 = (eyetrace[t_pre+1]-eyetrace[t_pre])/(eyetime[t_pre+1]-eyetime[t_pre])
    e_cross = (xcross-eyetime[t_pre])*m1+eyetrace[t_pre]
    return e_cross

def find_cycle(time, trace, pre_cross, startxy, endxy, resamp_rate, cycle_length):
    '''exports a cycle from an interpolated ''ideal'' start to an ''ideal'' interpolated end time and position'''
    i_end = np.where(time<endxy[0])[-1][-1]
    rawtime = time[pre_cross+1:i_end]
    rawtrace = trace[pre_cross+1:i_end]
    newtime = np.append(startxy[0], rawtime)
    newtime = np.append(newtime, endxy[0])
    newtrace = np.append(startxy[1], rawtrace)
    newtrace = np.append(newtrace, endxy[1])
    #plt.plot(newtime-newtime[0], newtrace)
    
    #resample the new trace to resamp rate
    t = np.linspace(newtime[0], newtime[-1], int(resamp_rate*cycle_length)) #linearly space a time vector from cycle start to end
    freye = interpolate.interp1d(newtime, newtrace)
    restrace = freye(t)
    #plt.plot(t-t[0], restrace, '.')
    return (t, restrace)

def butfilt(trace, filtfreq, filt_order, resamp_rate):
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
stimtime = imported_data['Ch1'][0][0][6].flatten()


if stimtype==0:
    stimpos = -imported_data['Ch1'][0][0][8][:,1:2] #this is for VOR
elif stimtype ==1:
    stimpos = -imported_data['Ch1'][0][0][8][:,0:1] #this is for OKR

eyetime = imported_data['Ch2'][0][0][6].flatten()
eyepos = imported_data['Ch2'][0][0][8]
#eyepos[:,0] = eyepos[:,1]

if section_cutout == 1:
    stimtime, stimpos, eyetime, eyepos = data_subsection(stimtime, stimpos, eyetime, eyepos)

#resample if wanted at this point
# stimtime, stimpos, eyepos = resample(stim_position, eye_position, resamp_rate)
# eyetime = stimtime #both eye and stim position are resampled according to the stimtime vector. best practise would be to call 
    #the resampled time t, and rename stimtime and eyetime from this point onward to t. 

#filter eye position

#preallocate the array where all the zero crossings will be saved
pre_crossing = np.array([])

#find zero crossings: find each point where the point itself is smaller than the threshold, 
#and the subsequent one is bigger than the threshold. store them in an array.
for i in range(len(stimpos)-1): #go through the stimulus channel
    if stimpos[i]>threshold and stimpos[i+1]<=threshold:  #for each datapoint, check if the datapoint is less than zero and the next one is equal to or more than zero
        pre_crossing = np.append(pre_crossing, i)   #if it is, get the first positive value after zero crossing, and append it's index to a
pre_crossing = pre_crossing.astype(int) #convert them to int

#for each previously point before zero crossing, find the true crossing point. From there, interpolate and export 
#the next cycle based on time

timei = np.zeros([1,resamp_rate*cycle_length])
stimi = np.zeros([1,resamp_rate*cycle_length])
l_eyei = np.zeros([1,resamp_rate*cycle_length])
r_eyei = np.zeros([1,resamp_rate*cycle_length])
l_eyeif = np.zeros([1,resamp_rate*cycle_length])
r_eyeif = np.zeros([1,resamp_rate*cycle_length])

timei[:] = np.nan
stimi[:] = np.nan
l_eyei[:] = np.nan
r_eyei[:] = np.nan
l_eyeif[:] = np.nan
r_eyeif[:] = np.nan

#comment the line with indexing to skip the last cycle. May be necessary if there is not at least half a cycle worth of recording time after stimulation end 
#for i in pre_crossing[0:-1]:
for i in pre_crossing:
    
    t_stimcross = findtime(stimtime, stimpos, i, threshold) #time of interpolated zero crossing of stimulus
    t_stimcross = t_stimcross-cycle_length*(-phase)
    
    #put the starting points into variables in the format [time, position]
    #start_stim = np.array([t_stimcross, threshold])
    start_stim = np.array([t_stimcross, findposition(stimtime, stimpos, t_stimcross)])
    start_left_eye = np.array([t_stimcross, findposition(eyetime, eyepos[:,0], t_stimcross)])#interpolate and find the position of the eye at that point
    start_right_eye = np.array([t_stimcross, findposition(eyetime, eyepos[:,1], t_stimcross)])#for both eyes
    
    #find the points [cycle length] seconds from start, and interpolate the position
    #then, put them into variables as well in the format [time, position]
    end_stim = np.array([t_stimcross+cycle_length, findposition(stimtime, stimpos, t_stimcross+cycle_length)])
    end_left_eye = findposition(eyetime,eyepos[:,0], t_stimcross+cycle_length)
    end_left_eye = np.array([t_stimcross+cycle_length, end_left_eye])
    end_right_eye = findposition(eyetime,eyepos[:,1], t_stimcross+cycle_length)
    end_right_eye = np.array([t_stimcross+cycle_length, end_right_eye])
    pre_cross_eye = np.where(eyetime<start_stim[0])[-1][-1]
    pre_cross_stim = np.where(stimtime<start_stim[0])[-1][-1]
    
    #using those start and end times: find the stimulus cycles, attach the generated 'ideal' start and end points, 
    #and resample them up to the resamp rate with linear interpolation
    # plt.plot(start_stim[0], start_stim[1], '.', 'r')
    # plt.plot(end_stim[0], end_stim[1], '.', 'b')
    res_time, res_stim = find_cycle(stimtime, stimpos, pre_cross_stim, start_stim, end_stim, resamp_rate, cycle_length)
    time_again, res_LE = find_cycle(eyetime, eyepos[:,0], pre_cross_eye, start_left_eye, end_left_eye, resamp_rate, cycle_length)
    time_again, res_RE = find_cycle(eyetime, eyepos[:,1], pre_cross_eye, start_right_eye, end_right_eye, resamp_rate, cycle_length)    
    
    #get the eye position half a cycle duration (in s) before and after the current cycle and append each
    #to the respective end of the cycle. 
    
    #calculate the index before start and the interpolated end point for the precycle time. endpoint is the start point of the current cycle 
    start_precycleL = np.array([start_left_eye[0]-cycle_length/2, np.nan])
    start_precycleL[1] = findposition(eyetime, eyepos[:,0], start_precycleL[0])
    pre_precycleL = np.where(eyetime<start_precycleL[0])[-1][-1]
    pre_timeL, pre_cycleL = find_cycle(eyetime, eyepos[:,0], pre_precycleL, start_precycleL, start_left_eye, resamp_rate, cycle_length/2)
    pre_timeL = pre_timeL[0:-1]
    pre_cycleL = pre_cycleL[0:-1]
    
    #same for right eye
    start_precycleR = np.array([start_right_eye[0]-cycle_length/2, np.nan])
    start_precycleR[1] = findposition(eyetime, eyepos[:,1], start_precycleR[0])
    pre_precycleR = np.where(eyetime<start_precycleR[0])[-1][-1]
    pre_timeR, pre_cycleR = find_cycle(eyetime, eyepos[:,1], pre_precycleR, start_precycleR, start_right_eye, resamp_rate, cycle_length/2)
    pre_timeR = pre_timeR[0:-1]
    pre_cycleR = pre_cycleR[0:-1]
    
    #calculate the post cycle period analog to the pre cycle. start point is the end
    #of the current cycle so we need to find the endpoint
    
    #left eye
    end_postcycleL = np.array([end_left_eye[0]+cycle_length/2, np.nan])
    end_postcycleL[1] = findposition(eyetime, eyepos[:,0], end_postcycleL[0])
    pre_postcycleL = np.where(eyetime<end_left_eye[0])[-1][-1]
    post_timeL, post_cycleL = find_cycle(eyetime, eyepos[:,0], pre_postcycleL, end_left_eye, end_postcycleL, resamp_rate, cycle_length/2)
    post_timeL = post_timeL[1:]
    post_cycleL = post_cycleL[1:]
    
    #right eye
    end_postcycleR = np.array([end_right_eye[0]+cycle_length/2, np.nan])
    end_postcycleR[1] = findposition(eyetime, eyepos[:,1], end_postcycleR[0])
    pre_postcycleR = np.where(eyetime<end_right_eye[0])[-1][-1]
    post_timeR, post_cycleR = find_cycle(eyetime, eyepos[:,1], pre_postcycleR, end_right_eye, end_postcycleR, resamp_rate, cycle_length/2)
    post_timeR = post_timeR[1:]
    post_cycleR = post_cycleR[1:]
    
    #append pre-, post- and main cycles
    app_cyclesL = np.append(np.append(pre_cycleL, res_LE), post_cycleL)
    app_cyclesR = np.append(np.append(pre_cycleR, res_RE), post_cycleR)
    
    #filter the appended cycles
    app_cyclesL = butfilt(app_cyclesL, filtfreq, filt_order, resamp_rate)
    app_cyclesR = butfilt(app_cyclesR, filtfreq, filt_order, resamp_rate)
    
    #cut out the main cycle again
    app_cyclesL = app_cyclesL[len(pre_cycleL):len(pre_cycleL)+len(res_LE)]
    app_cyclesR = app_cyclesR[len(pre_cycleR):len(pre_cycleR)+len(res_RE)]
    
    l_eyeif = np.vstack((l_eyeif, app_cyclesL))
    r_eyeif = np.vstack((r_eyeif, app_cyclesR))
    timei = np.vstack((timei, res_time))
    stimi = np.vstack((stimi, res_stim))
    l_eyei = np.vstack((l_eyei, res_LE))
    r_eyei = np.vstack((r_eyei, res_RE))
    
del_cycs = np.zeros(len(timei), dtype=bool)
del_cycs[:] = False
    
for ii in np.arange(1,len(timei)):
    
    f, ax = plt.subplots(2,1)
    mngr = plt.get_current_fig_manager()
    # to put it into the upper left corner for example:
    mngr.window.setGeometry(1,31,840, 840)
        
    ax[0].plot(eye_position[timei[ii,0]-cycle_length/2:timei[ii,-1]+cycle_length/2]['Left Eye'], color=[0.8, 0.8, 1])    
    ax[0].plot(timei[ii,:], l_eyei[ii,:], color='blue')
    ax[0].plot(timei[ii,:], l_eyeif[ii,:], color='red', linewidth=1)
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Eye Position (°)')
    ax[0].set_title('Left Eye')
    axr = ax[0].twinx()
    axr.plot(timei[ii,:], stimi[ii,:], color='grey')
    axr.set_ylabel('Stim Position (°)')
    
    ax[1].plot(eye_position[timei[ii,0]-cycle_length/2:timei[ii,-1]+cycle_length/2]['Right Eye'], color=[0.8, 0.8, 1])    
    ax[1].plot(timei[ii,:], r_eyei[ii,:], color='blue')
    ax[1].plot(timei[ii,:], r_eyeif[ii,:], color='red', linewidth=1)
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Eye Position (°)')
    ax[1].set_title('Right Eye')
    axr2 = ax[1].twinx()
    axr2.plot(timei[ii,:], stimi[ii,:], color='grey')
    axr2.set_ylabel('Stim Position (°)')
    f.suptitle('Cycle ' + str(ii) , fontsize=14)
    f.tight_layout()
    plt.pause(0.3)
    
    keep = input('Keep Cycle '+str(ii)+'? Enter to keep, any key to delete, q to abort')
    #use this if you want to just click yes for all
    #keep = ''
    if keep == '':
        print('Kept cycle Nr. ' + str(ii)) 
        del_cycs[ii] = False
        
    elif keep == 'q':
        print('You have selected to end the script. Please run again.')
        plt.close()
        break      
           
    else:    
        timei[ii,:] = np.nan
        stimi[ii,:] = np.nan
        l_eyei[ii,:] = np.nan
        r_eyei[ii,:] = np.nan
        l_eyeif[ii,:] = np.nan
        r_eyeif[ii,:] = np.nan
        del_cycs[ii] = True
        print('Deleted cycle Nr. ' + str(ii))
    
    plt.close()

kept_cycles = len(np.where(del_cycs==False)[0])-1
dftitles = ['Mean Time', 'Mean Stim', 'Mean Left Eye Interpolated', 'Mean Right Eye Interpolated', 'Mean Left Eye Interpolated & Filtered', 'Mean Right Eye Interpolated & Filtered', 'Mean Both Eyes Filtered', 'Mean Both Eyes', 'Integration Both Eyes Filtered']    
mean_time = np.nanmean(timei.transpose()-timei[:,0], axis=1)
mean_stim = np.nanmean(stimi, axis=0)
mean_eyeli = np.nanmean(l_eyei, axis=0)
mean_eyeri = np.nanmean(r_eyei, axis=0)
mean_eyelif = np.nanmean(l_eyeif, axis=0)
mean_eyerif = np.nanmean(r_eyeif, axis=0)
mean_both_eyes = np.nanmean(np.concatenate((l_eyei, r_eyei)), axis=0)
mean_both_eyes_filtered = np.nanmean(np.concatenate((l_eyeif, r_eyeif)), axis=0)
integration_both_eyes_filtered = np.cumsum(mean_both_eyes_filtered/resamp_rate)
mean_traces = pd.DataFrame(data=np.array([mean_time, mean_stim, mean_eyeli, mean_eyeri, mean_eyelif, mean_eyerif, mean_both_eyes_filtered, mean_both_eyes, integration_both_eyes_filtered]).transpose(), index = mean_time, columns=dftitles)

mean_traces_normalized = pd.DataFrame(data=np.array([mean_time, mean_stim-mean_stim[0], mean_eyeli-mean_eyeli[0], mean_eyeri-mean_eyeri[0], mean_eyelif-mean_eyelif[0], mean_eyerif-mean_eyerif[0], mean_both_eyes_filtered-mean_both_eyes_filtered[0], mean_both_eyes-mean_both_eyes[0], integration_both_eyes_filtered-integration_both_eyes_filtered[0]]).transpose(), index = mean_time, columns=dftitles)

stim_amp = np.diff((np.min(mean_stim), np.max(mean_stim)))
t_peak_stim = mean_time[np.argmin(mean_stim)]

#analysis of selected first cycles 
firstcycle_mean_both_eyes_filtered = np.nanmean(np.concatenate((l_eyeif[firstcycles], r_eyeif[firstcycles])), axis=0)
#take only the first half of the cycle, rounding to make sure it doesn't come back with a fraction which would crash the indexing
firsthalfcycle_mean_both_eyes_filtered = firstcycle_mean_both_eyes_filtered[0:round(len(firstcycle_mean_both_eyes_filtered)/2)]
firstcycle_mean_both_eyes_filtered_normalized = firstcycle_mean_both_eyes_filtered - firstcycle_mean_both_eyes_filtered[0]

#find indices of min and max of the 1st cycle
fc_min = np.argmin(firsthalfcycle_mean_both_eyes_filtered)
fc_max = np.argmax(firsthalfcycle_mean_both_eyes_filtered)

#calculate amplitude, depending on direction. always subtracts second peak from first peak, so + for increase, - on decrease
if fc_max>fc_min:
    results1c = pd.DataFrame(data=[firsthalfcycle_mean_both_eyes_filtered[fc_max] - firsthalfcycle_mean_both_eyes_filtered[fc_min]], index=['amplitude 1st half cycle'], columns=['both eyes filtered'])
    results1c = results1c.append(results1c.rename(index={'amplitude 1st half cycle':'gain 1st half cycle'})/stim_amp[0])
    results1c = results1c.append(pd.DataFrame(data=[mean_time[fc_max] - t_peak_stim], index=['phase (s)'], columns=['both eyes filtered']))

elif fc_min>fc_max:
    results1c = pd.DataFrame(data=[firsthalfcycle_mean_both_eyes_filtered[fc_min] - firsthalfcycle_mean_both_eyes_filtered[fc_max]], index=['amplitude 1st half cycle'], columns=['both eyes filtered'])
    results1c = results1c.append(results1c.rename(index={'amplitude 1st half cycle':'gain 1st half cycle'})/stim_amp[0])
    results1c = results1c.append(pd.DataFrame(data=[mean_time[fc_min] - t_peak_stim], index=['phase (s)'], columns=['both eyes filtered']))

#analysis of all valid cycles
if stimtype==1:
    
    if np.argmin(mean_eyeri)==0 |  np.argmin(mean_eyeri)==len(mean_eyeri)-1 | np.argmin(mean_eyeli)==0 |  np.argmin(mean_eyeri)==len(mean_eyeli)-1:
        results = pd.DataFrame(data=[np.diff((np.min(mean_both_eyes), np.max(mean_both_eyes)))/stim_amp], index=['amplitude late'], columns=['both eyes'])
        results['both eyes'] = np.nan
        results['both eyes filtered'] = np.nan
        results['left eye'] = np.nan
        results['left eye filtered'] = np.nan
        results['right eye'] = np.nan
        results['right eye filtered'] = np.nan
    
        
        results3 = pd.DataFrame(data=[np.diff((np.min(mean_both_eyes), np.max(mean_both_eyes)))/stim_amp], index=['amplitude early'], columns=['both eyes'])
        results3['both eyes'] = np.nan
        results3['both eyes filtered'] = np.nan
        results3['left eye'] = np.nan
        results3['left eye filtered'] = np.nan
        results3['right eye'] = np.nan
        results3['right eye filtered'] = np.nan
    
    else:    
        results = pd.DataFrame(data=[np.diff((np.min(mean_both_eyes), np.max(mean_both_eyes)))/stim_amp], index=['amplitude late'], columns=['both eyes'])
        results['both eyes'] = np.diff((np.min(mean_both_eyes), np.max(mean_both_eyes[np.argmin(mean_both_eyes):-1])))
        results['both eyes filtered'] = np.diff((np.min(mean_both_eyes_filtered), np.max(mean_both_eyes_filtered[np.argmin(mean_both_eyes_filtered):-1])))
        results['left eye'] = np.diff((np.min(mean_eyeli), np.max(mean_eyeli[np.argmin(mean_eyeli):-1])))
        results['left eye filtered'] = np.diff((np.min(mean_eyelif), np.max(mean_eyelif[np.argmin(mean_eyelif):-1])))
        results['right eye'] = np.diff((np.min(mean_eyeri), np.max(mean_eyeri[np.argmin(mean_eyeri):-1])))
        # results['right eye'] = np.diff((np.min(mean_eyeri), np.max(mean_eyeri)))
        results['right eye filtered'] = np.diff((np.min(mean_eyerif), np.max(mean_eyerif[np.argmin(mean_eyerif):-1])))
        # results['right eye filtered'] = np.diff((np.min(mean_eyerif), np.max(mean_eyerif)))
    
        results3 = pd.DataFrame(data=[np.diff((np.min(mean_both_eyes), np.max(mean_both_eyes)))/stim_amp], index=['amplitude early'], columns=['both eyes'])
        results3['both eyes'] = np.diff((np.min(mean_both_eyes), np.max(mean_both_eyes[0:np.argmin(mean_both_eyes)])))
        results3['both eyes filtered'] = np.diff((np.min(mean_both_eyes_filtered), np.max(mean_both_eyes_filtered[0:np.argmin(mean_both_eyes_filtered)])))
        results3['left eye'] = np.diff((np.min(mean_eyeli), np.max(mean_eyeli[0:np.argmin(mean_eyeli)])))
        results3['left eye filtered'] = np.diff((np.min(mean_eyelif), np.max(mean_eyelif[0:np.argmin(mean_eyelif)])))
        results3['right eye'] = np.diff((np.min(mean_eyeri), np.max(mean_eyeri[0:np.argmin(mean_eyeri)])))
        results3['right eye filtered'] = np.diff((np.min(mean_eyerif), np.max(mean_eyerif[0:np.argmin(mean_eyerif)])))
        # results3['right eye'] = np.diff((np.min(mean_eyeri), np.max(mean_eyeri)))
        # results3['right eye filtered'] = np.diff((np.min(mean_eyerif), np.max(mean_eyerif)))
    
    results2 = pd.DataFrame(data=[mean_time[np.argmin(mean_both_eyes)] - t_peak_stim], index=['phase (s)'], columns=['both eyes'])
    results2['both eyes filtered'] = mean_time[np.argmin(mean_both_eyes_filtered)] - t_peak_stim
    results2['left eye'] = mean_time[np.argmin(mean_eyeli)] - t_peak_stim
    results2['left eye filtered'] = mean_time[np.argmin(mean_eyelif)] - t_peak_stim
    results2['right eye'] = mean_time[np.argmin(mean_eyeri)] - t_peak_stim
    results2['right eye filtered'] = mean_time[np.argmin(mean_eyerif)] - t_peak_stim
    
elif stimtype==0:
    
    if np.argmax(mean_eyeri)==0 |  np.argmax(mean_eyeri)==len(mean_eyeri)-1 | np.argmax(mean_eyeli)==0 |  np.argmax(mean_eyeri)==len(mean_eyeli)-1:
        results = pd.DataFrame(data=[np.diff((np.min(mean_both_eyes), np.max(mean_both_eyes)))/stim_amp], index=['amplitude late'], columns=['both eyes'])
        results['both eyes'] = np.nan
        results['both eyes filtered'] = np.nan
        results['left eye'] = np.nan
        results['left eye filtered'] = np.nan
        results['right eye'] = np.nan
        results['right eye filtered'] = np.nan
    
        
        results3 = pd.DataFrame(data=[np.diff((np.min(mean_both_eyes), np.max(mean_both_eyes)))/stim_amp], index=['amplitude early'], columns=['both eyes'])
        results3['both eyes'] = np.nan
        results3['both eyes filtered'] = np.nan
        results3['left eye'] = np.nan
        results3['left eye filtered'] = np.nan
        results3['right eye'] = np.nan
        results3['right eye filtered'] = np.nan
    else:
        results = pd.DataFrame(data=[np.diff((np.max(mean_both_eyes), np.min(mean_both_eyes)))/stim_amp], index=['amplitude late'], columns=['both eyes'])
        results['both eyes'] = np.diff((np.max(mean_both_eyes), np.min(mean_both_eyes[np.argmax(mean_both_eyes):-1])))
        results['both eyes filtered'] = np.diff((np.max(mean_both_eyes_filtered), np.min(mean_both_eyes_filtered[np.argmax(mean_both_eyes_filtered):-1])))
        results['left eye'] = np.diff((np.max(mean_eyeli), np.min(mean_eyeli[np.argmax(mean_eyeli):-1])))
        results['left eye filtered'] = np.diff((np.max(mean_eyelif), np.min(mean_eyelif[np.argmax(mean_eyelif):-1])))
        results['right eye'] = np.diff((np.max(mean_eyeri), np.min(mean_eyeri[np.argmax(mean_eyeri):-1])))
        results['right eye filtered'] = np.diff((np.max(mean_eyerif), np.min(mean_eyerif[np.argmax(mean_eyerif):-1])))
        
        results3 = pd.DataFrame(data=[np.diff((np.max(mean_both_eyes), np.min(mean_both_eyes)))], index=['amplitude early'], columns=['both eyes'])
        results3['both eyes'] = np.diff((np.max(mean_both_eyes), np.min(mean_both_eyes[0:np.argmax(mean_both_eyes)])))
        results3['both eyes filtered'] = np.diff((np.max(mean_both_eyes_filtered), np.min(mean_both_eyes_filtered[0:np.argmax(mean_both_eyes_filtered)])))
        results3['left eye'] = np.diff((np.max(mean_eyeli), np.min(mean_eyeli[0:np.argmax(mean_eyeli)])))
        results3['left eye filtered'] = np.diff((np.max(mean_eyelif), np.min(mean_eyelif[0:np.argmax(mean_eyelif)])))
        results3['right eye'] = np.diff((np.max(mean_eyeri), np.min(mean_eyeri[0:np.argmax(mean_eyeri)])))
        results3['right eye filtered'] = np.diff((np.max(mean_eyerif), np.min(mean_eyerif[0:np.argmax(mean_eyerif)])))
        
        results2 = pd.DataFrame(data=[mean_time[np.argmax(mean_both_eyes)] - t_peak_stim], index=['phase (s)'], columns=['both eyes'])
        results2['both eyes filtered'] = mean_time[np.argmax(mean_both_eyes_filtered)] - t_peak_stim
        results2['left eye'] = mean_time[np.argmax(mean_eyeli)] - t_peak_stim
        results2['left eye filtered'] = mean_time[np.argmax(mean_eyelif)] - t_peak_stim
        results2['right eye'] = mean_time[np.argmax(mean_eyeri)] - t_peak_stim
        results2['right eye filtered'] = mean_time[np.argmax(mean_eyerif)] - t_peak_stim

results = results.append(abs(results.rename(index={'amplitude late':'gain late'}))/stim_amp[0]) 
results = results.append(results3) 
results = results.append(abs(results3.rename(index={'amplitude early':'gain early'}))/stim_amp[0])
results = results.append(results2)
results4 = results2.rename(index={'phase (s)':'phase (°)'})*360/np.max(mean_time)
results = results.append(results4)

results_total = pd.DataFrame(data=[np.diff((np.min(mean_both_eyes), np.max(mean_both_eyes)))], index=['amplitude total'], columns=['both eyes'])
results_total['both eyes filtered'] = np.diff((np.min(mean_both_eyes_filtered), np.max(mean_both_eyes_filtered)))
results_total['left eye'] = np.diff((np.min(mean_eyeli), np.max(mean_eyeli)))
results_total['left eye filtered'] = np.diff((np.min(mean_eyelif), np.max(mean_eyelif)))
results_total['right eye'] = np.diff((np.min(mean_eyeri), np.max(mean_eyeri)))
results_total['right eye filtered'] = np.diff((np.min(mean_eyerif), np.max(mean_eyerif)))
    
results.loc['amplitude total'] = results_total.values[0]
results.loc['gains total'] = results_total.values[0]/stim_amp[0]


metadata = pd.DataFrame([resamp_rate])
metadata['resamp rate'] = resamp_rate
metadata['threshold'] =threshold
metadata['cycle_length'] = cycle_length
metadata['phase'] = phase
metadata['filtfreq'] = filtfreq 
metadata['stimtype'] = stimtype
metadata['n Cycles'] = len(np.where(del_cycs==False)[0])-1
metadata['set start direction'] = 'leftward'
metadata['filt_order'] = filt_order
metadata['first cycles'] = str(firstcycles)

#normalized single cycles in case they're needed 
l_eyeifn = l_eyeif-l_eyeif[1,0]
r_eyeifn = r_eyeif-r_eyeif[1,0]
stimin = stimi-stimi[1,0]
l_eyein = l_eyei-l_eyei[1,0]
r_eyein = r_eyei - r_eyei[1,0]

verNr = 1
verTag = '_ver'+str(verNr)
while os.path.isfile(filename[filename.rfind('/')+1:filename.find('.smr')]+verTag+('.xlsx')) == True:
   verNr = verNr+1
   verTag = '_ver'+str(verNr)
   
#np.savez(filename[filename.rfind('/')+1:filename.find('.smr')]+verTag, stim_position=stim_position, eye_position=eye_position, timei=timei, stimi=stimi, l_eyei=l_eyei, r_eyei=r_eyei, l_eyeif=l_eyeif, r_eyeif=r_eyeif, mean_traces=mean_traces, del_cycs=del_cycs.astype(int))

with pd.ExcelWriter(filename[filename.rfind('/')+1:filename.find('.smr')]+verTag+('.xlsx')) as writer:  
    pd.DataFrame(stimi.transpose()).to_excel(writer, sheet_name='Stimuli')
    pd.DataFrame(l_eyei.transpose()).to_excel(writer, sheet_name='Left Eye Interpolated')
    pd.DataFrame(r_eyei.transpose()).to_excel(writer, sheet_name='Right Eye Interpolated')
    pd.DataFrame(l_eyeif.transpose()).to_excel(writer, sheet_name='Left Eye interpolated filtered')
    pd.DataFrame(r_eyeif.transpose()).to_excel(writer, sheet_name='Right Eye interpolated filtered')
    pd.DataFrame(timei.transpose()).to_excel(writer, sheet_name='Time')
    mean_traces.to_excel(writer, sheet_name='Mean Traces')
    mean_traces_normalized.to_excel(writer, sheet_name='Normalized Mean Traces')
    pd.DataFrame([kept_cycles]).to_excel(writer, sheet_name='n Cycles')
    metadata.to_excel(writer, sheet_name='metadata')
    results.to_excel(writer, sheet_name='results')
    # results1c.to_excel(writer, sheet_name='1st halfcycle results')
    # pd.DataFrame(pd.DataFrame(data=np.array([firstcycle_mean_both_eyes_filtered, firstcycle_mean_both_eyes_filtered_normalized]).transpose(), index = mean_time, columns=['mean trace 1stcycles', 'normalized mean trace 1stcycles']).to_excel(writer, sheet_name='1st cycle mean traces'))
#if you get an error saving to excel, try pip install openpyxl
