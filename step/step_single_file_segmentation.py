
##############adjust resting position_position function ###########################

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
latency_analysis = 0 #Do you wish to do manual latency selection? 0, no; 1, yes
analysis_delay = 0.75 #time in seconds until velocity measurement starts following
#stimulus onset
export_delay = 1.25 #Time prior to onset to export for position visualization 
resamp_rate = 200 #resampling rate
filtfreq = 4 #cutoff frequency for the lowpass butterworth filter 
filt_order = 4 #filter order. determines the strenght of the frequency cutoff 
#for the filter, higher means stronger. 4 is pretty standart, but 2 is fine too
eyevel_seconds = 2 #number of seconds to take at the beginning of each stim bout
#in the eye velocity trace
initial_indices = resamp_rate*eyevel_seconds #distance in datapoints from onset
#of stimulus used to profile velocity of eye motion response
omit_indices = int(analysis_delay*resamp_rate) #distance in datapoints until 
#velocity analysis starts 
pre_FP_indices = 20 #when a fast phase is detected, this amount of datapoints
    #gets tossed as a result of that. 
section_cutout = 1


def fastphase_finder(eye_velocity):  
    '''finds fast phases based on an acceleration threshold (10x the mean 
    across the whole recording). 
    Returns an array with all the starting indices of fast phases'''
    accel = abs(np.diff(eyevel)) #absolute, because we don't care about 
    #fast phase direciont, but about time. if we don'T do abs, we'd have to look
    #for the onset of leftward and rightward fast phases individually. 
    accel_threshold = 10*np.mean(accel)
    fastphase_indices = accel>accel_threshold
    fastphase_indices[1:][fastphase_indices[1:] & fastphase_indices[:-1]] = False
    fastphase_starts = np.where(fastphase_indices==True) 
    return fastphase_starts[0] 

def resting_position(stimtime, eyepos, eyevel, initial_indices, resamp_rate):
    ''' Given stimtime, eye velocity and sampling range (initial indices) for a 
    range of eye motion where no sitmulation is occuring (resting), determines
    the resting eye velocity rate. Exports data into npz and excel sheet.
    Returns resting eye velocity, time values for these, and the mean velocity.'''
    #Input how many instances of resting eye position to profile for velocity
    n_samples = 4
    start_diffs = (len(stimtime)-1600)//n_samples
    #Creates empty arrays to fill with sliced data
    resting_vel_traces = np.zeros([initial_indices, n_samples])
    resting_times = np.zeros([initial_indices, n_samples]) 
    real_cycles = []
    
    #For each sample range, data points are taken per the sampling range chosen
    #and eye velocity values are sliced out 
    for i  in range(n_samples): 
        f, ax = plt.subplots() 
        mngr = plt.get_current_fig_manager()
        # Places plot into the upper left corner for example:
        mngr.window.setGeometry(1,31,840, 640)
        ax.plot(stimtime[i*start_diffs:i*start_diffs+initial_indices+1800], 
        eyevel[i*start_diffs:i*start_diffs+initial_indices+1800], color='gray')
        ax.plot(stimtime[i*start_diffs:i*start_diffs+initial_indices], 
        eyevel[i*start_diffs:i*start_diffs+initial_indices], color='red')
        ax.set_title('Resting VELOCITY!!! for Step')  
        ax.set_ylim(np.mean(eyevel[i*start_diffs:i*start_diffs+initial_indices])-10, 
                    np.mean(eyevel[i*start_diffs:i*start_diffs+initial_indices])+10)
        plt.pause(0.3)
        keep = input('Keep this segment? Enter to keep, f key to remove false detection movement, press any other key to keep')
        plt.close()
        
        if keep == '':
            resting_vel_traces[:, i] = eyevel[i*start_diffs:i*start_diffs+initial_indices]
            resting_times[:,i] = stimtime[i*start_diffs:i*start_diffs+initial_indices] 
            real_cycles.append(i)
        elif keep == 'f': 
            print('False onset detection. Segement removed from analysis.')
            
        else:
            real_cycles.append(i)
            resting_vel_traces[:, i] = np.nan
            resting_times[:,i] = np.nan
    resting_vel_traces = resting_vel_traces [real_cycles]
    resting_times = resting_times [real_cycles]    
        
    #Also get baseline amplitudes for comparison with sinus recordings for 3 freqs. 
    n_sin_samples = 3
    frequencies = [0.1, 0.2, 0.5]
    start_diffs_sin = len(stimtime)//n_samples
    #Creates empty arrays to fill with sliced data
    resting_sin_traces = np.zeros([int(1/np.min(frequencies)*resamp_rate), n_sin_samples, len(frequencies)])
    resting_sin_traces[:] = np.nan
    resting_sin_times = np.zeros([int(1/np.min(frequencies)*resamp_rate), n_sin_samples, len(frequencies)]) 
    resting_sin_times[:] = np.nan
    
    #For each sample range, data points are taken per the sampling range chosen
    #and eye velocity values are sliced out 
    for i  in range(n_sin_samples): 
        for n_freq in range(len(frequencies)):
            f, ax = plt.subplots() 
            mngr = plt.get_current_fig_manager()
            # Places plot into the upper left corner for example:
            mngr.window.setGeometry(1,31,840, 640)
            ax.plot(stimtime[i*start_diffs_sin:i*start_diffs_sin+int(resamp_rate*1/frequencies[0])], 
            eyepos[i*start_diffs_sin:i*start_diffs_sin+int(resamp_rate*1/frequencies[0])], color='gray')
            ax.plot(stimtime[i*start_diffs_sin:i*start_diffs_sin+int(resamp_rate*1/frequencies[n_freq])], 
            eyepos[i*start_diffs_sin:i*start_diffs_sin+int(resamp_rate*1/frequencies[n_freq])], color='red')
            ax.set_title('Resting position for Sinus at ' + str(n_freq))  
            plt.pause(0.3)
            keep = input('Keep this segment? Enter to keep, f key to remove false detection movement, press any other key to keep')
            plt.close()
            
            if keep == '':
                resting_sin_traces[0:int(resamp_rate*1/frequencies[n_freq]), i, n_freq] = eyepos[i*start_diffs:int(i*start_diffs+resamp_rate*1/frequencies[n_freq])]
                resting_sin_times[0:int(resamp_rate*1/frequencies[n_freq]),i, n_freq] = stimtime[i*start_diffs:int(i*start_diffs+resamp_rate*1/frequencies[n_freq])] 
            else: 
                resting_sin_traces[:, i, n_freq] = np.nan
                resting_sin_times[:,i, n_freq] = np.nan
    
        
    #mean values are calculates for each period of resting eye velocity
    resting_mean_vels = np.mean(resting_vel_traces, axis=0)
    #Calculate  the absolute value of the velocity.
    average_vel = resting_mean_vels.mean()
    
    amplitudes = np.nanmin(resting_sin_traces, axis=0) - np.nanmax(resting_sin_traces, axis=0)
    amplitudes = pd.DataFrame(data = amplitudes, columns = ['0.1 Hz', '0.2 Hz', '0.5 Hz'])
    
    #Exporting data into python npz and Excel sheets for plotting 
    #Metadata with input variables and response bout counts
    metadata = [resamp_rate, filtfreq, filt_order, initial_indices]
    metadata_names = ['Resample Rate', 'Filter Cutoff Frequency', 'Filter Order',
                      'Velocity Sampling Datapoints Length']
    meta = pd.Series(metadata, index=metadata_names) 
    
    #Exporting data analysis into a npz python file with the following variables
    np.savez(filename[filename.rfind('/')+1:filename.find('.smr')], stimpos=stimpos,
             eyepos=eyepos, time=stimtime, eye_vel=resting_vel_traces,
             mean_resting_vels=resting_mean_vels, rest_times=resting_times, metadata=meta, amplitudes=amplitudes) 
    
    with pd.ExcelWriter(filename[filename.rfind('/')+1:filename.find('.smr')]+('_step_velocities')+('.xlsx')) as writer:
        pd.DataFrame(resting_mean_vels).to_excel(writer, sheet_name='Resting Velocities')
        pd.Series(average_vel).to_excel(writer, sheet_name='Average Velocity')
        pd.DataFrame(resting_times).to_excel(writer, sheet_name='Time traces')
        pd.DataFrame(resting_vel_traces).to_excel(writer, sheet_name='Velocity traces')
        amplitudes.to_excel(writer, sheet_name='Sinus amplitudes')
        pd.DataFrame(resting_sin_traces[:,:,0]).to_excel(writer, sheet_name='0.1 Hz Position traces sinus')
        pd.DataFrame(resting_sin_times[:,:,0]).to_excel(writer, sheet_name='0.1 Hz Times sinus')
        pd.DataFrame(resting_sin_traces[:,:,1]).to_excel(writer, sheet_name='0.2 Hz Position traces sinus')
        pd.DataFrame(resting_sin_times[:,:,1]).to_excel(writer, sheet_name='0.2 Hz Times sinus')
        pd.DataFrame(resting_sin_traces[:,:,2]).to_excel(writer, sheet_name='0.5 Hz Position traces sinus')
        pd.DataFrame(resting_sin_times[:,:,2]).to_excel(writer, sheet_name='0.5 Hz Times sinus')

    return resting_mean_vels, resting_vel_traces, resting_times, resting_sin_traces, resting_sin_times  

def latency_approximation(time_segments, stimtime, stimpos, eyepos, 
                          initial_indices, resamp_rate, analysis_delay):
    '''Given time egments of selected eye motion responses, plots these segments
    and corresponding stimulus traces for a defined window. Allows manual selection
    of location where eye motion begins following stimulus. Calculates latency
    based on selection relative to stimulus onset. Returns latency calculations
    and mean latency.'''
    
    print('Click on plot where following eye motion begins. Press enter after selection.')
    print('If no eye motion change appears to occur, click prior to stimulus onset to indicate no latency.')
    
    time_before = 5 
    time_after = 4
    
    delay = int(resamp_rate*analysis_delay)
    
    sample_index = []
    latencies = []
    latencies_labels = []
    
    for q in time_segments[0]:
        if np.isnan(q):
            sample_index.append(np.nan)
            
        else:
            sample_index.append(np.where(stimtime == q)[0][0])

    for i in range(len(sample_index)):
        if ~np.isnan(sample_index[i]):
            f, ax = plt.subplots() 
            f.suptitle('Stimulus Motion Eye Responses', fontsize=14, ha='center')
            mngr = plt.get_current_fig_manager()
            # Places plot into the upper left corner for example:
            mngr.window.setGeometry(1,31,1400, 840)
            ax.plot(stimtime[int(sample_index[i]-resamp_rate*time_before):int(sample_index[i]+resamp_rate*time_after)], 
                  stimpos[int(sample_index[i]-resamp_rate*time_before):int(sample_index[i]+resamp_rate*time_after)], color='gray')
            axr2 = ax.twinx()
            axr2.plot(stimtime[int(sample_index[i]-resamp_rate*time_before):int(sample_index[i]+resamp_rate*time_after)], 
                 eyepos[int(sample_index[i]-resamp_rate*time_before):int(sample_index[i]+resamp_rate*time_after)])
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Eye motion (°)')
            ax.set_title('Onset Index' + ' ' + str(i))         
            x_y_click_coordinates = []
            plt.pause(0.3)
            x_y_click_coordinates = plt.ginput(-1, 0)
            plt.close()
            selected_respone_onset = x_y_click_coordinates[-1][0]
            stim_onset = stimtime[sample_index[i]-delay] 
            latency = selected_respone_onset - stim_onset
            if latency > 0:
                latencies.append(latency)
            else:
                latency = np.nan
                latencies.append(latency)
            latencies_labels.append('Response ' + str(i+1))
        else:
            latencies.append(np.nan)
            latencies_labels.append('Response ' + str(i+1))
            
    latencies.append(np.nanmean(np.array(latencies))) 
    latencies_labels.append('Mean Latency Response')
    latencies = pd.Series(latencies, index=latencies_labels, name='Latencies') 
   
    return latencies 

def select_bouts(current_indices, stimtime, stimpos, eyepos, resamp_rate, fp_in_iteration):
    '''Given indices of stimulus onset, eye motion, stimulus time, and and stimulus
       'position, this function plots eye position over time for a specified period
       starting from each stimulus bout onset. Plotting requires input if one
       wishes to accept eye motion during bouts for analysis, or to exclude. 
       Returns choice of accept or exclude.'''
    time_before = 5
    time_after = 10
    
    f, ax = plt.subplots(1,2) 
    f.suptitle('Stimulus Motion Eye Responses', fontsize=14, ha='center')
    mngr = plt.get_current_fig_manager()
    # Places plot into the upper left corner for example:
    mngr.window.setGeometry(1,31,1400, 840)
    ax[1].plot(stimtime[current_indices], 
            eyepos[current_indices]-np.mean(eyepos[current_indices], axis=0),
            label='Eye Motion Onset Segment') 
    #adapt this to plot both eyes, not just one
    #plt.plot(eyepos_both-np.mean(eyepos_both.transpose(), axis=1))
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Eye motion (°)')
    ax[1].set_title('Onset Index' + ' ' + str(i) + ' ' + '(zoom in)')
    axr = ax[1].twinx()
    axr.plot(stimtime[current_indices], 
        stimpos[current_indices], color='gray',
        label='Stimulus Position')
    ax[1].legend(loc=3)
    
    ax[0].plot(stimtime[current_indices[0]-time_before*resamp_rate:current_indices[0]+time_after*resamp_rate], 
        eyepos[current_indices[0]-time_before*resamp_rate:current_indices[0]+time_after*resamp_rate]-
        np.nanmean(eyepos[current_indices[0]-time_before*resamp_rate:current_indices[0]+time_after*resamp_rate], axis=0))
    ax[0].plot(stimtime[current_indices], 
        eyepos[current_indices]-np.nanmean(eyepos[current_indices[0]-time_before*resamp_rate:
        current_indices[0]+time_after*resamp_rate], axis=0),
        linewidth=3, label='Eye Motion Onset Segment')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Eye motion (°)')
    ax[0].set_title('Onset Index' + ' ' + str(i))
    ax[0].set_ylim(-10, 10)
    axr2 = ax[0].twinx()
    axr2.plot(stimtime[current_indices[0]-time_before*resamp_rate:current_indices[0]+time_after*resamp_rate], 
        stimpos[current_indices[0]-time_before*resamp_rate:current_indices[0]+time_after*resamp_rate], color='gray',
        label='Stimulus Position')
    
    if len(fp_in_iteration)>0: 
        ax[1].text(0.5, 0.8, 'WITHIN FAST PHASE', horizontalalignment='center',
            verticalalignment='center', transform=ax[1].transAxes, color='red', size=20) 
    plt.pause(0.3)
    keep = input('Keep this segment? Enter to keep, f key to remove false detection movement, press any other key to keep')
    plt.close()
    return keep

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

def stimulusonset(stimulus_velocity_trace, resample_rate):
    '''Given the stimulus velocity array and the resample rate, this 
    function determines the location where stimulus bouts begins for each
    stimulus direction (leftward or rightward visual motion). Returns
    index of stimulus bout locations.''' 
    #Determines instances when stimulus is moving toward the left
    #and finds stimulus bout onsets
    velocityleftward = []
    for i in range(len((stimulus_velocity_trace))):
        #100 is a hard-coded cutoff for stimulus velocity, as nothing
        #should  be remotely close to this value
        if stimulus_velocity_trace[i] < 0 and stimulus_velocity_trace[i] > -100: 
            velocityleftward.append(i)       
    onset_velocityleftward = []  
    for i in velocityleftward: 
        window_first = []
        #Points equaling 1/10th of the sampling rate used for sliding window
        #to determine if points in window have motion of stimulus
        for q in range(i-resample_rate//10,i): 
            window_first.append(q)
        if np.array(stimulus_velocity_trace[window_first]).mean() == 0:
            onset_velocityleftward.append(i)
    #Determines instances when stimulus is moving toward the right
    #and finds stimulus bout onsets
    velocityrightward = []
    for i in range(len((stimulus_velocity_trace))):
        if stimulus_velocity_trace[i] > 0 and stimulus_velocity_trace[i] < 100:
            velocityrightward.append(i)   
    onset_velocityrightward = []  
    for i in velocityrightward:
        window_first = []
        #Points equaling 1/10th of the sampling rate used for sliding window
        #to determine if points in window have motion of stimulus
        for q in range(i-resample_rate//10,i): #taking a fraction samp rate
            window_first.append(q)
        if np.array(stimulus_velocity_trace[window_first]).mean() == 0:
            onset_velocityrightward.append(i)  
    #Generate arrays for returning
    onset_velocityleftward = np.array(onset_velocityleftward)
    onset_velocityrightward = np.array(onset_velocityrightward)  
    velocityleftward = np.array(velocityleftward)
    velocityrightward = np.array(velocityrightward)
    
    return onset_velocityleftward, onset_velocityrightward, velocityleftward, velocityrightward

def resting_vel(stimulus_onsets, stimtime, stimpos, eyepos):
    
    prestim_velocities = np.zeros(len(stimulus_onsets))
    
    loop_iter = 0
    real_onsets = []
    
    for cur_onset in stimulus_onsets:
        current_indices = np.arange(cur_onset-initial_indices, cur_onset)
        expanded_indices = np.arange(cur_onset-initial_indices-600, cur_onset+600)
        expanded_indices_far = np.arange(cur_onset-initial_indices-1200, cur_onset+2000)
        current_positions = eyepos[current_indices]
        current_velocity = np.diff(current_positions)
        current_time = stimtime[current_indices]
        current_stimpos = stimpos[current_indices]
        
        f, ax = plt.subplots(1,2) 
        f.suptitle('pre-stimulus eye position', fontsize=14, ha='center')
        mngr = plt.get_current_fig_manager()
        # Places plot into the upper left corner for example:
        mngr.window.setGeometry(1,31,1400, 840)
        ax[0].plot(stimtime[expanded_indices], eyepos[expanded_indices], color=[0.7, 0.7, 0.7],
                label='Eye Motion Onset Segment')
        ax[0].plot(current_time, current_positions, color=[0.7, 0, 0],
                label='Eye Motion Onset Segment') 
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Eye position (°)')
        ax[0].set_title('zoom in')
        axr = ax[0].twinx()
        axr.plot(current_time, current_stimpos, color='gray',
            label='Stimulus Position')
        ax[0].legend(loc=3)
        
        ax[1].plot(stimtime[expanded_indices_far], eyepos[expanded_indices_far], color=[0.7, 0.7, 0.7],
                label='Eye Motion Onset Segment')
        ax[1].plot(current_time, current_positions, color=[0.7, 0, 0],
                label='Eye Motion Onset Segment')   
        ax[1].set_ylim(np.mean(current_positions)-10, np.mean(current_positions)+10)
        plt.pause(0.3)
        keep = input('Keep this segment? Enter to keep, f key to remove false detection movement, press any other key to keep')
        plt.close()
        
        if keep == '':
            prestim_velocities[loop_iter] = np.nanmean(current_velocity)
            real_onsets.append(loop_iter)
        elif keep == 'f':
            print('deleted false detection')
        else:
            real_onsets.append(loop_iter)   
            
        loop_iter+=1
        print(loop_iter)
        
    prestim_velocities = prestim_velocities[real_onsets]
        
    return prestim_velocities
    
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

def df_with_average(traces, name):
    traces = pd.DataFrame(traces)
    traces_avg = traces.mean(axis=1).rename(name) 
    traces_avg = traces_avg.subtract(traces_avg[0])
    traces = pd.concat([traces, traces_avg], axis=1) 
    return (traces)

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

#Due to irregularly sampled data, resampling of stimulus, eye position and time
stimtime, stimpos, eyeposition = resample(stim_position['Visual'], eye_position, resamp_rate) 

if section_cutout == 1:
    stimtime, stimpos, eyetime, eyeposition = data_subsection(stimtime, stimpos, stimtime, eyeposition)



#Eye motion filtering for eye motion. Only right eye examined.
stimposf = butfilt(stimpos, filtfreq, 4)
eyepos = np.vstack([butfilt(eyeposition[:,0], filtfreq, 4), 
                    butfilt(eyeposition[:,1], filtfreq, 4)]).transpose()

#Conversion of stimulus position into stimulus velocity
#For velocity to be appropiate in amplitudes, sampling rate of data is needed
samprate_vel = len(stimpos)/(stimtime[-1]-stimtime[0])
stim_vel = np.diff(stimpos.flatten())*samprate_vel

#Generate velocity and accleration profiles of right eye motion
eyevel = np.diff(eyepos, axis=0)*samprate_vel

#Find indicies of fast phase onsets during stimulus based on accleration
#thresholding
fastphase_indices = fastphase_finder(eyevel[:,0])
fastphase_indices = np.append(fastphase_indices, fastphase_finder(eyevel[:,1]))
fastphase_indices = np.unique(fastphase_indices)


#Find indicies of stimulus onset for bouts of stimulus velocities in the left
#ward and rightward directions
stim_onsets = stimulusonset(stim_vel, resamp_rate) #first two arrays are onsets
#Check to see if any stimulus is occuring. If it is not, resting eye position is
#analyzed and exported. Rest of the script is not executed.

if len(stim_onsets[2]) == 0 and len(stim_onsets[3]) == 0:
   resting_position(stimtime, eyepos, eyevel, initial_indices, resamp_rate)
   print('Resting eye velocity function executed. Excel sheet has been exported. Script ends here.')
   raise KeyboardInterrupt 

keeplist_leftward = []
keeplist_rightward = []

for n_eye in range(2): 
    #Eye motion segementation re: stimulus & calculations of velocity
    empty_bout = np.zeros([initial_indices,2])
    empty_bout[:] = np.nan 
    #Segmentation for Leftward stimulus motion
    eyepos_segments_leftward = [] 
    eyevel_segments_leftward = []
    time_segments_leftward = []
    #Extended position included the sampled velocity area but also datapoints around it
    #for plotting purposes
    extended_eyepos_segments_leftward = []
    extended_eyevel_segments_leftward = []
    extended_time_segments_leftward = []
    extended_stimpos_segments_leftward = []
    
    #List of kept indices per selected stimulus onset bout, excluding detected false onsets
    #for the purposes of metadata
    leftward_onsets = []
    
    if n_eye == 0:
        resting_vels_leftward = resting_vel(stim_onsets[0], stimtime, stimpos, eyepos[:,1])


    for ii in stim_onsets[0]:

        
        i = ii+omit_indices
        current_indices = np.arange(i,i+initial_indices)
        #If a fast phase is identified in the sampling period of eyemotion velocity
        #then the indices are returned where it overlaps with the sample period
        fp_in_iteration = []
        fp_in_iteration = np.intersect1d(fastphase_indices, current_indices)
        
        #get the position and velocity values from the indices, first for the full 
        # analysis period
        current_velocities = eyevel[current_indices]
        current_positions = eyepos[current_indices]
        current_time = stimtime[current_indices]
        
        #If a fast phase is detected, the data point prior to the fastphase is
        #used as the cutoff for sampling period, rather than the defined period
        if len(fp_in_iteration)>0: 
            current_indices = current_indices[0:np.where(current_indices == np.min(fp_in_iteration))[0][0]-pre_FP_indices]
       #Velocities and position values are made into NaNs starting at the onset
       #of the FP 
      
            if len(current_indices)<resamp_rate:
                current_velocities[:] = np.nan 
                current_positions[:] = np.nan
                current_time[:] = np.nan
                
            else:
                current_velocities[len(current_indices):] = np.nan 
                current_positions[len(current_indices):] = np.nan
                current_time[len(current_indices):] = np.nan       
        #Manual selection of eye motion responses during stimulation
        #Selection by plotting and returns result of selection 
        if n_eye == 0:
            keep = select_bouts(current_indices, stimtime, stimpos, eyepos, 
                        resamp_rate, fp_in_iteration)  
            keeplist_leftward.append(keep)
        elif n_eye == 1: 
            keeplist_leftward = np.array(keeplist_leftward)
            keep = keeplist_leftward[np.where(stim_onsets[0]==ii)[0][0]]
    
        #If eye motion bout is chosen to be kept
        if keep == '':    
            eyepos_segments_leftward.append(current_positions)
            eyevel_segments_leftward.append(current_velocities)
            time_segments_leftward.append(current_time)
            
            extended_eyepos_segments_leftward.append(eyepos[ii-int(export_delay*resamp_rate):i+initial_indices]) 
            extended_eyevel_segments_leftward.append(eyevel[ii-int(export_delay*resamp_rate):i+initial_indices])
            extended_time_segments_leftward.append(stimtime[ii-int(export_delay*resamp_rate):i+initial_indices])
            extended_stimpos_segments_leftward.append(stimpos[ii-int(export_delay*resamp_rate):i+initial_indices])
            leftward_onsets.append(ii)
            print('Kept eye motion segment.')
            
        #If eye motion bout is chosen to not be kept, then NaNs are appended
        elif keep == 'f': 
            print('False onset detection. Segement removed from analysis.')
            
        #If eye motion bout is chosen to not be kept, then NaNs are appended
        else: 
            eyepos_segments_leftward.append(empty_bout)
            eyevel_segments_leftward.append(empty_bout)
            time_segments_leftward.append(empty_bout[:,0])
            leftward_onsets.append(ii)
            print('Segment excluded from analysis.') 
        
        
    #turn the lists generated in the loop into arrays
    eyepos_segments_leftward = np.array(eyepos_segments_leftward).T   
    extended_eyepos_segments_leftward = np.array(extended_eyepos_segments_leftward).T
    eyevel_segments_leftward = np.array(eyevel_segments_leftward).T
    extended_eyevel_segments_leftward = np.array(extended_eyevel_segments_leftward).T
    time_segments_leftward = np.array(time_segments_leftward)
    extended_time_segments_leftward = np.array(extended_time_segments_leftward)
    extended_stimpos_segments_leftward = np.array(extended_stimpos_segments_leftward).T
    
    eyepos_segments_leftward = df_with_average(eyepos_segments_leftward[n_eye,:,:], 'Normalized Mean Trace')
        
    #generate a dataframe for the extended eye position segments, and generate an average trace
    extended_eyepos_segments_leftward = df_with_average(extended_eyepos_segments_leftward[n_eye,:], 'Normalized Mean Trace') 
    
    #turn the dataframe into an array again after transposing it?
    extended_eyevel_segments_leftward = df_with_average(extended_eyevel_segments_leftward[n_eye,:], 'Normalized Mean Trace') 
    
    #turn the time data into a dataframe with an average trace 
    extended_time_segments_leftward = df_with_average(extended_time_segments_leftward.transpose(), 'Normalized Mean Trace') 
    
    #turn the position lists into a dataframe, and add the average
    extended_stimpos_segments_leftward = df_with_average(extended_stimpos_segments_leftward, 'Normalized Mean Trace') 
    
    #Segmentation for Rightward stimulus motion
    eyepos_segments_rightward = []
    eyevel_segments_rightward = []
    time_segments_rightward = []
    #Extended position included the sampled velocity area but also datapoints around it
    #for plotting purposes
    extended_eyepos_segments_rightward = []
    extended_eyevel_segments_rightward = []
    extended_time_segments_rightward = []
    extended_stimpos_segments_rightward = []
    
    #List of kept indices per selected stimulus onset bout, excluding detected false onsets
    #for the purposes of metadata
    rightward_onsets = []
    
    if n_eye == 0:
        resting_vels_rightward = resting_vel(stim_onsets[1], stimtime, stimpos, eyepos[:,1])

    for ii in stim_onsets[1]:
    
    
        i = ii+omit_indices
        current_indices = np.arange(i,i+initial_indices)
         #If a fast phase is identified in the sampling period of eyemotion velocity
        #then the indices are returned where it overlaps with the sample period
        fp_in_iteration = []
        fp_in_iteration = np.intersect1d(fastphase_indices, current_indices)
        #If a fast phase is detected, the data point prior to the fastphase is
        #used as the cutoff for sampling period, rather than the defined period
    
        #get the position and velocity values from the indices, first for the full 
        # analysis period
        current_velocities = eyevel[current_indices]
        current_positions = eyepos[current_indices]
        current_time = stimtime[current_indices]
        
        if len(fp_in_iteration)>0: 
            #replace everything after potential fast phases with nans in the index, 
            #position and velocity array to at the end, have arrays of the same length
            current_indices = current_indices[0:np.where(current_indices == np.min(fp_in_iteration))[0][0]-pre_FP_indices]
            
            if len(current_indices)<resamp_rate:
                current_velocities[:] = np.nan 
                current_positions[:] = np.nan
                current_time[:] = np.nan
                
            else:
                current_velocities[len(current_indices):] = np.nan 
                current_positions[len(current_indices):] = np.nan
                current_time[len(current_indices):] = np.nan 
        
        if n_eye == 0:
            keep = select_bouts(current_indices, stimtime, stimpos, eyepos, 
                        resamp_rate, fp_in_iteration)  
            keeplist_rightward.append(keep)
        elif n_eye == 1: 
            keeplist_rightward = np.array(keeplist_rightward)
            keep = keeplist_rightward[np.where(stim_onsets[1]==ii)[0][0]]
            
        if keep == '':
            eyepos_segments_rightward.append(current_positions)
            eyevel_segments_rightward.append(current_velocities)
            time_segments_rightward.append(current_time)
            
            extended_eyepos_segments_rightward.append(eyepos[ii-int(export_delay*resamp_rate):i+initial_indices]) 
            extended_eyevel_segments_rightward.append(eyevel[ii-int(export_delay*resamp_rate):i+initial_indices])
            extended_time_segments_rightward.append(stimtime[ii-int(export_delay*resamp_rate):i+initial_indices])
            extended_stimpos_segments_rightward.append(stimpos[ii-int(export_delay*resamp_rate):i+initial_indices])
            rightward_onsets.append(ii)
            print('Kept eye motion segment.')
        #If eye motion bout is chosen to not be kept, then NaNs are appended
        elif keep == 'f': 
            print('False onset detection. Segement removed from analysis.')
            
        else: 
            eyepos_segments_rightward.append(empty_bout)
            eyevel_segments_rightward.append(empty_bout)
            time_segments_rightward.append(empty_bout[:,0])
            rightward_onsets.append(ii)
            print('Segment excluded from analysis.')
        
    eyepos_segments_rightward = np.array(eyepos_segments_rightward).T  
    eyevel_segments_rightward = np.array(eyevel_segments_rightward).T  
    time_segments_rightward = np.array(time_segments_rightward)    
    extended_time_segments_rightward = np.array(extended_time_segments_rightward)
    extended_eyepos_segments_rightward = np.array(extended_eyepos_segments_rightward).T
    extended_eyevel_segments_rightward = np.array(extended_eyevel_segments_rightward).T
    extended_stimpos_segments_rightward = np.array(extended_stimpos_segments_rightward).T
    
    eyepos_segments_rightward = df_with_average(eyepos_segments_rightward[n_eye,:,:], 'Normalized Mean Trace')

    
    #generate a dataframe for the extended eye position segments, and generate an average trace
    extended_eyepos_segments_rightward = df_with_average(extended_eyepos_segments_rightward[n_eye,:], 'Normalized Mean Trace') 
    
    #turn the dataframe into an array again after transposing it?
    extended_eyevel_segments_rightward = df_with_average(extended_eyevel_segments_rightward[n_eye,:], 'Normalized Mean Trace') 
    
    #turn the time data into a dataframe with an average trace 
    extended_time_segments_rightward = df_with_average(extended_time_segments_rightward.transpose(), 'Normalized Mean Trace') 
    
    #turn the position lists into a dataframe, and add the average
    extended_stimpos_segments_rightward = df_with_average(extended_stimpos_segments_rightward, 'Normalized Mean Trace') 


    #Calculate means of each velocity trace to get average velocities of eye motion
    eyevel_leftward_means = np.nanmean(eyevel_segments_leftward[n_eye,:,:], axis=0)
    eyevel_rightward_means = np.nanmean(eyevel_segments_rightward[n_eye,:,:], axis=0) 
    eyevel_leftward_means_short = np.nanmean(eyevel_segments_leftward[n_eye, 0:200,:], axis=0)
    eyevel_rightward_means_short = np.nanmean(eyevel_segments_rightward[n_eye,0:200,:], axis=0) 
    
    eye_velocities_titles = ['Leftward Velocity', 'Rightward Velocity', 
                             'Prestim vel leftward', 'Prestim vel rightward', 
                             'Corrected Leftward Velocity', 'Corrected Rightward Velocity'] 
    eye_velocities = np.stack((eyevel_leftward_means, eyevel_rightward_means, 
                               resting_vels_leftward, resting_vels_rightward, 
                               eyevel_leftward_means-resting_vels_leftward, 
                               eyevel_rightward_means-resting_vels_rightward), axis=1)
    #Generate mean velocity for
    mean_velocities = np.zeros([1,6])
    #Leftward motion velocities are absolute value to faciliate statistical and
    #plotting fair comparisons for mean velocity
    mean_velocities[:] = np.nanmean(eye_velocities, axis=0)
    #Conversion into a DataFrame for exporting into excel
    eye_velocities = pd.DataFrame(eye_velocities, columns=eye_velocities_titles) 
    mean_velocities = pd.DataFrame(mean_velocities, columns=eye_velocities_titles) 
    
    
    #Manual determination of latencies of eye motion following responses
    # If determination is wished to be used from input variable above 
    if latency_analysis == 1:
        leftward_latencies = latency_approximation(time_segments_leftward, stimtime, 
                            stimpos, eyepos, initial_indices, 
                            resamp_rate, analysis_delay)
        rightward_latencies = latency_approximation(time_segments_rightward, stimtime,
                            stimpos, eyepos, initial_indices, 
                            resamp_rate, analysis_delay)  
        leftward_latencies = leftward_latencies.rename('Leftward latencies') 
        rightward_latencies = rightward_latencies.rename('Rightward latencies') 
        latencies = pd.concat([leftward_latencies, rightward_latencies], axis=1)    
    else:
        #Manual latency selection not chose. Empty Series made with NaNs stead.
        leftward_latencies = pd.Series(index=range(5), name='Leftward latencies', dtype='float64')
        rightward_latencies = pd.Series(index=range(5), name='Rightward latencies', dtype='float64')
        latencies = pd.concat([leftward_latencies, rightward_latencies], axis=1)
    
    #Exporting of data analysis and metadata into npz python file and excel sheet
    #Version control tag for re-analysis 
    verNr = 1
    verTag = '_ver'+str(verNr)
    while os.path.isfile(filename[filename.rfind('/')+1:filename.find('.smr')]+verTag+('.xlsx')) == True:
       verNr = verNr+1
       verTag = '_ver'+str(verNr)
       
    #Indices and corresponding time values for stimulus and eye motion responses
    indices_time_meta = np.array([leftward_onsets, stimtime[leftward_onsets], rightward_onsets, stimtime[rightward_onsets]]).T
    indices_time_meta_names = ['Stim Bout Onset Indices Leftward', 
                               'Stim Bout Onset Times Leftward', 
                               'Stim Bout Onset Indices Rightward', 
                               'Stim Bout Onset Times Rightward']
    indices_time_meta = pd.DataFrame(indices_time_meta, columns=indices_time_meta_names)
       
    #Count the number of eye motion bouts used for metadata
    left_count = eye_velocities['Leftward Velocity'].count()
    right_count = eye_velocities['Rightward Velocity'].count()
    
    #Metadata with input variables and response bout counts
    metadata = [resamp_rate, filtfreq, filt_order, eyevel_seconds, initial_indices,
                left_count, right_count, analysis_delay, export_delay]
    metadata_names = ['Resample Rate', 'Filter Cutoff Frequency', 'Filter Order',
                      'Velocity Sampling Period', 'Velocity Sampling Datapoints Length', 
                      'Leftward Motion Bouts', 'Rightward Motion Bouts', 'Analysis delay time',
                      'Extended Trace Delay']
    meta = pd.Series(metadata, index=metadata_names)
    
    if n_eye == 0: 
        eye_tag = 'LE'
    elif n_eye == 1:
        eye_tag = 'RE'
    
    #Exporting data analysis into a npz python file with the following variables
    # np.savez(filename[filename.rfind('/')+1:filename.find('.smr')]+eye_tag+verTag, stimpos=stimpos,
    #          eyepos=eyepos, time=stimtime, eye_velocities=eye_velocities, metadata=meta,
    #          bout_indices_time=indices_time_meta, eyepos_segments_leftward=eyevel_segments_leftward,
    #          eyepos_segments_rightward=eyevel_segments_rightward, analysis_delay=analysis_delay) 
    
    with pd.ExcelWriter(filename[filename.rfind('/')+1:filename.find('.smr')]+('_step_velocities')+eye_tag+verTag+('.xlsx')) as writer:  
        pd.DataFrame(time_segments_leftward.T).to_excel(writer, sheet_name='Time traces leftward')
        pd.DataFrame(eyevel_segments_leftward[n_eye,:,:]).to_excel(writer, sheet_name='Velocity traces leftward')
        pd.DataFrame(time_segments_rightward.T).to_excel(writer, sheet_name='Time traces rightward')
        pd.DataFrame(eyevel_segments_rightward[n_eye,:,:]).to_excel(writer, sheet_name='Velocity traces rightward')
        eye_velocities.to_excel(writer, sheet_name='Velocities')
        mean_velocities.to_excel(writer, sheet_name='Mean velocities per direction')
        latencies.to_excel(writer, sheet_name='Latencies')
        meta.to_excel(writer, sheet_name='Metadata')
        indices_time_meta.to_excel(writer, sheet_name='Onset Indices and Time')  
        pd.DataFrame(extended_time_segments_leftward).to_excel(writer, sheet_name='Extended Time traces leftward')
        pd.DataFrame(extended_eyepos_segments_leftward).to_excel(writer, sheet_name='Extended Position leftward')
        pd.DataFrame(extended_stimpos_segments_leftward).to_excel(writer, sheet_name='Extended Stim Pos leftward')
        pd.DataFrame(extended_time_segments_rightward).to_excel(writer, sheet_name='Extended Time rightward')
        pd.DataFrame(extended_eyepos_segments_rightward).to_excel(writer, sheet_name='Extended Position rightward')
        pd.DataFrame(extended_stimpos_segments_rightward).to_excel(writer, sheet_name='Extended Stim Pos rightward')

#Plotting of Data
# f, ax = plt.subplots()
# ax.plot(stimtime[:-1], stim_vel, '-', label='Velocity', color='tab:orange')
# ax.plot(stimtime[stim_onsets[2]], stim_vel[stim_onsets[2]], 'o', label='Leftward Velocity', color='tab:green')
# ax.plot(stimtime[stim_onsets[3]], stim_vel[stim_onsets[3]], 'o', label='Rightward Velocity', color='tab:red')
# ax.plot(stimtime[stim_onsets[0]], stim_vel[stim_onsets[0]], 'o', label='Leftward Onset', color='tab:purple')
# ax.plot(stimtime[stim_onsets[1]], stim_vel[stim_onsets[1]], 'o', label='Rightward Onset', color='tab:cyan')
# axr = ax.twinx()
# axr.plot(stimtime, stimpos, label='StimPos', color='tab:gray')
# ax.set_title('Stimulus Position and Velocity Trace')
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Velocity (°/s)')
# axr.set_ylabel('Stimulus Position (°)')
# ax.legend()
# axr.legend()

# f2, ax2 = plt.subplots()
# ax2.plot(stimtime, eyepos)
# for i in stim_onsets[0]:
#     linelabel = 'Leftward Stim Bout' + ' ' + str((np.where(stim_onsets[0]==i)[0][0]+1))
#     ax2.plot(stimtime[i:i+initial_indices], eyepos[i:i+initial_indices],
#     linewidth=5, label=linelabel)
# for i in stim_onsets[1]:
#     linelabel = 'Rightward Stim Bout' + ' ' + str((np.where(stim_onsets[1]==i)[0][0]+1))
#     ax2.plot(stimtime[i:i+initial_indices], eyepos[i:i+initial_indices],
#     linewidth=5, label=linelabel)
# ax2.plot(stimtime[fastphase_indices], eyepos[fastphase_indices], 'o',
# markersize=8, color='red', label='Fast Phases')
# axr2 = ax2.twinx()
# axr2.plot(stimtime, stimpos, label='StimPos', color='tab:gray')
# ax2.legend()
# ax2.set_title('Eye Traces re Stimulus Onset')
# ax2.set_xlabel('Time (s)')
# ax2.set_ylabel('Degrees (°)')
