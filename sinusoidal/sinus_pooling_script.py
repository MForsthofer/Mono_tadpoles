#AG Straka Excel Pooling Script. This script can take excel sheets generated 
#from the stimulus segementation script and combines multiple of them
#test into one excel sheet with averages generated for animal pools

import pandas as pd
import os 
import matplotlib.pyplot as plt
import numpy as np

################################

def plot_mean_std(mean, traces, time, fnr, axnr, meanclr):
    """ Given mean eye trace and individual eye traces, calculates plots the 
    mean ± SD"""
    axnr.plot(time, mean, linewidth=2, color=meanclr)
    axnr.plot(time, mean+np.nanstd(traces, axis=1), color=meanclr, linestyle='-', linewidth=0.5)
    axnr.plot(time, mean-np.nanstd(traces, axis=1), color=meanclr, linestyle='-', linewidth=0.5)
    axnr.fill_between(time,mean+np.nanstd(traces, axis=1), mean-np.nanstd(traces, axis=1), color = meanclr, alpha=0.1)
  
def singlecycle_amp(ExcelFile, excelsheet, targetvar):
    """ gets the amplitudes of individual cycles of each animal"""
    single_cycles = pd.read_excel(ExcelFile, ExcelSheets[excelsheet])
    targetvar.append(np.nanmax(single_cycles.values, axis=0)[1:]-np.nanmin(single_cycles.values, axis=0)[1:])
    return targetvar

def calc_phase(stim_df, pos_df, time_df):
    cyclecount = np.arange(0, len(time_df.transpose()))

    eyelows = np.argmin(pos_df.values, axis=0)
    stimlows = np.argmin(stim_df.values, axis=0)
    phase_t1 = time_df.values[eyelows, cyclecount]-time_df.values[stimlows,cyclecount]
    phase_t1_deg = phase_t1*360/np.max(time_df.values)

    eyemax = np.argmax(pos_df.values, axis=0)
    stimmax = np.argmax(stim_df.values, axis=0)
    phase_t2 = time_df.values[stimmax,cyclecount]-time_df.values[eyemax, cyclecount]
    phase_t2_deg = phase_t2*360/np.max(time_df.values) 
    return (phase_t1_deg, phase_t2_deg)
################################
    
#Global Input Variables
#Type out below the experimental condition unifying all the animals
Experiment = 'Pooled_sinus_one-eye_01Hz_Gr255.xlsx'  

#Do you wish to look at first cycles? If yes, then first cycles will be pooled
#together.

FirstCycles = 0 #0 for no, 1 for yes

################################

#Plotting variables 

# color for single cycles in the plots. In RGB values [Red, Green, Blue]. [0,0,0]=black, [1,1,1]=white
color=color_single_cycles = [0.5, 0.5, 0.5] 
#color vor the average cycle plotted onto the individual cycles
color_average_cycle = 'black'
#color for the mean trace and +-STD
color_average_trace = 'pink'
#do you want to use manually set y axis limits= Select 'yes' or 'no'
use_manual_y_axis_limits = 'no'
#here you can set manual y axis limits. [lower limit, upper limit]:
ylims_new = [-8, 1]
    
#Path will be where your working directory is at. Be sure the working directory
#is where the excel sheets you want to combine are!
Path = os.listdir()

################################

#Creates temporary lists to store the pooled "p" animal data
pEyes = []
pTime = []
pStim = []
pResultsBoth = []
pResultsLE = []
pResultsRE = []
pPhase = []
Animals = [] 
p1Cycles = []
p1Results = []
pReye =  []
pLeye = []
nCycles = []

amplitudes_LE = []
amplitudes_RE = []

#Iterates through each excel sheet and pulls the relevant parameter values
#the values are then saved in the lists above
for i in Path:
    if i.endswith('.xlsx'):
        ExcelFile = pd.ExcelFile(i)
        ExcelSheets = ExcelFile.sheet_names
        
        #Measure amplitudes of individual cycles throughout stimulatio
        #for left and right eye respectively
        singlecycle_amp(ExcelFile, 3, amplitudes_LE)
        singlecycle_amp(ExcelFile, 4, amplitudes_RE)


        #Iterate through the values for Time, Stim and EyePos 
        #Index in following line corresponds to appropiate tab in excel sheet
        Traces = pd.read_excel(ExcelFile, ExcelSheets[7]) 
        NormMeanBothEyes = Traces['Mean Both Eyes Filtered']
        pEyes.append(NormMeanBothEyes) 
        MeanTime = Traces['Mean Time']
        pTime.append(MeanTime)
        MeanStim = Traces['Mean Stim']
        pStim.append(MeanStim)  
        
        pLeye.append(Traces['Mean Left Eye Interpolated & Filtered'])
        pReye.append(Traces['Mean Right Eye Interpolated & Filtered'])
        
        #Iterate through results of Gain and Phase Values From Mean Traces
        #Index in following line corresponds to appropiate tab in excel sheet
        Results = pd.read_excel(ExcelFile, ExcelSheets[10]) 
        Res_Both = Results['both eyes filtered']
        Res_LE = Results['left eye filtered']
        Res_RE = Results['right eye filtered']
        pResultsBoth.append(Res_Both)
        pResultsLE.append(Res_LE)
        pResultsRE.append(Res_RE)
        
        #If animals had VOR responses during first cycles
        #Iterate through the values for 1st Cycle Analysis
        #Index in following line corresponds to appropiate tab in excel sheet
        if FirstCycles == 1:
        #If an animal has no first cycles, arrays with NaNs are made to keep
        #animal information in the excel sheet
            if len(ExcelSheets) <= 11:
                Traces1 = np.zeros((len(NormMeanBothEyes), 3))
                Traces1[:] = np.nan
                Traces1 = pd.DataFrame(Traces1)
                NormMeanBothEyes1 = Traces1[0]
                TimeBothEyes1 = Traces1[0]
                p1Cycles.append(NormMeanBothEyes1)
                
                Results1 = np.zeros((3,2))
                Results1[:] = np.nan
                Results1 = pd.DataFrame(Results1)
                Res_Both1 = Results1[0]
                p1Results.append(Res_Both1)
                
                
        #Otherwise, if the animal has first cycle info, it is pulled from the
        #spreadsheet
            else:
                Traces1 = pd.read_excel(ExcelFile, ExcelSheets[12]) 
                NormMeanBothEyes1 = Traces1['normalized mean trace 1stcycles']
                TimeBothEyes1 = Traces1['Unnamed: 0']
                p1Cycles.append(NormMeanBothEyes1)
                #Iterate through the values for Time, Stim and EyePos 
                #Index in following line corresponds to appropiate tab in excel sheet
                Results1 = pd.read_excel(ExcelFile, ExcelSheets[11])
                Res_Both1 = Results1['both eyes filtered']
                p1Results.append(Res_Both1)
            
        #Gathers the name of the animal/data file
        Animals.append(i)
        nCycles.append(pd.read_excel(ExcelFile, ExcelSheets[8]).iloc[0,1])
        
#Converts pooled animal lists into grouped "gr" DataFrames/Series
grEye = (pd.DataFrame(pEyes, index=Animals)).T 
grLeye = (pd.DataFrame(pLeye, index=Animals)).T
grReye = (pd.DataFrame(pReye, index=Animals)).T   
grTime = (pd.DataFrame(pTime, index=Animals)).T 
grStim = (pd.DataFrame(pStim, index=Animals)).T 
re_phase1, re_phase2 = calc_phase(grStim, grReye, grTime)
le_phase1, le_phase2 = calc_phase(grStim, grLeye, grTime)
grResultsBoth = (pd.DataFrame(pResultsBoth, index=Animals)).T  
grResultsBoth = grResultsBoth.rename(Results['Unnamed: 0'])
grResultsLE = (pd.DataFrame(pResultsLE, index=Animals)).T 
grResultsLE.loc[8,:] = le_phase1
grResultsLE.loc[9,:] = le_phase2
grResultsLE = grResultsLE.rename(Results['Unnamed: 0'])
grResultsRE = (pd.DataFrame(pResultsRE, index=Animals)).T 
grResultsRE.loc[8,:] = re_phase1
grResultsRE.loc[9,:] = re_phase2
grResultsRE = grResultsRE.rename(Results['Unnamed: 0'])

#calculate phase for both directions




if FirstCycles == 1:
    result_names = pd.Series(['amplitude 1st half cycle', 'gain 1st half cycle', 'phase (s)'])
    grResults1 = (pd.DataFrame(p1Results, index=Animals)).T 
    grResults1 = grResults1.rename(result_names) 
    grEye1c = (pd.DataFrame(p1Cycles, index=Animals)).T
grCyclenum = pd.Series(nCycles)  
 
#Computes the mean "Avg" from all parameters for each animal
grEyeAvg = (grEye.mean(axis=1)).rename('Average EyePosition')
grLeyeAvg = (grLeye.mean(axis=1)).rename('Average LeftEyePosition')
grReyeAvg = (grReye.mean(axis=1)).rename('Average RightEyePosition')
grTimeAvg = (grTime.mean(axis=1)).rename('Average Time')
grStimAvg = (grStim.mean(axis=1)).rename('Average Stimulus')
grResBothAvg = (grResultsBoth.mean(axis=1)).rename('Avg Gain Both Eyes')
grResLEAvg = (grResultsLE.mean(axis=1)).rename('Avg Gain Left Eye')
grResREAvg = (grResultsRE.mean(axis=1)).rename('Avg Gain Right Eye') 
if FirstCycles == 1: 
    grResults1Avg = (grResults1.mean(axis=1)).rename('Avg Results 1st Cycle')
    grEye1cAvg = (grEye1c.mean(axis=1)).rename('Avg 1st Cycle Traces')

#Computes the standard deviation "SD" from all parameters for each animal
grEyeSD = ((grEye.std(axis=1)).rename('SD eye position both'))
grLeyeSD = ((grLeye.std(axis=1)).rename('SD eye position left'))
grReyeSD = ((grReye.std(axis=1)).rename('SD eye position right'))


#Combines individual animal values and mean values in one DataFrame
EyePos = pd.concat([grEye, grEyeAvg, grEyeSD], axis=1) 
LEyePos = pd.concat([grLeye, grLeyeAvg, grLeyeSD], axis=1) 
REyePos = pd.concat([grReye, grReyeAvg, grReyeSD], axis=1) 
Time = pd.concat([grTime, grTimeAvg], axis=1) 
Stim = pd.concat([grStim, grStimAvg], axis=1)  
ResultBoth = pd.concat([grResultsBoth, grResBothAvg], axis=1)
ResultLE = pd.concat([grResultsLE, grResLEAvg], axis=1)
ResultRE = pd.concat([grResultsRE, grResREAvg], axis=1)
if FirstCycles == 1:
    Result1C = pd.concat([grResults1, grResults1Avg], axis=1)
    FirstCycle = pd.concat([grEye1c, grEye1cAvg], axis=1)

#Puts the Cycle Counts per animal into a pandas Series for exporting
CycleCounts = pd.Series(nCycles, index=Animals)
CountLabel = ['Max', 'Min']
CountMinMax = [CycleCounts.max(), CycleCounts.min()]
CountMinMax = pd.Series(CountMinMax, index=CountLabel)
CycleCounts = CycleCounts.append(CountMinMax)


#Exports the combined individual and mean values as an Excel sheet
with pd.ExcelWriter(Experiment) as writer:  
    EyePos.to_excel(writer, sheet_name='Eye Position Both Eyes', index=False)
    LEyePos.to_excel(writer, sheet_name='Eye Position Left Eye', index=False)
    REyePos.to_excel(writer, sheet_name='Eye Position Right Eye', index=False)
    Time.to_excel(writer, sheet_name='Time', index=False)
    Stim.to_excel(writer, sheet_name='Stimulus', index=False)
    ResultBoth.to_excel(writer, sheet_name='Results Both Eyes')
    ResultLE.to_excel(writer, sheet_name='Results Left Eye')
    ResultRE.to_excel(writer, sheet_name='Results Right Eye')
    if FirstCycles == 1: 
        FirstCycle.to_excel(writer, sheet_name='1st Cycle Traces Both Eyes')
        Result1C.to_excel(writer, sheet_name='1st Cycle Results')
    CycleCounts.to_excel(writer, sheet_name='Cycle Counts', header=False)
    

################################
    
#Plots the Pooled Traces and Data

xlim_pos = [np.min(grTimeAvg), np.max(grTimeAvg)]

f1, ax1 = plt.subplots(figsize=(7,4))
ax1.plot(grTime, grLeye, color=color_single_cycles)
ax1.plot(grTimeAvg, grLeyeAvg, color=color_average_cycle)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Position (°)')
ax1.set_xlim(xlim_pos)
# ax3.set_ylim(ylim_pos)
ax1_1 = ax1.twinx()
ax1_1.plot(grTime, grStim, color=[0.8,0.8,0.8], linewidth=1, linestyle='--')
ax1_1.set_ylabel('Stim position (°)')
# ax3_1.set_ylim(stimlim)
ax1.set_title('Left eye single traces & mean')
ax1.text(0.85, 0.05, 'N = '+str(len(grLeye.transpose()))+', n = ' + 
str(np.min(nCycles)) + '-' + str(np.max(nCycles)), horizontalalignment='center'
,verticalalignment='center', transform=ax1.transAxes, size=12)

f2, ax2 = plt.subplots(figsize=(7,4))
ax2.plot(grTime, grReye, color=color_single_cycles)
ax2.plot(grTimeAvg, grReyeAvg, color=color_average_cycle)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Position (°)')
ax2.set_xlim(xlim_pos)
# ax3.set_ylim(ylim_pos)
ax2_1 = ax2.twinx()
ax2_1.plot(grTime, grStim, color='grey', linewidth=1, linestyle='--')
ax2_1.set_ylabel('Stim position (°)')
# ax3_1.set_ylim(stimlim)
ax2.set_title('Right eye single traces & mean')
ax2.text(0.85, 0.05, 'N = '+str(len(grLeye.transpose()))+', n = ' + 
str(np.min(nCycles)) + '-' + str(np.max(nCycles)), horizontalalignment='center'
,verticalalignment='center', transform=ax1.transAxes, size=12)

f3, ax3 = plt.subplots(figsize=(7,4))
ax3.plot(grTime, grEye, color=color_single_cycles)
ax3.plot(grTimeAvg, grEyeAvg, color=color_average_cycle)
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Position (°)')
ax3.set_xlim(xlim_pos)
# ax3.set_ylim(ylim_pos)
ax3_1 = ax3.twinx()
ax3_1.plot(grTime, grStim, color='grey', linewidth=1, linestyle='--')
ax3_1.set_ylabel('Stim position (°)')
# ax3_1.set_ylim(stimlim)
ax3.set_title('Both eyes single traces & mean')
ax3.text(0.85, 0.05, 'N = '+str(len(grLeye.transpose()))+', n = ' + str(np.min(nCycles)) + '-' + str(np.max(nCycles)), horizontalalignment='center',verticalalignment='center', transform=ax1.transAxes, size=12)

f4, ax4 = plt.subplots(figsize=(7,4))
plot_mean_std(grLeyeAvg, grLeye, grTimeAvg, f4, ax4, color_average_trace)
ax4_1 = ax4.twinx()
ax4_1.plot(grTimeAvg, grStimAvg, color='grey', linewidth=1, linestyle='--')
#ax4_1.set_ylim(stimlim)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Position (°)')
ax4.set_xlim(xlim_pos)
# ax4.set_ylim(ylim_pos)
ax4.set_title('Mean left eye ±SD')
ax4.text(0.85, 0.05, 'N = '+str(len(grLeye.transpose()))+', n = ' + str(np.min(nCycles)) + '-' + str(np.max(nCycles)), horizontalalignment='center',verticalalignment='center', transform=ax1.transAxes, size=12)

f5, ax5 = plt.subplots(figsize=(7,4))
plot_mean_std(grReyeAvg, grReye, grTimeAvg, f5, ax5, color_average_trace)
ax5_1 = ax5.twinx()
ax5_1.plot(grTimeAvg, grStimAvg, color='grey', linewidth=1, linestyle='--')
#ax4_1.set_ylim(stimlim)
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Position (°)')
ax5.set_xlim(xlim_pos)
# ax4.set_ylim(ylim_pos)
ax5.set_title('Mean right eye ±SD')
ax5.text(0.85, 0.05, 'N = '+str(len(grLeye.transpose()))+', n = ' + str(np.min(nCycles)) + '-' + str(np.max(nCycles)), horizontalalignment='center',verticalalignment='center', transform=ax1.transAxes, size=12)


f6, ax6 = plt.subplots(figsize=(7,4))
plot_mean_std(grEyeAvg, grEye, grTimeAvg, f6, ax6, color_average_trace)
ax6_1 = ax6.twinx()
ax6_1.plot(grTimeAvg, grStimAvg, color='grey', linewidth=1, linestyle='--')
#ax4_1.set_ylim(stimlim)
ax6.set_xlabel('Time (s)')
ax6.set_ylabel('Position (°)')
ax6.set_xlim(xlim_pos)
# ax4.set_ylim(ylim_pos)
ax6.set_title('Both eyes ±SD')
ax6.text(0.85, 0.05, 'N = '+str(len(grLeye.transpose()))+', n = ' + str(np.min(nCycles)) + '-' + str(np.max(nCycles)), horizontalalignment='center',verticalalignment='center', transform=ax1.transAxes, size=12)

#Plots the average cycles for each animal as normalized position of left and
#right eye versus eachother. Plots gives an idea for how coordinated the eyes 
#are.
f7, ax7 = plt.subplots()
for (i, p) in zip(pLeye, pReye):
    ax7.plot(i, p, '.', markersize=2)
ax7.set_xlabel('Normalized Left Eye Position (°)')
ax7.set_ylabel('Normalized Right Eye Position (°)')
ax7.set_title('Normalized Eye Position XY Coordinate Plot')

if use_manual_y_axis_limits == 'no':
        
    #adjust y limits of the plots to be identical
    #step 1: get current ylims of all plots and find the maximum range
    ylims = []
    ylims.append(ax1.get_ylim())
    ylims.append(ax2.get_ylim())
    ylims.append(ax3.get_ylim())
    ylims.append(ax4.get_ylim())
    ylims.append(ax5.get_ylim())
    ylims.append(ax6.get_ylim())
    ylims = np.array(ylims)
    ylims_new = [np.min(ylims[:,0]), np.max(ylims[:,1])]
    
#step 2: apply that range to all plots
ax1.set_ylim(ylims_new)
ax2.set_ylim(ylims_new)
ax3.set_ylim(ylims_new)
ax4.set_ylim(ylims_new)
ax5.set_ylim(ylims_new)
ax6.set_ylim(ylims_new)

f8, ax8 = plt.subplots()     
std_devs_LE = np.nanstd(np.array(amplitudes_LE), axis=1)
std_devs_RE = np.nanstd(np.array(amplitudes_RE), axis=1)
ax8.plot(np.array(amplitudes_LE).transpose(), color='b')
ax8.plot(np.array(amplitudes_RE).transpose(), color='r')
ax8.set_xlabel('Cycle #')
ax8.set_ylabel('Amplitude')
ax8.set_xlim(1,31)
ax8.set_ylim(0,20)


#optional, but recommended:
ax7.text(0.2, 0.1, 'MICHI IS GREAT', transform=ax1.transAxes, size=36, color='red', rotation=40)
