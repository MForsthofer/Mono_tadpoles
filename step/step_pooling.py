#AG Straka Excel Pooling Script. This script can take excel sheets generated 
#from the stimulus segementation script and combines multiple of them
#test into one excel sheet with averages generated for animal pools

import pandas as pd
import os 
import matplotlib.pyplot as plt
import numpy as np

################################

def mean_SD_pooling_appending(pooled_values, animal_index):
    '''Given a list of parameter values from animals to be pooled, calculates
    the mean and standard deviation across the animals and returns
    as a pandas DataFrame with animal values and mean and standard deviation'''
    pooled_values = np.array(pooled_values)
    avg = np.nanmean(pooled_values, axis=0)  
    std = np.nanstd(pooled_values, axis=0)  
    grouped_values = (pd.DataFrame(pooled_values, index=animal_index)).T
    avg = pd.Series(avg, name='Average')
    std_dev = pd.Series(std, name='Standard Deviation')
    trace_with_avg_std = pd.concat([grouped_values, avg, std_dev], axis=1) 
    return trace_with_avg_std 

def plot_step_pool(time, eyepos, stim, animals, direction):
    ''' Given pooled time arrays and eyeposition arrays and animal names,
    function will plot average traces as well as mean and SD. Returns nothing.
    Direction variable is for title purposes.'''
    import matplotlib.pyplot as plt
    
    if direction == 'Left':
        suptitle = 'Leftward Stimulus Motion'
    else:
        suptitle = 'Rightward Stimuus Motion'
        
    avg = 'Average'
    std = 'Standard Deviation'
    
    f, ax = plt.subplots(1,2)
    ax[0].plot(time[animals], eyepos[animals], color='gray')
    ax[0].plot(time[avg], eyepos[avg], color='teal', linewidth=2, label='Mean Response')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Normalized Eye Position')
    ax[0].set_title('Individual Traces and Mean')
    ax2 = ax[0].twinx()
    ax2.plot(time[avg], stim[avg], color='black', label='Stimulus Position')
    f.suptitle(suptitle) 
    ax[0].legend()
    #Fill between functoin allows plotting of the SD and filling the space
    #between
    ax[1].plot(time[avg], eyepos[avg], color='teal', label='Mean Response')
    ax[1].fill_between(time[avg], eyepos[avg]+eyepos[std], eyepos[avg]-eyepos[std],
                       facecolor='teal', alpha=0.3)
    ax[1].legend()
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Normalized Eye Position')
    ax[1].legend()
    ax[1].set_title('Mean Eye Position and SD')
    ax3 = ax[1].twinx()
    ax3.plot(time[avg], stim[avg], color='black', label='Stimulus Position')
    ax[1].legend()
    return 

################################
    
#Global Input Variables
#Type out below the experimental condition unifying all the animals
Experiment = 'Pooled_1eye_2ndrep_Visual_Step_255.xlsx'   
#Are the excel sheets you're pooling from no-stimulus motion experiments?
no_stimulus_motion = 0 #0 for no, 1 for yes


################################

#Path will be where your working directory is at. Be sure the working directory
#is where the excel sheets you want to combine are!
Path = os.listdir()

################################

#Creates temporary lists to store the pooled "p" animal data
pLeftVel = []
pRightVel = []
pLeftLat = []
pRightLat = []
pLeftBouts = []
pRightBouts = []
pEyePos_Le = []
pEyePos_Ri = []
pTime_Le = []
pTime_Ri = []
pStim_Le = []
pStim_Ri = []
Animals = [] 
pRest_Vel = []

#Iterates through each excel sheet and pulls the relevant parameter values
#the values are then saved in the lists above
for i in Path:
    if i.endswith('.xlsx') and no_stimulus_motion == 0:
        ExcelFile = pd.ExcelFile(i)
        ExcelSheets = ExcelFile.sheet_names 

        #Iterate through the values for Time, EyePosition and Stimulus 
        #Index in following line corresponds to appropiate tab in excel sheet
        pEyePos_Le.append(pd.read_excel(ExcelFile, ExcelSheets[10])['Normalized Mean Trace'])
        pEyePos_Ri.append(pd.read_excel(ExcelFile, ExcelSheets[13])['Normalized Mean Trace'])
        
        pStim_Le.append(pd.read_excel(ExcelFile, ExcelSheets[11])['Normalized Mean Trace'])
        pStim_Ri.append(pd.read_excel(ExcelFile, ExcelSheets[14])['Normalized Mean Trace'])
        
        pTime_Le.append(pd.read_excel(ExcelFile, ExcelSheets[9])['Normalized Mean Trace'])
        pTime_Ri.append(pd.read_excel(ExcelFile, ExcelSheets[12])['Normalized Mean Trace'])
        
        #Iterate through the results for mean velocities
        #Index in following line corresponds to appropiate tab in excel sheet
        velocities = pd.read_excel(ExcelFile, ExcelSheets[5]) 
        vel_leftward = velocities['Corrected Leftward Velocity']
        vel_rightward = velocities['Corrected Rightward Velocity']
        pLeftVel.append(vel_leftward[0])
        pRightVel.append(vel_rightward[0])
        
        #Iterate through the results for latencies
        #Index in following line corresponds to appropiate tab in excel sheet
        latencies = pd.read_excel(ExcelFile, ExcelSheets[6]) 
        lat_leftward = latencies['Leftward latencies'].iloc[-1]
        lat_rightward = latencies['Rightward latencies'].iloc[-1] 
        pLeftLat.append(lat_leftward) 
        pRightLat.append(lat_rightward)
        
        #Iterate through the results for latencies
        #Index in following line corresponds to appropiate tab in excel sheet
        metadata = pd.read_excel(ExcelFile, ExcelSheets[7]) 
        left_bouts = metadata.iloc[5]
        right_bouts = metadata.iloc[6]
        pLeftBouts.append(left_bouts[0])
        pRightBouts.append(right_bouts[0])
            
        #Gathers the name of the animal/data file
        Animals.append(i)

    elif i.endswith('.xlsx'):
        ExcelFile = pd.ExcelFile(i)
        ExcelSheets = ExcelFile.sheet_names  
        
        pRest_Vel.append(pd.read_excel(ExcelFile, ExcelSheets[1])[0].rename(str(i)))
        
        #Gathers the name of the animal/data file
        Animals.append(i)

#Taken the pooled values for each metric across animals, runs through function
#to calculate mean and SD. Appends those values and creates a DataFrame for 
#exporting. 
if no_stimulus_motion == 0:
    pLeftVel = mean_SD_pooling_appending(pLeftVel, Animals) 
    pRightVel = mean_SD_pooling_appending(pRightVel, Animals)
    pLeftLat = mean_SD_pooling_appending(pLeftLat, Animals) 
    pRightLat = mean_SD_pooling_appending(pRightLat, Animals) 
    pEyePos_Le = mean_SD_pooling_appending(pEyePos_Le, Animals) 
    pEyePos_Ri = mean_SD_pooling_appending(pEyePos_Ri, Animals) 
    pTime_Le = mean_SD_pooling_appending(pTime_Le, Animals) 
    pTime_Ri = mean_SD_pooling_appending(pTime_Ri, Animals)  
    pStim_Le = mean_SD_pooling_appending(pStim_Le, Animals) 
    pStim_Ri = mean_SD_pooling_appending(pStim_Ri, Animals) 

    #Puts the response bout counts per animal into a pandas Series for exporting
    bout_label = ['Leftward Bouts', 'Rightward Bouts']
    bout_counts = pd.DataFrame((pLeftBouts, pRightBouts), index=bout_label).T
    bout_counts_max = bout_counts.max(axis=0).rename('Max')
    bout_counts_min = bout_counts.min(axis=0).rename('Min')
    BoutCounts = pd.concat([bout_counts_max, bout_counts_min], axis=1)

    #Exports the combined individual and mean values as an Excel sheet
    with pd.ExcelWriter(Experiment) as writer:  
        pLeftVel.to_excel(writer, sheet_name='Lefward Velocities')
        pRightVel.to_excel(writer, sheet_name='Rightward Velocities')
        pLeftLat.to_excel(writer, sheet_name='Leftward Latencies')
        pRightLat.to_excel(writer, sheet_name='Rightward Latencies')
        pRightLat.to_excel(writer, sheet_name='Rightward Latencies')
        pEyePos_Le.to_excel(writer, sheet_name='Eye Position Leftward') 
        pEyePos_Ri.to_excel(writer, sheet_name='Eye Position Rightward') 
        pTime_Le.to_excel(writer, sheet_name='Time Leftward') 
        pTime_Ri.to_excel(writer, sheet_name='Time Rightward') 
        pStim_Le.to_excel(writer, sheet_name='Stimulus Leftward') 
        pStim_Ri.to_excel(writer, sheet_name='Stimulus Rightward') 
        BoutCounts.to_excel(writer, sheet_name='BoutCounts') 
        
    #Plots the Pooled Traces and Data
    #Plot leftward stimulus evoked responses
    direction = 'Left'
    plot_step_pool(pTime_Le, pEyePos_Le, pStim_Le, Animals, direction)
    
    #Plot rightward stimulus evoked responses
    direction = 'Right'
    plot_step_pool(pTime_Ri, pEyePos_Ri, pStim_Ri, Animals, direction)

else:
    #Calculates the average and SD for resting velocities
    pRest_Vel = mean_SD_pooling_appending(pRest_Vel, Animals)
    #Exports into pooled excel sheet
    with pd.ExcelWriter(Experiment) as writer:  
        pRest_Vel.to_excel(writer, sheet_name='Resting Velocities')