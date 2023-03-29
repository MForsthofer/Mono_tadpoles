Contains data and analysis scripts for the Forsthofer et al. 2023 paper. 
In the folder 'sinusoidal', the script 'sinus_single_file_segmentation.py' was used to 
segment original data files in the spike2 format, yielding filtered traces and measured
gain and phase values in one excel sheet per animal. These files can be found in
'data/[paradigm (control, lesioned, or monocular]/sinus'
This data was then pooled across all animals within one paradigm into excel sheets found in
'data' in the naming scheme [experimental group]_[paradigm].xlsx, using the 'sinus_pooling_script'

Step (unidirectional) data follows the same logic with scripts from the 'step' folder. 

Additionally, the 'Histogram symmetry indicices.py' script generates a histogram of 
symmetry values of all excel sheets for step data within the current working directory. 
It can be run from e.g. 'data/controls/step' to generate a histogram of symmetry 
values as shown in Figure 3d in the manuscript. 

The script 'plot_raw_data.py' can be used to plot original data from a .mat file of eye recordings
as shown in Fig. 1c). The 'Conversion_spike2-mat.s2s' script is executable in Spike2, 
and is used to convert Spike2 data in which the data was recorded into the .mat file format. 