# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 19:32:18 2023

@author: mfors
"""
import pandas as pd
import os 
import matplotlib.pyplot as plt
import numpy as np

velocities = []
arr = np.array([])

Path = os.listdir()
for i in Path:
    if i.endswith('.xlsx'):
        excel_file = pd.ExcelFile(i)
        sheets = excel_file.sheet_names
        bino =pd.read_excel(excel_file, sheets[0]).iloc[:,0].values
        lesion =pd.read_excel(excel_file, sheets[0]).iloc[:,1].values
        mono =pd.read_excel(excel_file, sheets[0]).iloc[:,2].values
        data = np.vstack([bino, lesion, mono])
        
        barwidth = 0.7
        # f, ax = plt.subplots()
        n_bins = np.arange(0,2.5,0.25)
        # ax.hist(data.transpose(), n_bins, histtype='bar', align='mid', rwidth=barwidth)
        # plt.xlabel('T/N symmetry')
        # plt.ylabel('# of animals')
        #velocities.append(pd.read_excel(excel_file, sheets[1]).iloc[:,1].values)
        # arr = np.append(arr, pd.read_excel(excel_file, sheets[1]).iloc[:,1].values)
        f2, ax2 = plt.subplots(3,1)
        mngr = plt.get_current_fig_manager()
        # Places plot into the upper left corner for example:
        mngr.window.setGeometry(1,31,540, 640)
        ax2[0].hist(abs(bino), n_bins, color=[(0, 1, 1)], rwidth=barwidth)
        ax2[0].set_ylim(0,5)
        ax2[0].set_xlim(0,2.5)
        ax2[1].hist(abs(lesion), n_bins, color=[0, 1, 0], rwidth=barwidth)
        ax2[1].set_ylim(0,5)
        ax2[1].set_xlim(0,2.5)
        ax2[1].set_ylabel('# of animals')
        ax2[2].hist(abs(mono), n_bins, color=[1, 0, 1], rwidth=barwidth)
        ax2[2].set_ylim(0,5)
        ax2[2].set_xlim(0,2.5)
        # plt.xlim(10,1000)
        # plt.ylim(0, 1000)
        plt.xlabel('T/N symmetry')
