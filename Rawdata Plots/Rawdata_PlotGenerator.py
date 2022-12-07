# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 15:52:48 2022

@author: Alexandra Medeiros
"""

#%% Section 1

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

#GUI Selection of File

from tkinter import filedialog
from tkinter import *

root = Tk()
root.filename = filedialog.askopenfilename(initialdir="/",
                                           title="Select file",
                                           filetypes=(("excel files","*.xlsx"), ("all files","*.*")))
root.destroy()
print(root.filename)

#Select Sheetname
def experiment_parameters_GUI(filename):
    
    import tkinter as tk

    class MyAnalysis:
        
        def __init__(self,myParent):
            self.parent = myParent
            self.myContainer1 = tk.Frame(myParent)
            self.myContainer1.pack()
            
            self.label6 = tk.Label(self.myContainer1,text = filename ,foreground='red',bd=3,height=2,width=30,font=("Verdana",9))
            self.label6.pack(side=tk.TOP)
            
            self.label1 = tk.Label(self.myContainer1,text = "Sheetname \n Example: Sheet1")
            self.entry1 = tk.Entry(self.myContainer1,bd= 10)
            self.label1.pack(side=tk.TOP)
            self.entry1.pack(side=tk.TOP)
            
            self.button1 = tk.Button(self.myContainer1)
            self.button1.configure(text="Done",background = "tan")
            self.button1.pack(side=tk.TOP)	
            self.button1.bind("<Button-1>", self.button1Click) ### (2)
        
        def button1Click(self, event):  ### (5)
            global one
            one = self.entry1.get()


            self.parent.destroy()
        
    root = tk.Tk()
    
    MyAnalysis(root)
    #print(myapp.age)
    root.mainloop()
    
    return one

#GUI for name of the excel Sheet
Sheetname = experiment_parameters_GUI(root.filename)
print (Sheetname)

#Export data from exel

data = pd.read_excel(root.filename, sheet_name = Sheetname)

labels = data['Condition']
num_samples = labels.size
sample_groups = labels.unique()
print('Name of Groups:', sample_groups)


#%% Section 2

#function to plot Speed vs Selected Motor Parameter

def rawdataplots(sample_groups, motor_parameter):
    
    fig, ax = plt.subplots()
    for index, ids in enumerate(sample_groups):
        spec_data = data.loc[data['Condition']==ids]
        ax.scatter(spec_data['speed cm_s'], spec_data[motor_parameter], marker='o',edgecolor='none', 
               cmap='viridis', label=ids, s=60)
        
    ax.set_xlabel('speed cm_s', fontsize=16, fontweight='bold')
    ax.set_ylabel(motor_parameter, fontsize=16, fontweight='bold')
    ax.set_title('Raw data')
    ax.legend(bbox_to_anchor=(1.4, 0.4))
    plt.show()

#%% Section 3

#Function to plot raw data plots for every motor parameter
def rawdataplotsall():
    '''
    This function will perform boxplots all the motor parameter specified in the exel file. Use residuals data. 
    
    param dataframe: data frame that contains the data
    sample_groups: the conditions we want to compare, control and expetrimental. 
    return: Boxplots showing residual data for each condition and control 
    '''
    motor_parameters =(data.columns.values)[3:]
   # print (motor_parameters)
    for index, _param in enumerate(motor_parameters):
        rawdataplots(sample_groups, _param)
        plt.show()

rawdataplotsall()

