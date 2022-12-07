# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 16:55:15 2022

@author: Nox.workstation
"""
#%% Section 1

# import modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
sns.set_context("talk", rc={"font.size":15,"axes.titlesize":15,"axes.labelsize":15})   
import sklearn #Seaborn boxplots


# Import data

#Data in excel format using template

#GUI for Selecting the File

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

data_r = pd.read_excel(root.filename, sheet_name = Sheetname, skiprows = 0)

print(data_r)


#%% Section 2

#Select data type
def select_data_type():
    
    import tkinter as tk

    class MyAnalysis:
        
        def __init__(self,myParent):
            self.parent = myParent
            self.myContainer1 = tk.Frame(myParent)
            self.myContainer1.pack()
            
            self.label6 = tk.Label(self.myContainer1,text = 'Select data Type' , foreground='red',bd=3,height=2,width=30,font=("Verdana",9))
            self.label6.pack(side=tk.TOP)
            
            self.label1 = tk.Label(self.myContainer1,text = " Rawdata=0  Residuals=1 ")
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
Data_type = select_data_type()
print (Data_type)

if Data_type =='0':
    print('raw')
    motor_parameters =(data_r.columns.values)[2:]
else:
    print('residuALS')
    motor_parameters =(data_r.columns.values)[1:]
    


#%% Section 2

def boxplotflies(dataframe,motor_parameter):
    
    '''
    This function will perform boxplots of motor parameter specified. 
    
    param dataframe: data frame that contains the data
    motor_parameter: the motor parameter we want to analyse, it has to be a string for example 'speed'.
    The 
    ylaber: the label we want to see in the title and y axis. Has to be a string.
    return: Boxplots showing residual data for each condition and control 
    '''
    
    fig, ax = plt.subplots()  
    g=sns.boxplot(data=data_r, x='Condition', y= motor_parameter, palette= 'Spectral') #palete '' if flatui no ''
    ax.set_title(motor_parameter)
    ax.set_ylabel(motor_parameter)
    plt.xticks(rotation=80)
    plt.tick_params(labelsize=17)       
    plt.show()
    #fig.savefig('plot.png')
    
#boxplotflies(data_r,'speed cm_s') 

def boxplotsall(dataframe):
    '''
    This function will perform boxplots all the motor parameter specified in the exel file. Use residuals data. 
    
    param dataframe: data frame that contains the data
    types_fly: the conditions we want to compare, control and expetrimental. 
    return: Boxplots showing residual data for each condition and control 
    '''
    
    #1motor_parameters =(dataframe.columns.values)[2:]  #selet the motor parameters you want to include, started in 2 to not include mouse ID

    for index, _param in enumerate(motor_parameters): #loop to draw a plot for each parameter included in motor parameters
        fig, ax = plt.subplots()
        sns.boxplot(data=dataframe, x='Condition', y= _param, palette='Spectral')
        sns.despine(offset=10, trim=True)
        plt.xticks(rotation=80)
        #fig.savefig('./Figures/boxplot_{}.png'.format(_param)) #save each figure in a Folder called figures, you can chose any format you like png svg etc
        plt.show()

boxplotsall(data_r)