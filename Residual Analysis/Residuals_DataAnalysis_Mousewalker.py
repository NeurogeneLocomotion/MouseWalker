# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:44:59 2022

@author: Alexandra Medeiros
"""

# # Script for the analysis of the Residuals

'''This script will automatically output the Residuals data into and excel sheet. 
The input is the excel sheet resulting from the Mousewalker analysis. The input 
excel sheet should have in each row one individual Mouse and for each column the motor parameter. 
The name of the group should be under a column called 'condition' writen in these exact same way. 
'''

#%% Section 1

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


#%% Section 2
# Import data from excel sheet 
#Import excel file from directory GUI

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
            
            self.label1 = tk.Label(self.myContainer1,text = "Sheetname \n Example: Sheet 1")
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

#%% Section 3
#Export data from exel

dataframe = pd.read_excel(root.filename, sheet_name = Sheetname, skiprows = 0)
#print(dataframe)


labels = dataframe['Condition']
num_samples = labels.size
labels_list = labels.unique()
#print(labels_list)

#groups = mousegroups_GUI()
Control_name= labels_list[0]
mouse_group = labels_list[1:]
#mouse_groups = groups.split(',')
print ('Control group: ',Control_name)
print ('Mouse group: ',mouse_group)




#List of number of individuals that belong to each group and number of groups
dataframe['Condition'].value_counts()
num_ind_group = dataframe.groupby('Condition',sort=False).size().values
print ('Number of individuals in each group: ',num_ind_group)
num_groups = len(dataframe.groupby('Condition'))
print ('Number of Groups: ',num_groups)
df_selection = dataframe.set_index('Condition')


#define motor_parameters
motor_param =(dataframe.columns.values)[3:]
print(motor_param)



#%% Section 4

# Linear and non linear regresison functions

'''The next three modules are the functions with the code to perform Linear 
Regression, Logarithmic Regression, and power Regression. They do the Regresion
with the control data and then output the residuals for the control and Residuals.
There is no output. 
'''

#Normal Linear regression Functions 
def Lregression(dataframe, Control_name, motor_param):
    #Linear Regression for the Control and Residuals for control
    df = dataframe.set_index('Condition') #transform data into pandas organized by condition
    df_x= df['speed cm_s'].loc[Control_name] # select our independent variable speed for the control
    num_ind = (len(df_x)) #important parameter to define the results matrix later
    x = df_x.to_numpy()[:, None]       #organize data in a vector for linear regression
    df_y = df[motor_param].loc[Control_name]  #select dependent variable the motor parameter for the control
    y = df_y.to_numpy()[:, None]


    model = LinearRegression().fit(x,y)  #Do the linear regression model for the control 
    y_pred = model.predict(x)            #For the x valies draw the linear regression line
    residuals_control = y-y_pred         # Residuals for the control
    
    #plt.plot(x, y, 'o')
    #x_predict = np.linspace(26, 46, x.size)
    #plt.plot(x_predict, (pars[0]*np.power(x_predict, pars[1])).ravel(), '-')
    #plt.show()
    
    
    return model, residuals_control, num_ind


def make_residuals(model, motor_param, mouse_group):
    df = dataframe.set_index('Condition') 
    df_x= df['speed cm_s'].loc[mouse_group]
    num_ind = len(df_x)
    x_group1 = df_x.to_numpy()[:, None]
    df_y = df[motor_param].loc[mouse_group]
    y_group1 = df_y.to_numpy()[:, None]

    y_pred1 = model.predict(x_group1)
    #plt.plot(x_group1, y_group1,'o')
    #plt.plot(x_group1, y_pred1)
    #plt.show()
    residuals_groups = y_group1-y_pred1

    return num_ind, residuals_groups



# Logarithmic Regression Functions 

def Logregression(dataframe, Control_name, motor_param):
    '''This funtion does a Logarithmic Regression which is a Non-linear Regression the function is y= a*log(x)+b
     Inputs: dataframe-  is the data that comes from matlab Mousewalker
             Control_name- the name of the control as in the excel file
             motor_param- the motor parameter we want to analyse
     Outputs: pars gives us the a and of the logarithmic function
              residuals_control- the residuals of the control group
    '''
    #Linear Regression for the Control and Residuals for control
    df = dataframe.set_index('Condition') #transform data into pandas organized by condition
    df_x= df['speed cm_s'].loc[Control_name] # select our independent variable speed for the control
    num_ind = (len(df_x)) #important parameter to define the results matrix later
    x = df_x.to_numpy()      
    df_y = df[motor_param].loc[Control_name]  #select dependent variable the motor parameter for the control
    y = df_y.to_numpy()

    
    # Import curve fitting package from scipy
    from scipy.optimize import curve_fit
    # Function to calculate the exponential with constants a and b
    def log(x, a, b):
        return a*np.log(x)+b
    
    # Fit the exponential data
    pars, cov = curve_fit(f=log, xdata=x, ydata=y, p0=[-1, 0], bounds=(-np.inf, np.inf))
    # Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
    stdevs = np.sqrt(np.diag(cov))

    # Calculate the residuals
    residuals_control = y - log(x, *pars) # Residuals for the control
    
    #Calculate R squared
    ss_res = np.sum(residuals_control**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print(r_squared)
    
    plt.plot(x,pars[0]*np.log(x)+pars[1],'-')
    plt.plot(x,y, 'o')
    plt.show()
    
    
    return pars, residuals_control, num_ind

#pars, residuals_control,num_ind = Logregression(dataframe, Control_name, 'stance linearity')
#print(residuals_control)

def make_logresiduals(pars, motor_param, mouse_group):
    df = dataframe.set_index('Condition') 
    df_x= df['speed cm_s'].loc[mouse_group]
    num_ind = len(df_x)
    x_group1 = df_x.to_numpy()
    df_y = df[motor_param].loc[mouse_group]
    y_group1 = df_y.to_numpy()
    
    # Import curve fitting package from scipy
    from scipy.optimize import curve_fit
    # Function to calculate the exponential with constants a and b
    def log(x, a, b):
        return a*np.log(x)+b
    residuals_groups = y_group1 - log(x_group1, *pars)
    
    #If you want to see the plots uncomment the following code:
    #print(pars)
   # x_plot = np.arange(5,45,2)
    #print(x_plot)
    #plt.plot(x_plot,pars[0]*np.log(x_plot)+pars[1],'-')
   # plt.scatter(x_group1,y_group1)
    
    #print(x_plot)
    #print(pars[0]*np.log(x_plot)+pars[1])
    

    return num_ind, residuals_groups

#num_ind, residuals_groups = make_logresiduals(pars,'period ms','Sham -1dpi' )
#print(residuals_groups)



# Power Regression Functions 

def Power_regression(dataframe, Control_name, motor_param):
    '''This funtion does a Power Regression which is a Non-linear Regression the function is y= a*x^b
     Inputs: dataframe-  is the data that comes from matlab Mousewalker
             Control_name- the name of the control as in the excel file
             motor_param- the motor parameter we want to analyse
     Outputs: pars gives us the a and of the logarithmic function
              residuals_control- the residuals of the control group
    '''
    #Linear Regression for the Control and Residuals for control
    df = dataframe.set_index('Condition') #transform data into pandas organized by condition
    df_x= df['speed cm_s'].loc[Control_name] # select our independent variable speed for the control
    num_ind = (len(df_x)) #important parameter to define the results matrix later
    x = df_x.to_numpy()      
    df_y = df[motor_param].loc[Control_name]  #select dependent variable the motor parameter for the control
    y = df_y.to_numpy()
    #plt.plot(x, y,'o')

    
    # Import curve fitting package from scipy
    from scipy.optimize import curve_fit
    # Function to calculate the power-law with constants a and b
    def power_law(x, a, b):
        return a*np.power(x, b)

    # Fit the power-law data
    pars, cov = curve_fit(f=power_law, xdata=x, ydata=y, p0=[-1, 0], bounds=(-np.inf, np.inf))
    # Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
    stdevs = np.sqrt(np.diag(cov))
    # Calculate the residuals
    residuals_control = y - power_law(x, *pars)
    
       #Calculate R squared
    ss_res = np.sum(residuals_control**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print(r_squared)
    
    
    #If you want to see the plot with the model and group residuals uncomment the following:
    plt.plot(x,y, 'o')
    plt.plot(x, pars[0]*np.power(x, pars[1]), '--')
    plt.show()
    
    
    return pars, residuals_control, num_ind

#pars, residuals_control,num_ind = Power_regression(dataframe, Control_name, 'stance linearity') 
#print(residuals_control)

def make_powresiduals(pars, motor_param, mouse_group):
    df = dataframe.set_index('Condition') 
    df_x= df['speed cm_s'].loc[mouse_group]
    num_ind = len(df_x)
    x_group1 = df_x.to_numpy()
    df_y = df[motor_param].loc[mouse_group]
    y_group1 = df_y.to_numpy()
    
    # Import curve fitting package from scipy
    from scipy.optimize import curve_fit
    # Function to calculate the power-law with constants a and b
    def power_law(x, a, b):
        return a*np.power(x, b)

    residuals_groups = y_group1 - power_law(x_group1, *pars)

    return num_ind, residuals_groups

#num_ind, residuals_groups = make_powresiduals(pars,'stance linearity','SCI 15dpi' )
#print(residuals_groups)

#%% Section 6
    
# Linear and Non-Linear Regression for all the motor parameters for Control 

'''
The code bellow will do a for loop through all the motor parameters using Linear Regression for most of the parameters.
The exceptions are the following parameters: 

In these two parameters it will perform a power Regression y= a*x^b

    "period"  
    "stc t(AVR)" 
    
In these two it will perform a logarithmic Regression y= a*log(x)+b

    "stnc stabl"
    "Overall Stc Str"
    
 Troubleshooting: 

If there is an empty cell or NaN cell the code will give an error. 
Please make sure that in the excel there is no NaN Values or empty cells. 
You can either exclude this parameter or you can fill the blank cells 
(if not too many) with the average of the other values of the same group. 
To check which parameter is wrong uncomment the print(param) insithe the for loop.
The code will stop after the wrong parameter name appears. 
'''
#Loop over motor parameter to get the model for the control and residuals

#Determine the number of individuals in each group
num_ind_group = dataframe.groupby('Condition',sort=False).size().values  
num_ind_Control = num_ind_group[0]  #Select the first control group

#Build an empty matrix in which rows are the number of Individuals and columns the number of parameters 
Control_res = np.empty((num_ind_Control, len(motor_param))) 

#for loop to get the residuals for each motor parameter organized into columns 
for i, param in enumerate(motor_param):
    print(param)  #uncoment this if you want to see which parameters is failing
    
    if  param == 'stc t(AVR)' or param == 'stc t(F)' or param == 'stc t(H)' or param =='F PEP':
        pars, residuals_control,num_ind = Power_regression(dataframe, Control_name, param)
        #print('Power!')
        #print(residuals_control)
        Control_res[:, i] = residuals_control.ravel()  #Each column correspond to a specific motor parameter
    elif param == 'period ms' or param == 'body stability' or param == 'H PEP' :
        pars, residuals_control,num_ind = Logregression(dataframe, Control_name,param)
       # print('Log!)')
        #print(residuals_control)
        Control_res[:, i] = residuals_control.ravel()  #Each column correspond to a specific motor parameter 
    elif param == 'stance linearity' :
        df = dataframe.set_index('Condition')
        rawdata_control = df[param].loc[Control_name]
        residuals_control= rawdata_control.to_numpy()[:, None]
        print('raw!')
        #print(residuals_control)
        Control_res[:, i] = residuals_control.ravel()  #Each column correspond to a specific motor parameter 
    else: 
        model, residuals_control, num_ind = Lregression(dataframe, Control_name, param)
        #print('Linear!')
        Control_res[:, i] = residuals_control.ravel()  #Each column correspond to a specific motor parameter 

#print(Control_res)

#%% Section 7

#Linear and non linear Regression for the different groups and motor parameters

#loop over Mouse groups and motor parameters and fill the matrix 
all_groups = []
for i,group in enumerate(mouse_group):
    #print(group)
    #Get the num_ind for the matrix
    num_ind, residuals_groups = make_residuals(model, motor_param[0], group)
    #Make a empty matrix to fill with the values ofr each group and each parameter
    Group_res = np.zeros((num_ind, len(motor_param)))
    for i, param in enumerate(motor_param):
        if  param == 'stc t(AVR)' or param == 'stc t(F)' or param == 'stc t(H)' or param =='F PEP':
            pars, residuals_control,num_ind = Power_regression(dataframe, Control_name, param)
            num_ind, residuals_groups = make_powresiduals(pars, param, group)
            Group_res[:, i] = residuals_groups       
        elif param == 'period ms' or param == 'body stability' or param == 'H PEP' :
            pars, residuals_control,num_ind = Logregression(dataframe, Control_name,param)
            num_ind, residuals_groups = make_logresiduals(pars, param, group)
            Group_res[:, i] = residuals_groups.ravel() 
        elif  param == 'stance linearity' :
             df = dataframe.set_index('Condition')
             rawdata_group = df[param].loc[group]
             y = rawdata_group.to_numpy()[:, None]
             residuals_groups= rawdata_group.to_numpy()[:, None]
             print('raw!')
             Group_res[:, i] = residuals_groups.ravel() 
       
        else: 
            model, residuals_control, num_ind = Lregression(dataframe, Control_name,param)
            num_ind, residuals_groups = make_residuals(model, param, group)
            Group_res[:, i] = residuals_groups.ravel()

    all_groups.append(Group_res)

    
result_arr = np.concatenate(all_groups)  
#print(np.shape(result_arr))
resultsarr = pd.DataFrame(result_arr)
results_C = np.vstack((Control_res,result_arr))#Merge control Matrix with groups matrix

#print(np.shape(result_arr))



#%% Section 8


#print(results_C)
Results = pd.DataFrame(results_C)

# Change the column names 
Results.columns =motor_param

# Change the row indexes 
num_ind_group = dataframe.groupby('Condition',sort=False).size().values 
#print(num_ind_group)
groups_size = num_ind_group[1:] #size of the groups excluding the control

#print (num_ind_group)
#print(num_ind_group[0])
#print(mouse_group)


#make a list with all the names of the non control groups repeated the number of times in each group
groups_list_raw= []
for i in range (len(groups_size)):
    x = [mouse_group[i]]*groups_size[i]
    groups_list_raw.append(x)

groups_list = [item for sublist in groups_list_raw for item in sublist] #compreension for the lists


#append a list with the control 
control_list = [Control_name]*num_ind_group[0]

#List Combining control and groups
list_all = control_list +groups_list
#print(list_all)


Results.index = list_all
Results.index.name = 'Condition'
#print(Results)

#Results.to_csv('Residuals_python',index=True)
Results.to_excel('Residuals_python.xlsx')
#%% Section 9
#%% Section 10
