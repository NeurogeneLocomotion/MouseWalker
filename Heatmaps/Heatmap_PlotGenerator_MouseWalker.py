# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:06:43 2019

@author: Alexandra Medeiros
"""

#%% Section 1

# import modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import os
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.libqsturng import psturng
#from statsmodels.compat.python import range
sns.set_context("talk",
                rc={
                    "font.size": 15,
                    "axes.titlesize": 15,
                    "axes.labelsize": 15
                })

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

#Export data from exel

dataframe = pd.read_excel(root.filename, sheet_name = Sheetname, skiprows = 0)
#print(dataframe)



df_selection = dataframe.set_index('Condition')


#%% Section 1.1

 

#Select Sheetname
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
    motor_parameters =(dataframe.columns.values)[2:len(df_selection.columns)+2];
else:
    motor_parameters =(dataframe.columns.values)[1:len(df_selection.columns)+1]
    


'''
#Select Groups/Flygenotypes GUI


def mousegroups_GUI():

    import tkinter as tk

    class MyAnalysis:
        def __init__(self, myParent):
            self.parent = myParent
            self.myContainer1 = tk.Frame(myParent)
            self.myContainer1.pack()

            self.label6 = tk.Label(self.myContainer1,
                                   text='Fly Groups',
                                   foreground='red',
                                   bd=3,
                                   height=2,
                                   width=30,
                                   font=("Verdana", 9))
            self.label6.pack(side=tk.TOP)

            self.label1 = tk.Label(self.myContainer1,
                                   text="Name of the groups separated by Commas No spaces \n Example Group1,Group2,...")
            self.entry1 = tk.Entry(self.myContainer1, bd=10)
            self.label1.pack(side=tk.TOP)
            self.entry1.pack(side=tk.TOP)

            self.button1 = tk.Button(self.myContainer1)
            self.button1.configure(text="Done", background="tan")
            self.button1.pack(side=tk.TOP)
            self.button1.bind("<Button-1>", self.button1Click)  ### (2)

        def button1Click(self, event):  ### (5)
            global one
            one = self.entry1.get()

            self.parent.destroy()

    root = tk.Tk()

    MyAnalysis(root)
    #print(myapp.age)
    root.mainloop()

    return one
'''
#%% Section 1.2
labels = dataframe['Condition']
num_samples = labels.size
labels_unique = labels.unique()
print(labels_unique)

#groups = mousegroups_GUI()
mouse_groups = labels_unique
#mouse_groups = groups.split(',')

print (mouse_groups)

#  flyTypes = dmso_58_7days,dmso_58_14days,dmso_58_21days,dmso_463_7days,dmso_463_14days,dmso_463_21days


#List of number of individuals that belong to each group and number of groups
dataframe['Condition'].value_counts()
num_flies_group = dataframe.groupby('Condition',sort=False).size().values
print ('Number of individuals in each group: ',num_flies_group)
num_groups = len(dataframe.groupby('Condition'))
print ('Number of Groups: ',num_groups)


#%% Section 2

#Function for Dunn Multiple tests 
#Create "compare" list with list of stuff we want to compare for Dunn test


compare = []
for i, group in enumerate(mouse_groups[1:]):
    one = ((0,i+1))
    compare.append(one)
#print(compare)



def kw_dunn(dataframe,motor_parameter, to_compare=compare, alpha=0.05, method='bonf'):
    motor_data = df_selection[motor_parameter]
    groups = np.array(([motor_data.loc[type] for type in mouse_groups]))
 #Change fly types names and the tuple to compare, normally you compare the control with all the other groups    
    """
    Kruskal-Wallis 1-way ANOVA with Dunn's multiple comparison test
    Arguments:
    ---------------
    groups: sequence
        arrays corresponding to k mutually independent samples from
        continuous populations
    to_compare: sequence
        tuples specifying the indices of pairs of groups to compare, e.g.
        [(0, 1), (0, 2)] would compare group 0 with 1 & 2. by default, all
        possible pairwise comparisons between groups are performed.
    alpha: float
        family-wise error rate used for correcting for multiple comparisons
        (see statsmodels.stats.multitest.multipletests for details)
    method: string
        method used to adjust p-values to account for multiple corrections (see
        statsmodels.stats.multitest.multipletests for options)
    Returns:
    ---------------
    H: float
        Kruskal-Wallis H-statistic
    p_omnibus: float
        p-value corresponding to the global null hypothesis that the medians of
        the groups are all equal
    Z_pairs: float array
        Z-scores computed for the absolute difference in mean ranks for each
        pairwise comparison
    p_corrected: float array
        corrected p-values for each pairwise comparison, corresponding to the
        null hypothesis that the pair of groups has equal medians. note that
        these are only meaningful if the global null hypothesis is rejected.
    reject: bool array
        True for pairs where the null hypothesis can be rejected for the given
        alpha
    Reference:
    ---------------
    Gibbons, J. D., & Chakraborti, S. (2011). Nonparametric Statistical
    Inference (5th ed., pp. 353-357). Boca Raton, FL: Chapman & Hall.
    """
    # omnibus test (K-W ANOVA)
    # -------------------------------------------------------------------------

    #groups = [np.array(gg) for gg in groups]
    #motor_data = [motor_data.loc[type] for type in mouse_groups]
    k = len(groups)

    n = np.array([len(gg) for gg in groups])
    if np.any(n < 5):
        warnings.warn("Sample sizes < 5 are not recommended (K-W test assumes "
                      "a chi square distribution)")
    
    allgroups = np.concatenate(groups)
    N = len(allgroups)
    ranked = stats.rankdata(allgroups)

    # correction factor for ties
    T = stats.tiecorrect(ranked)
    if T == 0:
        raise ValueError('All numbers are identical in kruskal')

    # sum of ranks for each group
    j = np.insert(np.cumsum(n), 0, 0)
    R = np.empty(k, dtype=np.float)
    for ii in range(k):
        R[ii] = ranked[j[ii]:j[ii + 1]].sum()

    # the Kruskal-Wallis H-statistic
    H = (12. / (N * (N + 1.))) * ((R ** 2.) / n).sum() - 3 * (N + 1)

    # apply correction factor for ties
    H /= T

    df_omnibus = k - 1
    p_omnibus = stats.distributions.chi2.sf(H, df_omnibus)

    # multiple comparisons
    # -------------------------------------------------------------------------

    # by default we compare every possible pair of groups
    if to_compare is None:
        to_compare = tuple(combinations(range(k), 2))
        
    ncomp = len(to_compare)
    
    Z_pairs = np.empty(ncomp, dtype=np.float)
    p_uncorrected = np.empty(ncomp, dtype=np.float)
    Rmean = R / n
    
    for pp, (ii, jj) in enumerate(to_compare):
    
    # standardized score
        Zij = (np.abs(Rmean[ii] - Rmean[jj]) / np.sqrt((1. / 12.) * N * (N + 1) * (1. / n[ii] + 1. / n[jj])))
        Z_pairs[pp] = Zij

    # corresponding p-values obtained from upper quantiles of the standard
    # normal distribution
    p_uncorrected = stats.norm.sf(Z_pairs) * 2.

    # correction for multiple comparisons
    reject, p_corrected, alphac_sidak, alphac_bonf = multipletests(p_uncorrected, method=method)
    #return  p_omnibus,p_corrected
    return  p_corrected


#Function for statistic analysis
    
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi
import pandas as pd


def statsflies(dataframe, motor_parameter):
    '''
    This function will perform all statistic analyses including normality and standard deviation tests. 
    If data is normally distributed and homoscedastic (i.e. equal variance for all predictors),
    it will perform a one-way anova; otherwise it will perform a non-parametric test (kruskal-wallis).
    Moreover it will perform multicomparison tests to compare between the data and the control.
    
    param dataframe: data frame that contains the data
    param motor_parameter: string argument that calls the motor variable under consideration
    return: List of pvalues for a specific motor parameter
    '''
    print(motor_parameter)
    pvalues_all = []
    motor_data = df_selection[motor_parameter]
    ('Data treated: {}'.format(motor_data))
    

    # test for normality - shapiro-wilk test
   # print(motor_data.loc['SCI -1dpi'])[1]
    shapiro_tests_p = np.array([stats.shapiro(motor_data.loc[type])[1] for type in mouse_groups])
   # print ('Shapiro p values = {}'.format(shapiro_tests_p))
    #print ((np.where(shapiro_tests_p>0.05)))

    # test for homoscedasticity - levene test
    levene_test_p = stats.levene(*[motor_data.loc[type]
                                   for type in mouse_groups])[1]
   #print ('Levene p value = {}'.format(levene_test_p))

    # condition to check that the data is both normal and homoscedastic
    #print (np.where(shapiro_tests_p>0.05))
    #print (levene_test_p>0.05)
    if (len(np.where(shapiro_tests_p > 0.05)[0]) == len(mouse_groups)) and (
            levene_test_p > 0.05):
        print('Data for each group satisfies normality condition and is homoscedastic')
        anova_test = stats.mstats.f_oneway(
            *[motor_data.loc[type].values for type in mouse_groups])
        anova_test_p = anova_test[1]
        #print('{}Anova pvalue ={}'.format(motor_parameter, anova_test_p))

        # post-hoc analysis to check which pairs of means are significantly different from each other
        tukey_test = pairwise_tukeyhsd(
            endog=motor_data,  #Tukey test predefined function
            groups=dataframe['Condition'],  #fly groups
            alpha=0.05)
        Results = tukey_test.summary()
        #print (Results)
        tukey_results = pd.DataFrame(
            data=Results.data[1:],
            columns=Results.data[0])  #transform summary table into dataframe so we can manipulate data
        #print (tukey_results)

        group1 = tukey_results['group1']         #select first two columns of the table that contain the groups
        group2 = tukey_results['group2']
        control = mouse_groups[0]

        mean_diffs = []
        std_pairs = []
        for x in mouse_groups[1:]:
            for i, group in enumerate(group1):                   #loop so we find on the table the comparisons Control with each exp group
                if group == control and group2[i] == x:
                    mean_diffs.append(tukey_test.meandiffs[i])
                    std_pairs.append(tukey_test.std_pairs[i])
                elif group == x and group2[i] == control:
                    mean_diffs.append(tukey_test.meandiffs[i])
                    std_pairs.append(tukey_test.std_pairs[i])

        st_range = np.abs(mean_diffs) / std_pairs
        pvalues = psturng(
            st_range, len(tukey_test.groupsunique[:(len(mouse_groups) - 1)]),                   #formula to calculate pvalues
            tukey_test.df_total
        )  #df.groups unique is the name of groups the order doesn't matter

    else:
        print ('Normality or homoscedasticity condition is not satified, Kruskal Walls test')
        #kruskal_test = stats.mstats.kruskalwallis(*[motor_data.loc[type].values for type in mouse_groups]) #H, pval
        pvalues = kw_dunn(dataframe,
                          motor_parameter,
                          to_compare=compare,
                          alpha=0.05,
                          method='bonf')

    pvalues_all.append(pvalues)
    return pvalues_all

#statsflies(dataframe,'diag swing')

#%% Section 3
def compute_means(dataframe, motor_parameter):
    """
    compare_means will take all the values of motor_parameter dataframe and compute the means for each group 
    of flies. This function is just to see if values increase or decrease in relation to the control. Afterwards we will cross them with 
    the p values and attribute each p values with a positive or negative value depending if the mean of thegroups increase or decrease 
    respectively
    Input: dataframe receives the exel file where the data is stored.
           motor_parameter receives the motor parameter 
    Output: mean_list:  a list of means for each group of flies
    """
    mean_list=[]
    motor_data = df_selection[motor_parameter]
    for type_index in mouse_groups:
        mean_group = np.mean(motor_data.loc[type_index].values)
        mean_list.append(mean_group)
    return (mean_list)

#compute_means(dataframe, 'diag swing')

#%% Section 4

def classify_values(dataframe,motor_parameter):
    """
    classify values function will separate in bins the p_values of each group
    Input: dataframe receives the exel file where the data is stored.
           motor_parameter receives the motor parameter 
    Output: pvalues_class: a list of values separating the pvalues in bins with higher value if more significant
    """
    pvalues_class= []
    pvalues_all = statsflies(dataframe, motor_parameter)
    pvalues_all = np.array(pvalues_all)
    pvalues_all = pvalues_all[0,:]
    for index in range(len(pvalues_all)):
        if pvalues_all[index] >= 0.05:
            pvalues_class.append(0)
        elif pvalues_all[index] < 0.05 and pvalues_all[index] >= 0.01:
            pvalues_class.append(2.5)
        elif pvalues_all[index] < 0.01 and pvalues_all[index] > 0.001:
             pvalues_class.append(5) 
        elif pvalues_all[index] <= 0.001:
            pvalues_class.append(7.5)
        else:
            pvalues_class.append(0)
    return pvalues_class

#%% Section 5

def pvalue_mod(dataframe,motor_parameter):
    """
    pvalue_mod will transform pvalues into negative values if the group values decrease in relation to the control
    Input: dataframe receives the exel file where the data is stored.
           motor_parameter receives the motor parameter
    Output: pvalues_mod: list of binned pvalues positive or negative depending if groups values increase or decrease in relation
    to the control, respoectively
    """
    pvalue_mod= []
    pvalues_class = classify_values(dataframe,motor_parameter)
    mean_list= compute_means(dataframe, motor_parameter)
    #print(mean_list)


    for n in range(len(mean_list)-1):
        if mean_list[0] > mean_list[n+1]: 
            pvalue_mod.append(-1)
        else:
            pvalue_mod.append(1)
        #print (pvalue_mod)
    pvalue_mod = np.array(pvalue_mod)
    pvalues_mod = pvalue_mod * pvalues_class
    return pvalues_mod

#%% Section 6
    
def matrix_dataframe(dataframe):
    """
    Creates a matrix with all the pvalues for all motor parameters
    """
    matrix_pvalues = []
    #motor_parameters =(dataframe.columns.values)[2:len(df_selection.columns)+1]   
    for index, _param in enumerate(motor_parameters):
        pvalue_motor = pvalue_mod(dataframe,_param)
        matrix_pvalues.append(pvalue_motor)
        print(_param)
    matrix_pvalues = np.array(matrix_pvalues)
    print ( (matrix_pvalues))
    return matrix_pvalues

def heatmap(dataframe):
    """
    Creates a heatmap that returns the differences from the control for each group and for each motor parameter
    """
    hdata = matrix_dataframe(dataframe)
    sns.set(font_scale=1.5)
   #motor_parameters =(dataframe.columns.values)[2:len(df_selection.columns)+1]  
    f, ax = plt.subplots(figsize=(10,20))
    sns_heatmap = sns.heatmap(hdata,vmin=-7.5, vmax=7.5,cmap='RdBu_r', annot=False, linewidths=.7, ax=ax, yticklabels = motor_parameters, xticklabels=mouse_groups[1:])
    sns_fig = sns_heatmap.get_figure()
    def createFolder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print ('Error: Creating directory. ' +  directory)
        
    createFolder('./Figures')    
    
    sns_fig.savefig("./Figures/heatmap.png")
    return sns_heatmap 
heatmap(dataframe)