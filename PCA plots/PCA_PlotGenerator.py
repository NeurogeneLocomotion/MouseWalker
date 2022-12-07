# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 12:52:14 2022

@author: Alexandra Medeiros
"""

#%% Section 1

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_context("talk",
                rc={
                    "font.size": 15,
                    "axes.titlesize": 15,
                    "axes.labelsize": 15
                })
from matplotlib import style
style.use('ggplot')

# data preprocessing
from sklearn.preprocessing import StandardScaler

# pca using sklearn decomposition
from sklearn.decomposition import PCA


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


df = pd.read_excel(root.filename, sheet_name = Sheetname)



labels = df['Condition']
num_samples = labels.size
print ('Number of samples: ', num_samples)
labels_unique = labels.unique()
print ('label unique values: ', labels_unique)
print ('num. of classes: ', labels_unique.size)


#%% Section 2
# check if there are any NaNs in the data frame

for attribute in df.columns[2:]:
    print ("column", attribute, "na \t: ", df[attribute].isnull().sum())

#%% Section 3

df_index = df.set_index('Condition')

# Drop first column of dataframe 'Animal ID'
df_pca = df_index.iloc[: , 1:]


df_pca.head()

#define Motor Variables
motor_variables = df_pca.columns.values[0:]
print(motor_variables)


# define function for PCA
import scipy
def PCA(data_scaled, num_components):
    """
    :param data_rescaled: N x P data matrix (should be normalized across features) 
                            where N is number of features and P number of data points
    :param num_components: number of components to keep
    
    :return: principal components of shape P x num_components, top eigenvectors and singular values
    """

    #### Applying PCA to the data #####
    data_centered = data_scaled - np.mean(data_scaled, axis=1)[:, None]

    # SVD decomposition of the rescaled data matrix
    U, S, V = scipy.linalg.svd(data_centered)

    # U vectors now contain the Eigenvectors of the covariance matrix
    # reduce the U matrix to the number of components stated
    U_reduced = U[:, :num_components]

    # obtain the loadings of the PC
    principal_components = np.dot(U_reduced.T, data_centered).T
    
    return principal_components, U_reduced, S


#%% Section 4
# normalize data across features
scaler = StandardScaler()
X_rescaled = scaler.fit_transform(df_pca)

# run PCA
n_components = 10
PCs, U, S = PCA(X_rescaled.T, n_components)
print(PCs.shape)
# np.set_printoptions(suppress=True)
exp_variance_ratio = (S*S)/(np.sum(S*S))
explained_variance = (exp_variance_ratio*100).round(1)
print ('explained variance by each component: ', explained_variance)

# store pcs in dataframe with labels
df_pcs_init = pd.DataFrame(data=PCs, columns=['PC{}'.format(i+1) for i in range(n_components)])
df_pcs = pd.concat([df.Condition, df_pcs_init], axis=1)
# compute centroids for each class; group by class label and compute mean for each PC
df_pcs_centroid = df_pcs.groupby('Condition').mean().reset_index()

#print(df_pcs.head())

#%% Section 5
#Weights of Parameters 

#Plots with Weights
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 5))
ax1.bar(np.arange(U[:, 0].size)+1, U[:, 0], color='#D82148')
ax1.set_xticks(np.arange(U[:, 0].size)+1)
ax1.set_xticklabels(motor_variables, rotation=90,fontsize=8)
ax1.axhline(y=0.04, color='grey', linestyle='--',linewidth=1)
ax1.axhline(y=-0.04, color='grey', linestyle='--',linewidth=1)
ax1.axhline(y=0, color='black', linestyle='-',linewidth=1)


ax2.bar(np.arange(U[:, 0].size)+1, U[:, 1], color='#D82148')
ax2.set_xticks(np.arange(U[:, 0].size)+1)
ax2.set_xticklabels(motor_variables, rotation=90,fontsize=8)
ax2.axhline(y=0.04, color='grey', linestyle='--',linewidth=1)
ax2.axhline(y=-0.04, color='grey', linestyle='--',linewidth=1)
ax2.axhline(y=0, color='black', linestyle='-',linewidth=1)
print(np.arange(U[:, 0].size)+1, U[:, 1])

ax3.bar(np.arange(U[:, 0].size)+1, U[:, 2],color='#D82148')
ax3.set_xticks(np.arange(U[:, 0].size)+1)
ax3.set_xticklabels(motor_variables, rotation=90,fontsize=8)
ax3.axhline(y=0.04, color='grey', linestyle='--',linewidth=1)
ax3.axhline(y=-0.04, color='grey', linestyle='--',linewidth=1)
ax3.axhline(y=0, color='black', linestyle='-',linewidth=1)

plt.tight_layout()
plt.savefig('weightsPCA.svg')
plt.show()

#%% Section 6

PCA_coord= PCs[:, :3]
#print(PCA_coord.shape)
#print(PCA_coord.T.shape)
df_PCs = pd.DataFrame(PCA_coord, columns=['PC1', 'PC2', 'PC3'])
df_PCs_labels = pd.concat([df['Condition'], df_PCs], axis=1)
#print(df_PCs_labels)

#Save the coordinates of the PC

df_PCs_labels.to_csv('PCallCoord.csv')

#%% Section 7

#Plot PCs

def get_cmap(n,name="RdYlGn"):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

from scipy.stats.distributions import chi2

def draw_ellipse_group(df, name_condition, coverage=0.5):
    '''
    :param df:  Principal Components dataframe
    :param name_condition: name of the parameter
    :param coverage: perecentage of coverage of the elipse, default = 50%
    
    :return: retuns an array that designs the elipse
    '''
    scale = np.sqrt(chi2.ppf(coverage, df=2))

    data = df_pcs.loc[df.Condition == name_condition, ['PC1', 'PC2']].values
    data_mean = np.mean(data, axis=0)
    data_centered = data - data_mean[None, :]
    data_cov = np.cov(data_centered.T)
    eig_val, eig_vec = np.linalg.eig(data_cov)
    sorted_indices = np.argsort(eig_val)[::-1]
    eig_val_sorted = eig_val[sorted_indices]
    eig_vec_sorted = eig_vec[:, sorted_indices]

    t = np.linspace(0, 2*np.pi, 100)
    ellipse = np.array([np.cos(t), np.sin(t)])
    vv = eig_vec_sorted * np.sqrt(eig_val_sorted)*scale
    ellipse_adj = vv @ ellipse + data_mean[:, None]
    return ellipse_adj


def plot_PC_groups_2D(group_names_list, exp_variance):
    '''
    :param group_names_list:  list of the names of the groups
    :param exp_variance: array with percentage of explained variance by each component
    
    :return: 2D PC plot
    '''

    # create color map based on number of groups
    cmap = get_cmap(len(group_names_list))
    
    # plot in 2D the first 2PCs
    fig, ax = plt.subplots()

    for i, name_str in enumerate(group_names_list):
        
        df_plot = df_pcs.loc[df_pcs.Condition == name_str]
        df_plot_centroid = df_pcs_centroid.loc[df_pcs_centroid.Condition == name_str]

        ax.scatter(
            df_plot['PC1'],
            df_plot['PC2'],
            marker='o',
            s=10,
            alpha=0.8,
            color=cmap(i),
            #  label= name_str
        )

        # plot centroid of data
        ax.scatter(df_plot_centroid['PC1'],
                   df_plot_centroid['PC2'],
                   marker='o',
                   s=150,
                   alpha=1,
                   color=cmap(i),
                   label=name_str)
        
        # draw ellipse around cloud of points
        ellipse_group = draw_ellipse_group(df_pcs,
                                           name_condition=name_str,
                                           coverage=0.5)
        ax.plot(ellipse_group[0, :], ellipse_group[1, :],
                color=cmap(i)
               )

    ax.set_xlabel('PC1: {} %'.format(exp_variance[0]), fontsize=15)
    ax.set_ylabel('PC2: {} %'.format(exp_variance[1]), fontsize=15)

    ax.legend(bbox_to_anchor=[1.5, 1])
    plt.show()
    
    
def plot_PC_groups_3D(group_names_list, exp_variance):
    '''
    :param group_names_list:  list of the names of the groups
    :param exp_variance: array with percentage of explained variance by each component
    
    :return: 3D PC plot
    '''
    
    cmap = get_cmap(len(group_names_list))

    # plot in 3D the first 3PCs
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(True)
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')

    plt.gca().patch.set_facecolor('white')
    plt.axis('On')

    for i, name_str in enumerate(group_names_list):

        df_plot = df_pcs.loc[df_pcs.Condition == name_str]
        df_plot_centroid = df_pcs_centroid.loc[df_pcs_centroid.Condition ==
                                               name_str]

        ax.scatter3D(
            df_plot['PC1'],
            df_plot['PC2'],
            df_plot['PC3'],
            marker='o',
            s=10,
            alpha=0.8,
            color=cmap(i),
            #                  label=name_str
        )

        # plot centroid of data
        ax.scatter3D(df_plot_centroid['PC1'],
                     df_plot_centroid['PC2'],
                     df_plot_centroid['PC3'],
                     marker='o',
                     s=150,
                     alpha=0.8,
                     color=cmap(i),
                     label=name_str)
        
    ax.set_xlabel('PC1: {} %'.format(exp_variance[0]), fontsize=15)
    ax.set_ylabel('PC2: {} %'.format(exp_variance[1]), fontsize=15)
    ax.set_zlabel('PC3: {} %'.format(exp_variance[2]), fontsize=15)
    plt.legend(bbox_to_anchor=[1.5, 1])
    plt.show()
    
    
group_names_list = labels_unique    
plot_PC_groups_2D(group_names_list, explained_variance)
plot_PC_groups_3D(group_names_list, explained_variance)