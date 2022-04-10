#!/usr/bin/env python
from nilearn.plotting import view_img
from nilearn.reporting import get_clusters_table
import pandas as pd
import os
import numpy as np 
from matplotlib import pyplot as plt
import sys
import datetime

basedir = '/media/nfarrugi/datapal/results'
firstlevelpath = os.path.join(basedir,'univariate')

now = datetime.datetime.now()

Df = pd.read_pickle('BAIS_GMSI.pkl')

secondlevelpath = os.path.join('/media/nfarrugi/datapal/results','RESULTS_GLM_covar_{}'.format(now.strftime("%Y-%m-%d_%H-%M")))

if not(os.path.isdir(secondlevelpath)):
    os.makedirs(secondlevelpath)

allsubj = Df['Subj code'].to_list()
print(Df)

print(allsubj)

P_TvsD = []
P_CvsF = []
I_TvsD = []
I_CvsF = []
PsupI = []
IsupP = []
control = []
play = []
for subj in allsubj:
    P_TvsD.append(os.path.join(firstlevelpath,f"{subj}",f"P_TvsD_{subj}.nii.gz"))
    P_CvsF.append(os.path.join(firstlevelpath,f"{subj}",f"P_CvsF_{subj}.nii.gz"))
    I_TvsD.append(os.path.join(firstlevelpath,f"{subj}",f"I_TvsD_{subj}.nii.gz"))
    I_CvsF.append(os.path.join(firstlevelpath,f"{subj}",f"I_CvsF_{subj}.nii.gz"))
    PsupI.append(os.path.join(firstlevelpath,f"{subj}",f"PsupI_{subj}.nii.gz"))
    IsupP.append(os.path.join(firstlevelpath,f"{subj}",f"IsupP_{subj}.nii.gz"))
    control.append(os.path.join(firstlevelpath,f"{subj}",f"Control_{subj}.nii.gz"))
    play.append(os.path.join(firstlevelpath,f"{subj}",f"Play_{subj}.nii.gz"))

def second_level(cmap_filenames,regressor = 'BAIS-Control',Df=Df,twosided=True,clust=20,thr=0.001):
        
    ## Estimate second level model 
    
    ## The design matrix is given by the Dataframe with the behavioral variables.

    n_samples = len(cmap_filenames)
    assert n_samples == len(Df)

    ## add an intercept 
    design_matrix = Df.drop(columns=['Subj code'])
    design_matrix['intercept'] = 1

    from nilearn.glm.second_level import SecondLevelModel
       
    second_level_model = SecondLevelModel(smoothing_fwhm=6).fit(
        cmap_filenames, design_matrix=design_matrix)

    z_map = second_level_model.compute_contrast(regressor,output_type='z_score')
    from nilearn.glm.thresholding import threshold_stats_img

    thresholded_map1, threshold1 = threshold_stats_img(
        z_map, alpha=thr, height_control='fpr', cluster_threshold=clust,two_sided=twosided)

    print('The FPR=%.3g threshold is %.3g' % (thr,threshold1))

    return z_map,thresholded_map1,threshold1
 


def calc_view(listmaps,label,regressor,thr,Df=Df,twosided=True,clust=20):

    z_map,thresholded_map1,threshold1 = second_level(listmaps,regressor,Df=Df,twosided=twosided,clust=clust,thr=thr)
    #view_img(z_map,title=f"{label} - raw map").open_in_browser()
    view_img(thresholded_map1,threshold=threshold1,title=f"{label} {regressor}- FPR <{thr}").open_in_browser()
    Df = get_clusters_table(z_map,threshold1,clust,two_sided=twosided)
    print(f"Regressor : {regressor}, Table for {label}")
    print(Df)

    return z_map,thresholded_map1,threshold1

for regressor in ['BAIS-Vivid','BAIS-Control','MT_mean']:
    thr=0.001
    calc_view(P_TvsD,"C_CvsF",regressor=regressor,clust=10,thr=thr)
    calc_view(P_TvsD,"P_TvsD",regressor=regressor,clust=10,thr=thr)
    calc_view(P_CvsF,"P_CvsF",regressor=regressor,clust=10,thr=thr)
    calc_view(I_TvsD,"I_TvsD",regressor=regressor,clust=10,thr=thr)
    calc_view(I_CvsF,"I_CvsF",regressor=regressor,clust=10,thr=thr)
    calc_view(PsupI,"PsupI",regressor=regressor,twosided=True,clust=100,thr=thr)
    #calc_view(IsupP,"IsupP",regressor=regressor,twosided=False,clust=100)

