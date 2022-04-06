#!/usr/bin/env python
from nilearn.plotting import view_img,view_img_on_surf,plot_stat_map,plot_glass_brain
from nilearn.reporting import get_clusters_table
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img as map_threshold
import pandas as pd
import os
import numpy as np 
from matplotlib import pyplot as plt
from utils import fetch_imgs_motions,fetch_X_y,compute_matrix_contrast_betas,tr,fetch_ica_ts
import sys
from nilearn.image import mean_img,threshold_img
import datetime

basedir = '/media/nfarrugi/datapal/imagery-mvpa/results'
firstlevelpath = os.path.join(basedir,'univariate')

now = datetime.datetime.now()

secondlevelpath = os.path.join('/media/nfarrugi/datapal/results','RESULTS_GLM_{}'.format(now.strftime("%Y-%m-%d_%H-%M")))

if not(os.path.isdir(secondlevelpath)):
    os.makedirs(secondlevelpath)

allsubj = os.listdir(firstlevelpath)


P_TvsD = []
P_CvsF = []
I_TvsD = []
I_CvsF = []
PsupI = []
IsupP = []
control = []
play = []
for subj in allsubj:
    P_TvsD.append(os.path.join(firstlevelpath,subj,f"P_TvsD_{subj}.nii.gz"))
    P_CvsF.append(os.path.join(firstlevelpath,subj,f"P_CvsF_{subj}.nii.gz"))
    I_TvsD.append(os.path.join(firstlevelpath,subj,f"I_TvsD_{subj}.nii.gz"))
    I_CvsF.append(os.path.join(firstlevelpath,subj,f"I_CvsF_{subj}.nii.gz"))
    PsupI.append(os.path.join(firstlevelpath,subj,f"PsupI_{subj}.nii.gz"))
    IsupP.append(os.path.join(firstlevelpath,subj,f"IsupP_{subj}.nii.gz"))
    control.append(os.path.join(firstlevelpath,subj,f"C_CvsF_{subj}.nii.gz"))
    play.append(os.path.join(firstlevelpath,subj,f"Play_{subj}.nii.gz"))
        
def second_level(cmap_filenames,twosided=True,clust=20):
        
    ## Estimate second level model 
    # do the design matrix 
    n_samples = len(cmap_filenames)
    design_matrix = pd.DataFrame([1] * n_samples,columns=['intercept'])

       
    second_level_model = SecondLevelModel(smoothing_fwhm=6).fit(
        cmap_filenames, design_matrix=design_matrix)

    z_map = second_level_model.compute_contrast(output_type='z_score')
    

    thresholded_map1, threshold1 = map_threshold(
        z_map, alpha=.001, height_control='fpr', cluster_threshold=clust,two_sided=twosided)

    print('The FPR=.001 threshold is %.3g' % threshold1)

    thresholded_map2, threshold2 = map_threshold(
        z_map, alpha=.05, height_control='fdr',cluster_threshold=clust,two_sided=twosided)
    print('The FDR=.05 threshold is %.3g' % threshold2)

    thresholded_map3, threshold3 = map_threshold(
        z_map, alpha=.05, height_control='bonferroni',two_sided=twosided)
    print('The p<.05 Bonferroni-corrected threshold is %.3g' % threshold3)

    thresholded_map4, threshold4 = map_threshold(
        z_map, alpha=.01, height_control=None,threshold=2.5,two_sided=twosided)
    print('The p<.01 ucorrected threshold is %.3g' % threshold4)

    from nilearn.glm import cluster_level_inference
    proportion_true_discoveries_img = cluster_level_inference(z_map, threshold=threshold1, alpha=.05)
    
    return z_map,thresholded_map1,threshold1,thresholded_map2,threshold2,thresholded_map3,threshold3,thresholded_map4,threshold4,proportion_true_discoveries_img
 


def calc_view(listmaps,label,twosided=True,clust=20,savepath=secondlevelpath,color='r'):

    z_map,thresholded_map1,threshold1,thresholded_map2,threshold2,thresholded_map3,threshold3,thresholded_map4,threshold4,prop = second_level(listmaps,twosided=twosided,clust=clust)
    #view_img(thresholded_map1,threshold=threshold1,title=f"{label} - FPR <.001").open_in_browser()
    
    f,ax = plt.subplots()
    plot_stat_map(stat_map_img=thresholded_map2,threshold=threshold2,figure=f,axes=ax)
    
    f.savefig(os.path.join(savepath,f"{label}.png"))

    #plot_glass_brain(stat_map_img=thresholded_map1,threshold=threshold1,output_file=os.path.join(savepath,f"{label}_glass.png"))
    #plot_glass_brain(stat_map_img=thresholded_map1,threshold=threshold1,output_file=os.path.join(savepath,f"{label}_glass.pdf"))

    display = plot_glass_brain(None)
    display.add_contours(thresholded_map2, levels=[threshold2], filled=True,colors=color)
    display.savefig(os.path.join(savepath,f"{label}_glass.svg"))
    
    Df = get_clusters_table(thresholded_map2,threshold2,clust,two_sided=twosided)
    print(Df)
    Df.to_csv(os.path.join(savepath,f"{label}.csv"),index=False)
    z_map.to_filename(os.path.join(savepath,f"{label}.nii.gz"))

    return z_map,thresholded_map3,threshold3

if True:
    #calc_view(P_TvsD,"C_CvsF",clust=10)
    #calc_view(P_TvsD,"P_TvsD",clust=10)
    zP_IcvsF,tmapPcvsF,tPcvsF =calc_view(P_CvsF,"P_CvsF",clust=10,color='g')### FPR 3.28, FDR 4.93 , FWER is 4.94
    zI_ITvsD,tmapITvsD,tITvsD  = calc_view(I_TvsD,"I_TvsD",clust=10,color='r') ### FDR is inf ??? , FWER is 4.94
    zI_IcvsF,tmapIcvsF,tIcvsF = calc_view(I_CvsF,"I_CvsF",clust=10,color='b') ### FDR is 4.22, FWER is 4.94
    zPsupI,tmapPsupI,tPsupI = calc_view(PsupI,"PsupI",twosided=True,clust=10)### FDR is 3.38 , FWER is 4.94
    """ 
    f,ax = plt.subplots(nrows=2,ncols=2,figsize=(20,20))
    plot_stat_map(stat_map_img=tmapITvsD,threshold=tITvsD,figure=f,axes=ax[0,0],title= "Imagery - T > D")
    plot_stat_map(stat_map_img=tmapIcvsF,threshold=tIcvsF,figure=f,axes=ax[0,1],title= "Imagery - C > F#")
    plot_stat_map(stat_map_img=tmapPcvsF,threshold=tPcvsF,figure=f,axes=ax[1,0],title= "Perception - C > F#")
    plot_stat_map(stat_map_img=tmapPsupI,threshold=tPsupI,figure=f,axes=ax[1,1],title= "Perception > Imagery")
    plt.tight_layout()
    f.savefig(os.path.join(secondlevelpath,f"clusters.png"))
    """
    f,ax = plt.subplots(nrows=2,ncols=2,figsize=(10,10))
    plot_glass_brain(stat_map_img=tmapITvsD,threshold=tITvsD,figure=f,axes=ax[0,0],title= "Imagery - T > D")
    plot_glass_brain(stat_map_img=tmapIcvsF,threshold=tIcvsF,figure=f,axes=ax[0,1],title= "Imagery - C > F#")
    plot_glass_brain(stat_map_img=tmapPcvsF,threshold=tPcvsF,figure=f,axes=ax[1,0],title= "Perception - C > F#")
    plot_glass_brain(stat_map_img=tmapPsupI,threshold=tPsupI,figure=f,axes=ax[1,1],title= "Perception > Imagery")
    f.savefig(os.path.join(secondlevelpath,f"clusters_glass.svg"))

    f,ax = plt.subplots(nrows=2,ncols=1,figsize=(10,10))
    #plot_glass_brain(stat_map_img=tmapITvsD,threshold=tITvsD,figure=f,axes=ax[0],title= "Imagery - T > D")

    display = plot_glass_brain(None,figure=f,axes=ax[1],title= "B")
    display.add_contours(tmapITvsD, levels=[tITvsD], filled=True,colors='r')
    display.add_contours(tmapIcvsF, levels=[tIcvsF], filled=True,colors='b')
    display.add_contours(tmapPcvsF, levels=[tPcvsF], filled=True,colors='g')
    plot_glass_brain(stat_map_img=tmapPsupI,threshold=tPsupI,figure=f,axes=ax[0],title= "A",colorbar=True,plot_abs=False)
    f.savefig(os.path.join(secondlevelpath,f"clusters_glass_2.svg"))




zP_CcvsF,tmapCcvsF,tCcvsF =calc_view(control,"control",clust=10,color='g',twosided=True) ## FDR is inf 

#calc_view(IsupP,"IsupP",twosided=False,clust=100)
#calc_view(control,"Control",clust=100)
#calc_view(play,"Play",clust=100)
