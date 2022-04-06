#!/usr/bin/env python

import pandas as pd
import os
import numpy as np 
from matplotlib import pyplot as plt
from utils import compute_matrix_contrast_betas,tr,fetch_ica_ts,fetch_imgs_confounds
import sys

cursubj = np.str(sys.argv[1])

basedir = '/home/nfarrugi/Documents/data/tudor/'

datacleanpath = os.path.join('/media/nfarrugi/datapal/beluga/fmriprep/',f"sub-{cursubj}")

betapath = os.path.join('/media/nfarrugi/datapal/','results','univariate',cursubj)

behavpath = os.path.join(basedir,'behav/timestamps/XLSX/')

if not(os.path.isdir(betapath)):
    os.makedirs(betapath)


def paradigm_modelI(behavpath,cursubj,betapath):
    #paradigm_allruns = paradigm_modelI(behavpath,cursubj,betapath)
    global tr 

    xlsfile = os.path.join(behavpath,'timestamps_%s.xls' % cursubj)

    goodheader = ['run',
    'rep',
    'blockNr',
    'blockType',
    'trialNrOfCrtBlock',
    'stimWAV',
    'categHarm',
    'subcategAcoust',
    'onset_cross',
    'onset_play',
    'onset_imagine',
    'onset_sing',
    'onset_ratingAppears',
    'onset_ratingMade',
    'onsetRunWise_cross',
    'onsetRunWise_play',
    'onsetRunWise_imagine',
    'onsetRunWise_sing',
    'onsetRunWise_ratingAppears',
    'onsetRunWise_ratingMade']

    print("file : ", xlsfile)
    MetaData = pd.read_excel(xlsfile)

    curhead = list(MetaData)

    if curhead[0]==1:
        print("fixing missing header!")
        MetaData = pd.read_excel(xlsfile,header=None,names=goodheader)
            

    # build events Dataframes, per run

    # This is where we modify to have one column for each trial. We simply add a number (count_trials) at the end of the condition identifier (it is called "conditions")


    paradigm_allruns = []


    for currun in range(4):
        ### Keep only this run's data 
        run_mask = MetaData['run']==currun+1
        
        
        #### First fetch the onsets of interest and their name 
        #onsets = MetaData['onsetRunWise_imagine'][run_mask] - tr ### this is tm1
        onsets = MetaData['onsetRunWise_imagine'][run_mask]### this is the normal one which is supposed to trigger on the onset of Imagine (so AFTER the last chord)
        ### What we would expect from a univariate analysis is to have a super clear effect of P > I (tm1) when triggering on the Last chord because there si a chord for P and not for I
        
        
        ## Modify the categHarm vale for the Control trials so that it is not NaN
        
        ctr_trials = MetaData['blockType'][run_mask]=='C'
        curcategharm = MetaData['categHarm'][run_mask]
        curcategharm[ctr_trials] = 'Z'
        names = MetaData['blockType'][run_mask] + '_' + curcategharm + '_' + MetaData['subcategAcoust'][run_mask]
        durations = MetaData['onsetRunWise_sing'][run_mask] - MetaData['onsetRunWise_imagine'][run_mask] ### this is the normal one
        #durations = MetaData['onsetRunWise_sing'][run_mask] - MetaData['onsetRunWise_imagine'][run_mask] + tr 
        #durations[:] = 3* tr ### this is for tm1
        #durations[:] = 2* tr ### this is for the normal one
        

        #### Now fetch also the onsets of fixation cross
        onsets_cross = MetaData['onsetRunWise_cross'][run_mask]
        
        names_cross = MetaData['onsetRunWise_cross'][run_mask] ## Copying something random just to have the good size 
        names_cross[:] = 'cross'
        durations_cross = MetaData['onsetRunWise_play'][run_mask] - MetaData['onsetRunWise_cross'][run_mask]
        #durations_cross[:] = tr ### Actually putting just one TR duration for the cross, which means only modeling the onset

        #### Now fetch also the onsets of music
        onsets_play = MetaData['onsetRunWise_play'][run_mask]
        
        names_play = MetaData['onsetRunWise_play'][run_mask] ## Copying something random just to have the good size 
        names_play[:] = 'play'
        durations_play = MetaData['onsetRunWise_imagine'][run_mask] - MetaData['onsetRunWise_play'][run_mask] ### this is the normal one 
        #durations_play = MetaData['onsetRunWise_imagine'][run_mask] - MetaData['onsetRunWise_play'][run_mask]  - tr ### this is tm1
        #durations_play[:] = tr ### Testing with just one Tr for the play period, which means only modeling the onset 

        #### Now fetch also the onsets of sing
        onsets_sing = MetaData['onsetRunWise_sing'][run_mask]
        
        names_sing = MetaData['onsetRunWise_sing'][run_mask] ## Copying something random just to have the good size 
        names_sing[:] = 'sing'
        durations_sing = MetaData['onsetRunWise_ratingAppears'][run_mask] - MetaData['onsetRunWise_sing'][run_mask]


        ### And finally add the rating as a shared factor 
        onsets_rating = MetaData['onsetRunWise_ratingAppears'][run_mask]
        names_rating = MetaData['onsetRunWise_cross'][run_mask] ## Copying something random just to have the good size
        durations_rating = MetaData['onsetRunWise_ratingMade'][run_mask] - MetaData['onsetRunWise_ratingAppears'][run_mask]
        names_rating[:] = 'rating'
        
        paradigm = pd.DataFrame({'trial_type': pd.concat([names_cross,names_play,names,names_sing,names_rating]), 
                                'onset': pd.concat([onsets_cross,onsets_play,onsets,onsets_sing,onsets_rating]),
                                'duration' : pd.concat([durations_cross,durations_play,durations,durations_sing,durations_rating])})
        
        ### Export paradigm for reference 
        filename = cursubj + '_paradigm_run' + str(currun+1) + '.csv'
        filepath = os.path.join(betapath,filename)
        paradigm.sort_values(by='onset').to_csv(filepath,index=False)

        paradigm_allruns.append(paradigm)


    return paradigm_allruns


print("Fetching fmri data...")
imgs,masks,confounds,frame_times = fetch_imgs_confounds(datacleanpath,cursubj)
paradigm_allruns = paradigm_modelI(behavpath,cursubj,betapath)

compute_matrix_contrast_betas(paradigm_allruns,imgs,masks,confounds,frame_times,cursubj,betapath,smoothing=None,univariate=True,calc_residuals = None)

print("Success!!") 