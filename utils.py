import pandas as pd
import os
import numpy as np 
from matplotlib import pyplot as plt
from nilearn.plotting import plot_stat_map,view_img
from nilearn.plotting import plot_roi
from nilearn.input_data import NiftiMasker
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
from sklearn.model_selection import permutation_test_score,cross_val_score,cross_validate
from sklearn.decomposition import PCA
import sys
import glob
from sklearn.pipeline import Pipeline
from joblib import Memory,dump
from sklearn.metrics import pairwise_distances
from nilearn.decoding import SpaceNetClassifier

#from nistats.first_level_model import FirstLevelModel
from nilearn.glm.first_level import FirstLevelModel
from nilearn.input_data import NiftiMasker
#from nistats.design_matrix import make_first_level_design_matrix as make_design_matrix
from nilearn.glm.first_level import make_first_level_design_matrix as make_design_matrix

from nilearn.image import load_img,iter_img
from nibabel import load

tr = 2.0

def fetch_imgs_motions(datacleanpath,cursubj):
    # imgs,motions,frame_times = fetch_imgs_motions(datacleanpath,cursubj)
    ### also calculates fd
    global tr
    # Generate file paths of images for all runs 
    imgs = []
    motions=[]
    for currun in range(4):

        imgs.append(os.path.join(datacleanpath,"func_%s_run%d.nii.gz" % (cursubj,currun+1)))
        motions.append(np.load(os.path.join(datacleanpath,"func_%s_run%d.npz" % (cursubj,currun+1)))['motion'])

    # Just fetch the number of TRs in files (frame times)

    frame_times = []

    

    for currun in range(4):        
        n_scans = load(imgs[currun]).shape[3]
        frame_times.append(np.arange(n_scans) * tr)

    return imgs,motions,frame_times

def fetch_imgs_confounds(datacleanpath,cursubj):
    # imgs,masks,confounds,frame_times = fetch_imgs_confounds(datacleanpath,cursubj)
    from load_confounds import Confounds
    from nilearn.image import clean_img,load_img
    import pandas as pd 
    import numpy as np

    global tr
    # Generate file paths of images for all runs 
    imgs = []
    confounds=[]
    masks=[]
    for currun in range(4):

        imgs.append(os.path.join(datacleanpath,"func/sub-%s_task-imagery_run-%d_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" % (cursubj,currun+1)))
        masks.append(os.path.join(datacleanpath,"func/sub-%s_task-imagery_run-%d_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz" % (cursubj,currun+1)))

        tsvfile = os.path.join(datacleanpath,"func/sub-%s_task-imagery_run-%d_desc-confounds_timeseries.tsv" % (cursubj,currun+1))
        conf = Confounds(strategy=['high_pass', 'motion', 'global','wm_csf','compcor'])

        Df_aroma = pd.read_csv(tsvfile,sep='\t').filter(regex='aroma').to_numpy()

        conf_array = np.hstack([conf.load(tsvfile),Df_aroma])
        confounds.append(conf_array)

    # Just fetch the number of TRs in files (frame times)

    frame_times = []
    
    for currun in range(4):        
        n_scans = load(imgs[currun]).shape[3]
        frame_times.append(np.arange(n_scans) * tr)

    return imgs,masks,confounds,frame_times


def fetch_ica_ts(cursubj,datacleanpath,basedir):
    from sklearn.preprocessing import normalize

    icapath = os.path.join(basedir,'ica',cursubj)

    comp_signal = os.path.join(icapath,'ica_ts.txt')

    _,motions_ts,_ = fetch_imgs_motions(datacleanpath,cursubj)

    motions_ts1 = normalize(motions_ts[0],axis=0)

    motions_ts2 = normalize(motions_ts[1],axis=0)

    motions_ts3 = normalize(motions_ts[2],axis=0)

    #motions_ts4 = normalize(motions_ts[3],axis=0)

    ica_ts = []
    # Extract ICA timeseries values
    data = np.loadtxt(comp_signal)
    ncomp = data.shape[0]

    icalistfile = os.path.join(icapath,'ica_list.txt')
    if not(os.path.isfile(icalistfile)):
        print("Manual inspection was not done for subject {}".format(cursubj))
        nbad=0
    else:
        # Extract which components to add as nuisance regressor from text file
        badcomps = np.loadtxt(icalistfile,dtype=bool)
        #print(badcomps)
        #print(data.shape,motions_ts1.shape)
        data = data[badcomps]
        nbad = data.shape[0]
        print("{} good components remaining, regressing {} ICA correlation time courses".format(ncomp-nbad,nbad))
        ica_ts.append(data[:,:motions_ts1.shape[0]].T)
        ica_ts.append( data[:,motions_ts1.shape[0]:(motions_ts1.shape[0]+motions_ts2.shape[0])].T)
        ica_ts.append(data[:,(motions_ts1.shape[0]+motions_ts2.shape[0]):(motions_ts1.shape[0]+motions_ts2.shape[0]+motions_ts3.shape[0])].T)
        ica_ts.append(data[:,(motions_ts1.shape[0]+motions_ts2.shape[0]+motions_ts3.shape[0]):].T)

        #l = [(aa.shape) for aa in ica_ts]
        #m = [(aa.shape) for aa in motions_ts]
        #print(l)
        #print(m)
    return ica_ts,nbad





def compute_matrix_contrast_betas(paradigm_allruns,imgs,masks,confounds,frame_times,cursubj,betapath,smoothing,univariate=False,calc_residuals = None):

    global tr    

    hrf_model = 'spm + derivative + dispersion'
    drift_model = None
    drift_order = 3

    """ hrf_model = 'glover' 
    drift_model = 'polynomial'
    drift_order = 3

    hrf_model = 'glover + derivative + dispersion' 
    drift_model = 'polynomial'
    drift_order = 3  """


    """ hrf_model = 'fir' 
    drift_model = 'polynomial'
    drift_order = 3 
    fir_delays=[1, 2, 3]
 """

    ### Compute betas
    if not(univariate):
        n_reg = []
        for ses in range(4):
            
            masker = NiftiMasker(mask_img=masks[ses],standardize_confounds=False,smoothing_fwhm=smoothing,standardize=False,detrend=False,low_pass=None,high_pass=None,t_r=tr)
            masker.fit(imgs[ses])

            fmri_glm = FirstLevelModel(tr, noise_model='ar1',mask_img = masker,
                                drift_model=drift_model,standardize=True,
                                drift_order=drift_order,hrf_model=hrf_model,minimize_memory=False)

            print("Estimating GLM run {}...".format(ses+1))
            fmri_glm.fit(run_imgs=imgs[ses],events=paradigm_allruns[ses],confounds=[pd.DataFrame(confounds[ses])])

            design_matrix = fmri_glm.design_matrices_[0]     

            ## save number of regressors
            n_reg.append(design_matrix.shape[1])

            if False:
                from nilearn.plotting import plot_design_matrix
                plot_design_matrix(design_matrix)
                plt.show()

            contrasts = []
            contrast_matrix = np.eye(design_matrix.shape[1])
            contrasts =  dict([(column, contrast_matrix[jj])
                        for jj, column in enumerate(design_matrix.columns)])

            
            contrasts = {
            "D_F1_p0": (contrasts["D_F1_p0"]),
            "D_F2_p0": (contrasts["D_F2_p0"]),
            "D_C1_p0": (contrasts["D_C1_p0"]),
            "D_C2_p0": (contrasts["D_C2_p0"]),
            "T_F1_p0": (contrasts["T_F1_p0"]),
            "T_F2_p0": (contrasts["T_F2_p0"]),
            "T_C1_p0": (contrasts["T_C1_p0"]),
            "T_C2_p0": (contrasts["T_C2_p0"]),
            "D_F1_i0": (contrasts["D_F1_i0"]),
            "D_F2_i0": (contrasts["D_F2_i0"]),
            "D_C1_i0": (contrasts["D_C1_i0"]),
            "D_C2_i0": (contrasts["D_C2_i0"]),
            "T_F1_i0": (contrasts["T_F1_i0"]),
            "T_F2_i0": (contrasts["T_F2_i0"]),
            "T_C1_i0": (contrasts["T_C1_i0"]),
            "T_C2_i0": (contrasts["T_C2_i0"]),

            "D_F1_p1": (contrasts["D_F1_p1"]),
            "D_F2_p1": (contrasts["D_F2_p1"]),
            "D_C1_p1": (contrasts["D_C1_p1"]),
            "D_C2_p1": (contrasts["D_C2_p1"]),
            "T_F1_p1": (contrasts["T_F1_p1"]),
            "T_F2_p1": (contrasts["T_F2_p1"]),
            "T_C1_p1": (contrasts["T_C1_p1"]),
            "T_C2_p1": (contrasts["T_C2_p1"]),
            "D_F1_i1": (contrasts["D_F1_i1"]),
            "D_F2_i1": (contrasts["D_F2_i1"]),
            "D_C1_i1": (contrasts["D_C1_i1"]),
            "D_C2_i1": (contrasts["D_C2_i1"]),
            "T_F1_i1": (contrasts["T_F1_i1"]),
            "T_F2_i1": (contrasts["T_F2_i1"]),
            "T_C1_i1": (contrasts["T_C1_i1"]),
            "T_C2_i1": (contrasts["T_C2_i1"]),
            "C_c0": (contrasts["C_c0"]),
            "F_c0": (contrasts["F_c0"]),
            "C_c1": (contrasts["C_c1"]),
            "F_c1": (contrasts["F_c1"]),            
            }

            
                
            if calc_residuals:
                print("Calculating Residuals run {}...".format(ses+1))
                res_map = fmri_glm.residuals[0]
                filename_res = 'residuals_run{}.nii.gz'.format(ses+1)
                res_map.to_filename(os.path.join(betapath,filename_res))

            print("Calculating All Betas run {}...".format(ses+1))
            for contrast_id, contrast_val in contrasts.items():
                
                #print("\tcontrast val: %s" % contrast_val)
                
                z_map = fmri_glm.compute_contrast(
                    contrast_val, output_type='z_score',stat_type = 't')

                filename = str(contrast_id) + '_beta_' + cursubj + '_run' + str(ses+1) + '.nii.gz'

                filepath = os.path.join(betapath,filename)

                #print(filepath)

                z_map.to_filename(filepath)
            
        print("Saving number of regressors for DoF calculation...")
        np.savez_compressed(os.path.join(betapath,"n_reg.npz"),nreg = n_reg)



    if univariate:
        
        from nilearn.masking import intersect_masks
        from nilearn.glm import threshold_stats_img
        ## We consider the intersection across runs of the brain masks        
        mask_inter = intersect_masks(mask_imgs=masks,threshold=1,connected=True)

        masker = NiftiMasker(mask_img=mask_inter,standardize_confounds=False,smoothing_fwhm=smoothing,standardize=False,detrend=False,low_pass=None,high_pass=None,t_r=tr)
        masker.fit(imgs)

        fmri_glm = FirstLevelModel(tr, noise_model='ar1',mask_img = masker,
                            standardize=True,drift_model=drift_model,
                            drift_order=drift_order,hrf_model=hrf_model)

        print("Calculating GLM across runs ...")
        fmri_glm.fit(run_imgs=imgs,events=paradigm_allruns,confounds=[pd.DataFrame(confounds[i]) for i in range(4)])
        print("Calculating Fixed effects Contrasts ...")
        
        contrasts_dict = {
            "P_TvsD": "P_T_C + P_T_F - P_D_C - P_D_F",
            "C_CvsF": "C_Z_C - C_Z_F",            
            "P_CvsF": "P_T_C + P_D_C - P_T_F - P_D_F",
            "I_TvsD": "I_T_C + I_T_F - I_D_C - I_D_F",
            "I_CvsF": "I_T_C + I_D_C - I_T_F - I_D_F",
            "PsupI": "P_T_C + P_T_F - I_T_C - I_T_F",
            "IsupP": "-P_T_C - P_T_F + I_T_C + I_T_F",
            }

        for contrast_id,contrast_val in contrasts_dict.items():

            z_map = fmri_glm.compute_contrast(contrast_val, output_type='z_score',stat_type = 't')
            filename = contrast_id + '_' + cursubj + '.nii.gz'
            filepath = os.path.join(betapath,filename)
            if False:
                zmap_thr,thr = threshold_stats_img(z_map,mask_inter)
                view=view_img(zmap_thr,title=contrast_id + cursubj,threshold=thr,height_control='fpr')
                view.open_in_browser()
            z_map.to_filename(filepath)        


    print('Success ! ')
    return True

def fetch_X_y(betapath):

    # (img_P_cat,y_P_cat),(img_I_cat,y_I_cat),(img_P_acou,y_P_acou),(img_I_acou,y_I_acou) = fetch_X_y(betapath)
    ### Gathering all betas and building target (y) vectors

    """ img_C_C = glob.glob(os.path.join(betapath,'C_Z_C*.nii.gz'))

    img_C_F = glob.glob(os.path.join(betapath,'C_Z_F*.nii.gz'))

    img_C = img_C_F + img_C_C

    y_control= np.hstack([np.ones(len(img_C_F)) ,2*np.ones(len(img_C_C))])
 """

    img_P_D = glob.glob(os.path.join(betapath,'P_D_*.nii.gz'))

    img_P_T = glob.glob(os.path.join(betapath,'P_T_*.nii.gz'))

    img_P_cat = img_P_D + img_P_T

    y_P_cat= np.hstack([1*np.ones(len(img_P_D)) ,2*np.ones(len(img_P_T))])

    img_I_D = glob.glob(os.path.join(betapath,'I_D_*.nii.gz'))

    img_I_T = glob.glob(os.path.join(betapath,'I_T_*.nii.gz'))

    img_I_cat = img_I_D + img_I_T

    y_I_cat= np.hstack([1*np.ones(len(img_I_D)) ,2*np.ones(len(img_I_T))])

    img_P_C = glob.glob(os.path.join(betapath,'P_*_C*.nii.gz'))

    img_P_F = glob.glob(os.path.join(betapath,'P_*_F*.nii.gz'))

    img_P_acou = img_P_C + img_P_F

    y_P_acou = np.hstack([3*np.ones(len(img_P_C)) ,4*np.ones(len(img_P_F))])

    img_I_C = glob.glob(os.path.join(betapath,'I_*_C*.nii.gz'))

    img_I_F = glob.glob(os.path.join(betapath,'I_*_F*.nii.gz'))

    img_I_acou = img_I_C + img_I_F

    y_I_acou = np.hstack([3*np.ones(len(img_I_C)) ,4*np.ones(len(img_I_F))])

    return (img_P_cat,y_P_cat),(img_I_cat,y_I_cat),(img_P_acou,y_P_acou),(img_I_acou,y_I_acou)

def fetch_run(betaimgs):
    ### returns a list of run numbers given a list of betaimgs calculate by function compute_matrix_contrast_betas
    l = [curfile[-8] for curfile in betaimgs]

    return l


def fetch_anat(cursubj,basedir):

    basefolder = os.path.join(basedir,'903_data','903_{}'.format(cursubj))
    filelist = []
    for root, _, files in os.walk(basefolder, topdown=True):
        
        for name in files:
            _,ext = os.path.splitext(name)
            if ext=='.nii':
                if name[:2] == 's1':
                    
                    if 'aa_t1' in root:
                        #print(root,name)
                        filelist.append(os.path.join(root, name))


    return(filelist[0])


def fetch_tissues(cursubj,basedir):
# gm,wm,csf = fetch_tissues(cursubj,basedir)
    basefolder = os.path.join(basedir,'903_data','903_{}'.format(cursubj))
    filelist = []
    for root, _, files in os.walk(basefolder, topdown=True):
        
        for name in files:
            _,ext = os.path.splitext(name)
            if ext=='.nii':
                if 'aa_t1' in root:
                    if name[:2] == 'c1':
                    
                    
                        #print(root,name)
                        gm=os.path.join(root, name)
                    
                    if name[:2] == 'c2':
                    
                    
                        #print(root,name)
                        wm=os.path.join(root, name)

                    
                    if name[:2] == 'c3':
                    
                    
                        #print(root,name)
                        csf=os.path.join(root, name)


    return gm,wm,csf
def get_betas(condname,betapath,cursubj):    
    return load_img([((os.path.join(betapath,f"{condname}{rep}_beta_{cursubj}_run{crun}.nii.gz"))) for crun,rep in zip([1,1,2,2,3,3,4,4],[0,1,0,1,0,1,0,1])])
