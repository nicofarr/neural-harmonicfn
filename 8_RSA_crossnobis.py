import numpy as np
import pandas as pd 
from scipy import io
import matplotlib.pyplot as plt
import pyrsa
import datetime
import sys,os
import nibabel as nib
from nilearn.image import load_img
from nilearn.input_data import NiftiMasker
from utils import get_betas
now = datetime.datetime.now()

cursubj = np.str(sys.argv[1])

betapath =  os.path.join('/media/nfarrugi/datapal/','results','glm_M',cursubj)

rsapath = os.path.join('/media/nfarrugi/datapal/','results','rsa_crossnobis',cursubj)

rerun = True 

if not(os.path.isdir(rsapath)):
    os.makedirs(rsapath)
elif (not(rerun) and os.path.isfile(os.path.join(rsapath,'rsa.csv'))):
    raise NameError("This subject was already processed and rerun is False")


T_C1_p = get_betas('T_C1_p',betapath,cursubj)
T_C1_i = get_betas('T_C1_i',betapath,cursubj)
T_C2_p = get_betas('T_C2_p',betapath,cursubj)
T_C2_i = get_betas('T_C2_i',betapath,cursubj)

T_F1_p = get_betas('T_F1_p',betapath,cursubj)
T_F1_i = get_betas('T_F1_i',betapath,cursubj)
T_F2_p = get_betas('T_F2_p',betapath,cursubj)
T_F2_i = get_betas('T_F2_i',betapath,cursubj)

D_C1_p = get_betas('D_C1_p',betapath,cursubj)
D_C1_i = get_betas('D_C1_i',betapath,cursubj)
D_C2_p = get_betas('D_C2_p',betapath,cursubj)
D_C2_i = get_betas('D_C2_i',betapath,cursubj)

D_F1_p = get_betas('D_F1_p',betapath,cursubj)
D_F1_i = get_betas('D_F1_i',betapath,cursubj)
D_F2_p = get_betas('D_F2_p',betapath,cursubj)
D_F2_i = get_betas('D_F2_i',betapath,cursubj)

C_c = get_betas('C_c',betapath,cursubj)
F_c = get_betas('F_c',betapath,cursubj)

listimg = [T_C1_p,T_C1_i,T_C2_p,T_C2_i,T_F1_p,T_F1_i,T_F2_p,T_F2_i,D_C1_p,D_C1_i,D_C2_p,D_C2_i,D_F1_p,D_F1_i,D_F2_p,D_F2_i]

residuals = [((os.path.join(betapath,f"residuals_run{crun}.nii.gz"))) for crun in ([1,2,3,4])]

### Experimental design
trials = ['T_C1_p','T_C1_i','T_C2_p','T_C2_i','T_F1_p','T_F1_i','T_F2_p','T_F2_i','D_C1_p','D_C1_i','D_C2_p','D_C2_i','D_F1_p','D_F1_i','D_F2_p','D_F2_i']
trialsvector =  np.repeat(trials,8)

runvector = np.tile([1,2,3,4],32)

### Model RDMs

stimuli = ['T_C1','T_C2','T_F1','T_F2','D_C1','D_C2','D_F1','D_F2']
stimvector = np.repeat(stimuli,2)
condition = ['P','I']
conditionvector = np.tile(condition,8)
harmonic = ['Tonic','Dominant']
harmonicvector = np.repeat(harmonic,8)
surface = ['C#','F#']
surfacevector = np.tile(np.repeat(surface,4),2)

harm_rdm = pyrsa.rdm.get_categorical_rdm(category_vector=harmonicvector,category_name='harmony')
surf_rdm = pyrsa.rdm.get_categorical_rdm(category_vector=surfacevector,category_name='surface')
modality_rdm = pyrsa.rdm.get_categorical_rdm(category_vector=conditionvector,category_name='condition')
stim_rdm = pyrsa.rdm.get_categorical_rdm(category_vector=stimvector,category_name='stimulus')

visu_rdm = True
if visu_rdm:
    pyrsa.vis.show_rdm(harm_rdm,filename='rdm_harm.png',pattern_descriptor='harmony')
    pyrsa.vis.show_rdm(surf_rdm,filename='rdm_surf.png',pattern_descriptor='surface')
    pyrsa.vis.show_rdm(modality_rdm,filename='rdm_modality.png',pattern_descriptor='condition')
    pyrsa.vis.show_rdm(stim_rdm,filename='rdm_stimulus.png',pattern_descriptor='stimulus')

### Loop over rois 
df_subject = {}
roilist = os.listdir('rois')
for curroi in roilist:
    roititle = curroi[:-7]
    print(roititle)

    ## Prepare ROI masker 
    masker = NiftiMasker(mask_img=os.path.join('rois',curroi))
    masker.fit()

    ## Mask the data (extract only ROI voxels)
    print("Masking betas")
    data = np.vstack([masker.transform(curimg) for curimg in listimg])

    ## mask the residuals (extract only ROI voxels)
    print("Masking Residuals")
    res_masked = ([masker.transform(residuals[i]) for i in range(4)])
    
    ## Calculate the precision matrix
    dof=res_masked[0].shape[0] - np.load(os.path.join(betapath,"n_reg.npz"))['nreg'] ## number of degrees of freedom of the regressors
    precisions = pyrsa.data.noise.prec_from_residuals(res_masked,dof=dof)
    
    ## Prepare the structure for pyrsa

    RsaDataset = pyrsa.data.Dataset(data,obs_descriptors={'run':runvector,'trials':trialsvector})

    ## Calculate the Neural RDM using Cross Validated Mahalanobis Distance, using Run as cross validation
    obs_descriptor = 'trials'
    neural_rdm =  pyrsa.rdm.calc_rdm_crossnobis(RsaDataset,obs_descriptor,noise=precisions,cv_descriptor='run')

    ## Save the matrix
    np.savez_compressed(os.path.join(rsapath,f"{roititle}_rsa.npz"),rdm=neural_rdm.get_matrices())

    ## Compare the Neural RDM with Model RDMS
    dict_results = {'harmony' : pyrsa.rdm.compare_cosine(neural_rdm,harm_rdm),
    'surface' : pyrsa.rdm.compare_cosine(neural_rdm,surf_rdm),
    'modality' : pyrsa.rdm.compare_cosine(neural_rdm,modality_rdm),
    'stimulus': pyrsa.rdm.compare_cosine(neural_rdm,stim_rdm)}

    df_subject[roititle] = dict_results

Df = pd.DataFrame(df_subject)

Df.to_csv(os.path.join(rsapath,'rsa.csv'))
