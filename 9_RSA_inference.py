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
now = datetime.datetime.now()

rsapath = os.path.join('/media/nfarrugi/datapal/imagery-mvpa','results','rsa_crossnobis')

roilist =  ['R_PSMcor_p_rsa.npz', 'R_HSgyr_rsa.npz', 'L_PSMcor_p_rsa.npz', 'L_HSgyr_rsa.npz', 'R_STgyr_m_rsa.npz',  'L_STgyr_m_rsa.npz', 'R_PSMcor_a_rsa.npz', 'L_PSMcor_a_rsa.npz']

now = datetime.datetime.now()

resultspath = os.path.join('/media/nfarrugi/datapal/results','RESULTS_RSA_{}'.format(now.strftime("%Y-%m-%d_%H-%M")))

if not(os.path.isdir(resultspath)):
    os.makedirs(resultspath)


### Experimental design
trials = ['T_C1_p','T_C1_i','T_C2_p','T_C2_i','T_F1_p','T_F1_i','T_F2_p','T_F2_i','D_C1_p','D_C1_i','D_C2_p','D_C2_i','D_F1_p','D_F1_i','D_F2_p','D_F2_i']
trialsvector =  np.repeat(trials,8)

runvector = np.tile([1,2,3,4],32)

### Model RDMs

stimuli = ['T_C1','T_C2','T_F1','T_F2','D_C1','D_C2','D_F1','D_F2']
stimvector = np.repeat(stimuli,2)
condition = ['P','I']
conditionvector = np.tile(condition,8)
harmonic = ['T','D']
harmonicvector = np.repeat(harmonic,8)
surface = ['C ','F#']
surfacevector = np.tile(np.repeat(surface,4),2)
tonality = ['C ','F#','F ','B ']
tonalityvector = np.repeat(tonality,4)

harm_rdm = pyrsa.rdm.get_categorical_rdm(category_vector=harmonicvector,category_name='harmony')
surf_rdm = pyrsa.rdm.get_categorical_rdm(category_vector=surfacevector,category_name='Surface')
modality_rdm = pyrsa.rdm.get_categorical_rdm(category_vector=conditionvector,category_name='Condition')
stim_rdm = pyrsa.rdm.get_categorical_rdm(category_vector=stimvector,category_name='stimulus')
tonality_rdm = pyrsa.rdm.get_categorical_rdm(category_vector=tonalityvector,category_name='Key')


### Plotting of Theoretical RDM 

if True:
    pyrsa.vis.show_rdm(harm_rdm,filename=os.path.join(resultspath,'rdm_harmony.svg'),pattern_descriptor='harmony')
    pyrsa.vis.show_rdm(modality_rdm,filename=os.path.join(resultspath,'rdm_condition.svg'),pattern_descriptor='Condition')
    pyrsa.vis.show_rdm(tonality_rdm,filename=os.path.join(resultspath,'rdm_tonality.svg'),pattern_descriptor='Key')
    pyrsa.vis.show_rdm(surf_rdm,filename=os.path.join(resultspath,'rdm_chord.svg'),pattern_descriptor='Surface')

#pyrsa.vis.show_rdm(stim_rdm,filename=os.path.join(resultspath,'rdm_stim.svg'),pattern_descriptor='stimulus')
models = []
#for name,rdm_m in zip(['stimuli','modality','surface','harmony',],[stim_rdm,modality_rdm,surf_rdm,harm_rdm,tonality_rdm]):
#for name,rdm_m in zip(['Condition','Surface','Harmony','Key','Stimuli'],[modality_rdm,surf_rdm,harm_rdm,tonality_rdm,stim_rdm]):
for name,rdm_m in zip(['Surface','Harmony','Key','Stimuli'],[surf_rdm,harm_rdm,tonality_rdm,stim_rdm]):
#for name,rdm_m in zip(['surface','harmony','tonality'],[surf_rdm,harm_rdm,tonality_rdm]):
    m = pyrsa.model.ModelFixed(name, rdm_m)
    models.append(m)



### plotting of a few neural RDM 
if True:
    ### Neural RDM 
    for roi in roilist:
        all_neural_rdms = []
        roititle = roi[:-7]

        ## Load all RDM (output of 11e_RSA_crossnobis.py)
        for cursubj in (os.listdir(rsapath))[:3]:
            rdm = np.load(os.path.join(rsapath,cursubj,roi))['rdm']
            currdm = pyrsa.rdm.RDMs(rdm,rdm_descriptors={'subject':cursubj},pattern_descriptors={'trials':trials})
            #pyrsa.vis.show_rdm(currdm,filename=os.path.join(resultspath,f"{roititle}_{cursubj}.png"))
            pyrsa.vis.show_rdm(currdm,filename=os.path.join(resultspath,f"{roititle}_{cursubj}.svg"),pattern_descriptor='trials',show_colorbar=True)


metric = 'cosine'
nperm=1000
savenp = False
### Neural RDM 
for roi in roilist:
    all_neural_rdms = []
    roititle = roi[:-7]

    ## Load all RDM (output of 11e_RSA_crossnobis.py)
    for cursubj in (os.listdir(rsapath)):
        rdm = np.load(os.path.join(rsapath,cursubj,roi))['rdm']
        currdm = pyrsa.rdm.RDMs(rdm,rdm_descriptors={'subject':cursubj})
        all_neural_rdms.append(currdm)
        #pyrsa.vis.show_rdm(currdm,filename=os.path.join(resultspath,f"{roititle}_{cursubj}.png"))
        #pyrsa.vis.show_rdm(currdm,filename=os.path.join(resultspath,f"{roititle}_{cursubj}.svg"))


    all_neural_rdms = pyrsa.rdm.concat(all_neural_rdms)

    

    #results = pyrsa.inference.eval_fixed(models, all_neural_rdms, method=metric)
    #pyrsa.vis.plot_model_comparison(results,savepath=os.path.join(resultspath,f"{roi[:-4]}_fixed.png"),test_pair_comparisons='nili',test_above_0='icicles',test_below_noise_ceil='icicles')

    ### Bootstrapping using patterns
    results = pyrsa.inference.eval_bootstrap_pattern(models, all_neural_rdms, method=metric,N=nperm)
    evaluations,perf,errorbar_low,errorbar_high = pyrsa.vis.plot_model_comparison(results,savepath=os.path.join(resultspath,f"{roi[:-4]}_bootstrap_pattern.svg"),test_pair_comparisons='nili',test_above_0='icicles',test_below_noise_ceil='icicles',error_bars='sem')
    
    noise_lower = np.repeat(np.nanmean(results.noise_ceiling[0]),len(models))
    noise_upper = np.repeat(np.nanmean(results.noise_ceiling[1]),len(models))
    A = np.stack([perf,errorbar_low,errorbar_high,noise_lower,noise_upper])
    if savenp:
        Df = pd.DataFrame(A,index = ['correlation','err_low','err_high','noise_lower','noise_upper'],columns=['condition','chord','harmony','key'])
        Df.to_csv(os.path.join(resultspath,f"{roi[:-4]}_bootstrap_pattern.csv"))

    ### Bootstrapping usng subjects
    results = pyrsa.inference.eval_bootstrap_rdm(models, all_neural_rdms, method=metric,N=nperm)
    evaluations,perf,errorbar_low,errorbar_high = pyrsa.vis.plot_model_comparison(results,savepath=os.path.join(resultspath,f"{roi[:-4]}_bootstrap_rdm.svg"),test_pair_comparisons='nili',test_above_0='icicles',test_below_noise_ceil='icicles',error_bars='sem')

    noise_lower = np.repeat(np.nanmean(results.noise_ceiling[0]),len(models))
    noise_upper = np.repeat(np.nanmean(results.noise_ceiling[1]),len(models))
    A = np.stack([perf,errorbar_low,errorbar_high,noise_lower,noise_upper])
    if savenp:
        Df = pd.DataFrame(A,index = ['correlation','err_low','err_high','noise_lower','noise_upper'],columns=['condition','chord','harmony','key'])
        Df.to_csv(os.path.join(resultspath,f"{roi[:-4]}_bootstrap_rdm.csv"))