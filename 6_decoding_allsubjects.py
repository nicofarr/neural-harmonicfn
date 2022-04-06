import numpy as np
import pandas as pd 
from scipy import io
import matplotlib.pyplot as plt
import pyrsa
import datetime
import sys,os
import nibabel as nib
from nilearn.image import load_img
from nilearn.plotting import view_img
from nilearn.input_data import NiftiMasker
from utils import get_betas
from nilearn.image import concat_imgs
from sklearn.model_selection import LeaveOneGroupOut
from nilearn.decoding import FREMClassifier,Decoder
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
now = datetime.datetime.now()

### For now RSA is only done with ModelM
betapath =  os.path.join('/media/nfarrugi/datapal/imagery-mvpa','results','glm_M')

#secondlevelpath = os.path.join('/media/nfarrugi/datapal/results','RESULTS_GLM_{}'.format(now.strftime("%Y-%m-%d_%H-%M")))

decodingpath = os.path.join('/media/nfarrugi/datapal/','results','decoding_allsubj_logistic_{}'.format(now.strftime("%Y-%m-%d_%H-%M")))

rerun = False 

if not(os.path.isdir(decodingpath)):
    os.makedirs(decodingpath)
elif (not(rerun) and os.path.isfile(os.path.join(decodingpath,'rsa.csv'))):
    raise NameError("This subject was already processed and rerun is False")

def get_betas_str(condname,betapath,cursubj):    
    return [((os.path.join(betapath,f"{condname}{rep}_beta_{cursubj}_run{crun}.nii.gz"))) for crun,rep in zip([1,1,2,2,3,3,4,4],[0,1,0,1,0,1,0,1])]


def get_allbetas_subj(betapath,cursubj):
    T_C1_p = get_betas_str('T_C1_p',betapath,cursubj)
    T_C1_i = get_betas_str('T_C1_i',betapath,cursubj)
    T_C2_p = get_betas_str('T_C2_p',betapath,cursubj)
    T_C2_i = get_betas_str('T_C2_i',betapath,cursubj)

    T_F1_p = get_betas_str('T_F1_p',betapath,cursubj)
    T_F1_i = get_betas_str('T_F1_i',betapath,cursubj)
    T_F2_p = get_betas_str('T_F2_p',betapath,cursubj)
    T_F2_i = get_betas_str('T_F2_i',betapath,cursubj)

    D_C1_p = get_betas_str('D_C1_p',betapath,cursubj)
    D_C1_i = get_betas_str('D_C1_i',betapath,cursubj)
    D_C2_p = get_betas_str('D_C2_p',betapath,cursubj)
    D_C2_i = get_betas_str('D_C2_i',betapath,cursubj)

    D_F1_p = get_betas_str('D_F1_p',betapath,cursubj)
    D_F1_i = get_betas_str('D_F1_i',betapath,cursubj)
    D_F2_p = get_betas_str('D_F2_p',betapath,cursubj)
    D_F2_i = get_betas_str('D_F2_i',betapath,cursubj)

    C_c = get_betas_str('C_c',betapath,cursubj)
    F_c = get_betas_str('F_c',betapath,cursubj)

    listimg_P = ([T_C1_p,T_C2_p,T_F1_p,T_F2_p,D_C1_p,D_C2_p,D_F1_p,D_F2_p])
    listimg_I = ([T_C1_i,T_C2_i,T_F1_i,T_F2_i,D_C1_i,D_C2_i,D_F1_i,D_F2_i])
    listimg_C = [C_c,F_c]
    return listimg_P,listimg_I,listimg_C







allsubj = os.listdir(betapath)
all_subj_beta_P = [get_allbetas_subj(os.path.join(betapath,subject),subject)[0] for subject in allsubj]
all_subj_beta_I = [get_allbetas_subj(os.path.join(betapath,subject),subject)[1] for subject in allsubj]
all_subj_beta_C = [get_allbetas_subj(os.path.join(betapath,subject),subject)[2] for subject in allsubj]

all_subj_beta_P_fl = (([item for subjectlist in all_subj_beta_P for item in subjectlist]))
listimg_P = (([item for subjectlist in all_subj_beta_P_fl for item in subjectlist]))

all_subj_beta_I_fl = (([item for subjectlist in all_subj_beta_I for item in subjectlist]))
listimg_I = (([item for subjectlist in all_subj_beta_I_fl for item in subjectlist]))

all_subj_beta_C_fl = (([item for subjectlist in all_subj_beta_C for item in subjectlist]))
listimg_C = (([item for subjectlist in all_subj_beta_C_fl for item in subjectlist]))

listimg_PvI = listimg_P + listimg_I

"""
if not(os.path.isfile(os.path.join(decodingpath,'list_img_I.nii.gz'))):
    listimg_P = concat_imgs(all_subj_beta_P_fl_fl)
    listimg_I = concat_imgs(all_subj_beta_I_fl_fl)
    listimg_I.to_filename(os.path.join(decodingpath,'list_img_I.nii.gz'))
    listimg_P.to_filename(os.path.join(decodingpath,'list_img_P.nii.gz'))
else:
    listimg_P = load_img(listimg_P)
    listimg_I = load_img(listimg_I)
"""

runvec = np.tile(np.tile([1,1,2,2,3,3,4,4],8),(35))
cv_subj = np.repeat(np.arange(35),64)
harm_label = np.tile(np.repeat(['T','D'],8*4),(35))
surf_label = np.tile(np.tile(np.repeat(['C','F#'],8*2),2),(35))
stim_label = np.tile(np.repeat(['T_C1','T_C2','T_F1','T_F2','D_C1','D_C2','D_F1','D_F2'],8),35)
tonality_label = np.tile(np.repeat(['Cmaj','F#maj','Fmaj','Bmaj'],16),35)
control_label = np.tile(np.repeat(['C','F#'],8),35)
PvI_label = np.repeat(['P','I'],64*35)
runvec_control = np.tile(np.tile([1,1,2,2,3,3,4,4],2),(35))
runvec_PvI = np.tile(runvec,(2))
screening = 100
smoothing = 5 
estimator = 'logistic' ### Initial results from early april 21 where with logistic
notdummy = False
logo = LeaveOneGroupOut()

def MakeConfusionMatrix(decoder,label,listimg,clustlabels = harm_label, savepath = '.',stri='test',debug=False):

    #filename = f"{stri}_confusion_run{i+1}"
    def plotConfConfig_(conf,contig,decoder,ax,filename,saveimg=False):
        ConfMat = ConfusionMatrixDisplay(conf,display_labels=decoder.classes_)
        ConfMat.plot(ax=ax[0])

        im = ax[1].imshow(contig)

        # Loop over data dimensions and create text annotations.
        for i in range(contig.shape[0]):
            for j in range(contig.shape[1]):
                text = ax[1].text(j, i, contig[i, j].round(decimals=2),
                            ha="center", va="center", color="w")

        # Create colorbar
        cbar = ax[1].figure.colorbar(im, ax=ax[1])

        # We want to show all ticks...
        ax[1].set_xticks(np.arange(contig.shape[1]))
        ax[1].set_yticks(np.arange(contig.shape[0]))
        # ... and label them with the respective list entries
        ax[1].set_xticklabels(ll.inverse_transform(np.arange(contig.shape[1])))
        ax[1].set_yticklabels(ll2.inverse_transform(np.arange(contig.shape[0])))

        plt.tight_layout()
        
        

        np.savez_compressed(os.path.join(savepath,filename + '.npz'),cm=conf,contig=contig)    
        if saveimg:        
            #plt.savefig(os.path.join(savepath,filename + '.png'))
            plt.savefig(os.path.join(savepath,filename + '.svg'))
        plt.close()
    
    ll = LabelEncoder()
    ll2 = LabelEncoder()
    conf_allrun=[]
    contig_allrun=[]
    for i,(train_ind,test_ind) in enumerate(decoder.cv_):
        y_true = ll.fit_transform(label[test_ind])
        clust_true = ll2.fit_transform(harm_label[test_ind])

        list_img_run = [listimg[t] for t in test_ind]
        y_pred = ll.transform(decoder.predict(list_img_run))

        

        f,ax = plt.subplots(nrows=2)
        
        conf = confusion_matrix(y_true,y_pred,normalize=None)
        conf_allrun.append(conf)

        contig = contingency_matrix(labels_true=clust_true,labels_pred=y_pred)
        #contig = contig / len(clust_true)

        contig_allrun.append(contig)

        if debug:
            print(y_true.shape)

            print(y_true[:15])
            print(y_pred[:15])
            print(clust_true[:15])
            

            print(ll.inverse_transform(y_true[:15]))
            print(ll.inverse_transform(y_pred[:15]))
            print(ll2.inverse_transform(clust_true[:15]))

            print(conf)
            print(contig)
        
        plotConfConfig_(conf,contig,decoder,ax,filename = f"{stri}_confusion_run{i+1}",saveimg=True)
    
    f,ax = plt.subplots(nrows=2)
    plotConfConfig_(np.mean(np.stack(conf_allrun),axis=0),np.mean(np.stack(contig_allrun),axis=0),decoder,ax,filename = f"{stri}_confusion_mean",saveimg=True)
    """
    ConfMat = ConfusionMatrixDisplay(conf,decoder.classes_)
    ConfMat.plot(ax=ax[0])

    im = ax[1].imshow(contig)

    # Create colorbar
    cbar = ax[1].figure.colorbar(im, ax=ax[1])

    # We want to show all ticks...
    ax[1].set_xticks(np.arange(contig.shape[1]))
    ax[1].set_yticks(np.arange(contig.shape[0]))
    # ... and label them with the respective list entries
    ax[1].set_xticklabels(ll.inverse_transform(np.arange(contig.shape[1])))
    ax[1].set_yticklabels(ll2.inverse_transform(np.arange(contig.shape[0])))

    plt.tight_layout()
    
    filename = f"{stri}_confusion_run{i+1}"

    np.savez_compressed(os.path.join(savepath,filename + '.npz'),cm=conf,contig=contig)            
    plt.savefig(os.path.join(savepath,filename + '.png'))
    plt.close()
    """

cv_vec = runvec
#cv_vec = cv_subj ### does not work with leave one subject out CV, not sure why ? 

n_jobs=1
doPvI = True
docontrol = True
dotonality = True

roilist = os.listdir('rois')

for curroi in roilist:
    roititle = curroi[:-7]
    print(f"ROI : {roititle}")

    if doPvI:
        decoder_PvI = Decoder(estimator=estimator,cv=logo,scoring='accuracy',screening_percentile=screening,standardize=False,smoothing_fwhm=smoothing,mask=curroi)
        decoder_PvI.fit(listimg_PvI,PvI_label,groups=runvec_PvI)
        acc = (np.mean([np.mean(curclas[1]) for curclas in decoder_PvI.cv_scores_.items()]))
        std = (np.mean([np.std(curclas[1]) for curclas in decoder_PvI.cv_scores_.items()]))
        np.savez_compressed(os.path.join(decodingpath,f"PvsI_{roititle}_scores.npz"),scores =decoder_PvI.cv_scores_ )
        print("Mean accuracy on P versus I : {}({})".format(acc,std))
        #print("All scores : ", decoder_PvI.cv_scores_)

        ### Confusion matrices 

        #MakeConfusionMatrix(decoder_C,control_label,listimg_C,savepath = decodingpath,stri=f"Control_{roititle}")
        
        if False:
            for curclas in decoder_PvI.classes_:
                coef = decoder_PvI.coef_img_[curclas]
                coef_str = f"{roititle}_{curclas}.nii.gz"

                coef_std = decoder_PvI.std_coef_img_[curclas]
                coef_std_str = f"{roititle}_{curclas}.nii.gz"
                
                coef.to_filename(os.path.join(decodingpath,coef_str))
                coef_std.to_filename(os.path.join(decodingpath,coef_std_str))
                
    
    
    if docontrol:
        decoder_C = Decoder(estimator=estimator,cv=logo,scoring='accuracy',screening_percentile=screening,standardize=False,smoothing_fwhm=smoothing,mask=curroi)
        decoder_C.fit(listimg_C,control_label,groups=runvec_control)
        acc_C = (np.mean([np.mean(curclas[1]) for curclas in decoder_C.cv_scores_.items()]))

        np.savez_compressed(os.path.join(decodingpath,f"C_{roititle}_scores.npz"),scores =decoder_C.cv_scores_ )

        print("Mean accuracy on Control : {}".format(acc_C))
        #print("All scores Control: ", decoder_C.cv_scores_)

        ### Confusion matrices 

        #MakeConfusionMatrix(decoder_C,control_label,listimg_C,savepath = decodingpath,stri=f"Control_{roititle}")
    
    if not(dotonality):
        continue
    else:
    
        for curlabel,cond in zip([harm_label,surf_label],['harmony','surface']):
            print(f"Current Condition : {cond}")
            decoder_P = Decoder(estimator=estimator,cv=logo,scoring='accuracy',screening_percentile=screening,standardize=False,smoothing_fwhm=smoothing,mask=curroi,n_jobs=n_jobs)
            decoder_I = Decoder(estimator=estimator,cv=logo,scoring='accuracy',screening_percentile=screening,standardize=False,smoothing_fwhm=smoothing,mask=curroi,n_jobs=n_jobs)
            

            decoder_P.fit(listimg_P,curlabel,groups=cv_vec)
            #MakeConfusionMatrix(decoder_P,curlabel,listimg_P,savepath = decodingpath,stri=f"{cond}_P_{roititle}")
            np.savez_compressed(os.path.join(decodingpath,f"{cond}_P_{roititle}_scores.npz"),scores =decoder_P.cv_scores_ )
            preds_I = decoder_P.predict(listimg_I)

            acc_P_train_P = (np.mean([np.mean(curclas[1]) for curclas in decoder_P.cv_scores_.items()]))
            acc_P_train_P_std = (np.mean([np.std(curclas[1]) for curclas in decoder_P.cv_scores_.items()]))
            
            acc_I_train_P = accuracy_score(curlabel,preds_I)
                    
            print("Training on Perception, Decoding on Perception : {}({})".format(acc_P_train_P,acc_P_train_P_std))
            #print("Training on Perception, Decoding on Imagery : {}".format(acc_I_train_P))
            #print("All scores Perception: ", decoder_P.cv_scores_)

            decoder_I.fit(listimg_I,curlabel,groups=cv_vec)
            #MakeConfusionMatrix(decoder_I,curlabel,listimg_I,savepath = decodingpath,stri=f"{cond}_I_{roititle}")
            preds_P = decoder_I.predict(listimg_P)
            acc_I_train_I = np.mean([np.mean(curclas[1]) for curclas in decoder_I.cv_scores_.items()])
            acc_I_train_I_std = (np.mean([np.std(curclas[1]) for curclas in decoder_I.cv_scores_.items()]))
            np.savez_compressed(os.path.join(decodingpath,f"{cond}_I_{roititle}_scores.npz"),scores =decoder_I.cv_scores_ )
            acc_P_train_I = accuracy_score(curlabel,preds_P)

            print("Training on Imagery, Decoding on Imagery : {}({})".format(acc_I_train_I,acc_I_train_I_std))
            #print("Training on Imagery, Decoding on Perception : {}".format(acc_P_train_I))
            #print("All scores Imagery: ", decoder_I.cv_scores_)

            visu = False
            
            #rfe_p = RFE(decoder_P,n_features_to_select=30,step=5)
            #rfe_p.fit(decoder_P.masker_.transform(listimg_P),curlabel)
            #ranking_P = decoder_P.masker_.inverse_transform(rfe_p.ranking_)
            #print(rfe_p.ranking_)

            #rfe_i = RFE(decoder_I,n_features_to_select=30,step=5)
            #rfe_i.fit(decoder_I.masker_.transform(listimg_I),curlabel)
            #ranking_I = decoder_I.masker_.inverse_transform(rfe_i.ranking_)
            #print(rfe_i.ranking_)

            #ranking_P.to_filename(os.path.join(decodingpath,f"{roititle}_ranks_P.nii.gz"))
            #ranking_I.to_filename(os.path.join(decodingpath,f"{roititle}_ranks_I.nii.gz"))

            if notdummy:
                for curclas in decoder_P.classes_:
                    P_coef = decoder_P.coef_img_[curclas]
                    P_coef_str = f"{roititle}_{curclas}_P.nii.gz"
                    I_coef = decoder_I.coef_img_[curclas]
                    I_coef_str = f"{roititle}_{curclas}_I.nii.gz"

                    P_coef_std = decoder_P.std_coef_img_[curclas]
                    P_coef_std_str = f"{roititle}_{curclas}_P.nii.gz"
                    I_coef_std = decoder_I.std_coef_img_[curclas]
                    I_coef_std_str = f"{roititle}_{curclas}_I.nii.gz"

                    P_coef.to_filename(os.path.join(decodingpath,P_coef_str))
                    I_coef.to_filename(os.path.join(decodingpath,I_coef_str))

                    P_coef_std.to_filename(os.path.join(decodingpath,P_coef_std_str))
                    I_coef_std.to_filename(os.path.join(decodingpath,I_coef_std_str))

                    if visu:

                        view = view_img(P_coef,title=P_coef_str)
                        view.open_in_browser()

                        view = view_img(I_coef,title=I_coef_str)
                        view.open_in_browser()
