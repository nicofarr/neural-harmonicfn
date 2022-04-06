#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np 
from nilearn.image import math_img
from nilearn.masking import intersect_masks
from matplotlib import pyplot as plt 
from nilearn.plotting import plot_roi

roi_info=pd.read_csv('MIST_ROI.csv',sep=';')

# Fetch just one region


labels_img = 'MIST_ROI.nii.gz'

selectrois = [51,67,89,104,105,27,28]

f,axes = plt.subplots(nrows=15,figsize=(10,10))

roi_imgs = []
for i,roinum in enumerate(selectrois):
    roi_name = roi_info[roi_info['roi'] == roinum]['label'][roinum-1]
    print(roi_name)
    curroi = math_img(formula='img=={}'.format(roinum),img=labels_img)
    plot_roi(curroi,figure=f,axes=axes[i],title=(roi_name))
    curroi.to_filename(f"visrois/{roi_name}.nii.gz")
    roi_imgs.append(curroi)
plt.show()