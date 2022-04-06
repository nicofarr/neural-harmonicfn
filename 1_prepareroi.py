#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np 
from nilearn.image import math_img
from nilearn.masking import intersect_masks
from matplotlib import pyplot as plt 
from nilearn.plotting import plot_roi
import os

roi_info=pd.read_csv('MIST_ROI.csv',sep=';')

# Fetch just one region

os.makedirs('rois',exist_ok=True)

labels_img = 'MIST_ROI.nii.gz'

selectrois = [170,171,153,154,209,210,49,50]

roi_imgs = []
for i,roinum in enumerate(selectrois):
    roi_name = roi_info[roi_info['roi'] == roinum]['label'][roinum-1]
    print(roi_name)
    curroi = math_img(formula='img=={}'.format(roinum),img=labels_img)
    curroi.to_filename(f"rois/{roi_name}.nii.gz")
    roi_imgs.append(curroi)