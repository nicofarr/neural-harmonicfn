# neural-harmonicfn
Repository for paper ["Neural representations of harmonic function in musical imagery and perception"](https://osf.io/qu5vy/)

# Script organisation 

- [1_prepareroi.py](1_prepareroi.py) generates nifti files corresponds to masks for each ROI. Will be used for decoding scripts and RSA. 
- [2_GLM_univariate.py](2_GLM_univariate.py) performs (for one subject) the univariate analysis (GLM) described in section 2.8.1
- [3_GLM_singletrial.py](3_GLM_singletrial.py) performs (for one subject) the GLM for multivariate analysis described in section 2.8.2
- [4_GLM_secondlevel.py](4_GLM_secondlevel.py) performs the second level analysis described in section 2.8.1
- [5_GLM_secondlevel_covar.py](5_GLM_secondlevel_covar.py) performs the second level analysis with covariates, described in section 2.8.1
- [6_decoding_allsubjects.py](6_decoding_allsubjects.py) performs the ROI based decoding across all subjects, described in section 2.8.3
- [7_decoding_wholebrain.py](7_decoding_wholebrain.py) performs (for one subject) the whole brain decoding analysis, described in Supplementary Material.
- [8_RSA_crossnobis.py](8_RSA_crossnobis.py) performs (for one subject) the estimation of RSA models, described in section 2.8.4
- [9_RSA_inference.py](9_RSA_inference.py) performs inference at the group level on the RSA models, described in section 2.8.4.