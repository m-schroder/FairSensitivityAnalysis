# FairSensitivityAnalysis

We provide this repository for reproducibility purposes of our paper "Causal Fairness under Unobserved Confounding: A Neural Sensitivity Framework" (https://arxiv.org/pdf/2311.18460.pdf).

The folder structure is as follows:
- data: Config files for generating the synthetic datasets as well as a dataframe containing the pre-processed real-world dataset. The original files are too large to be uploaded in this reporsitory. Please refer to https://www.icpsr.umich.edu/web/ICPSR/studies/37692/datadocumentation (data version DS0002) for downloading the original files.
- modules: Python scripts including the prediction models, helper functions for calculating the bounds, plotting and evaluation
- notebooks: Jupyter notebooks for data simulation, pre-processing and model training