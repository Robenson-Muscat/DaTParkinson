# DaTParkinson


## Overview

Parkinsonian syndromes affect millions of people worldwide, yet diagnosing them accurately and early remains challenging. Dopamine transporter (DaT) imaging is an important tool for distinguishing neurodegenerative parkinsonian syndromes from other conditions, but reliable interpretation requires specialized expertise that is not available everywhere.

Each year, more than 20,000 DaT scans are performed in France alone. While many can be classified as normal or abnormal, approximately one in five cases remains difficult to interpret — particularly in early-stage or atypical presentations — delaying diagnosis, complicating treatment decisions, and increasing demands on specialist readers.


## Goal (or Task)
In this challenge, we invite data scientists, machine learning engineers, medical imaging researchers, and nuclear medicine specialists to advance AI tools for DaT scan interpretation. Using a unique multicenter dataset of scans collected and annotated by French experts across ten hospital centers in France, your goal is to develop computer vision models that classify DaT scans as normal or abnormal.


The imaging data are provided as three-dimensional DaT scan reconstructions in compressed Neuroimaging Informatics Technology Initiative (NIfTI) format (.nii.gz). Each examination is a single 3D volume, and each patient is represented by one file. The filename (minus the .nii.gz extension) is the uid that links each image to its label.

In addition, participants are welcome to incorporate external datasets for training; however, use of external data is subject to important exceptions and caveats.

## Data

### Images

Each .nii.gz file contains a single 3D reconstructed volume stored as 16-bit unsigned integer voxel intensities. Note that acquisition and reconstruction parameters vary across the dataset, so images are not all the same size or resolution:

Volume dimensions differ from scan to scan (for example, 142 × 142 × 112 or 128 × 128 × 128). Your pipeline should not assume a fixed input shape.
Voxel spacing also varies (e.g., 2.46 mm and 3.895 mm isotropic), reflecting differences in scanners and acquisition protocols across contributing centers. The spacing is recorded in each file's NIfTI header and may be useful for resampling images to a common resolution.

These differences are a normal consequence of pooling data from multiple institutions, and building models that generalize across them is part of the challenge.

### Labels

train_labels.csv contains one row per DaT scan examination in the training set, with the following columns:

uid (str) — unique identifier for each DaT scan examination; matches the image filename without the .nii.gz extension
is_pathologic (float) — classification of the DaT scan examination, where 0.0 = Normal and 1.0 = abnormal


Label example — the first five rows of train_labels.csv:
uid	is_pathologic
xaji0y6d	0.0
pbhsahxt	0.0
hv3a3zmf	0.0
8mdd4v30	0.0
t9nt3w5u	1.0


### Test set

The test set is withheld and is not available for download. Its images are only accessible from within the runtime container, where they are mounted alongside the training data. Because this is a code execution challenge, you will not see the test examinations directly — your submitted code reads them at inference time and generates predictions for each one.

The test examinations are the same NIfTI format as the training data and follow the same conventions (one 3D volume per file, named <uid>.nii.gz, with varying dimensions and voxel spacing).



## Performance metric

Leaderboard performance is evaluated according to log loss. Log loss (a.k.a. logistic loss or cross-entropy loss) penalizes confident but incorrect predictions. It also rewards confidence scores that are well-calibrated probabilities, meaning that they accurately reflect the long-run probability of being correct. This is an error metric, so a lower value is better.

Log loss for a single observation is calculated as follows:

𝐿log⁡(𝑦,𝑝)=−(𝑦⁢log⁡(𝑝)+(1−𝑦)⁢log⁡(1−𝑝))

where 𝑦 is a binary variable indicating whether the examination is abnormal (1) or normal (0), and 𝑝 is the user-predicted probability that the examination is abnormal. The loss for the entire dataset is the average loss across all observations.

