# Data Installation

## 1. Download datasets

Download the required datasets (training set and smoke test set) and ground truth file:

https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/data/

Then, remove the downloaded files in your current repository DaTParkinson/. 


## 2. Unzip and rename

Create the destination directories

```
mkdir -p smoke_test_data/ data/niftis/
```

```
unzip niftis_utCGpHE.zip -d data/niftis/
tar -xf smoke_test_data_2gePzfM.tar.gz -C smoke_test_data/
mv train_labels_JNDlMjr.csv data/train_labels.csv
```

Finally, you can remove zip and tar files 
```
rm niftis_utCGpHE.zip smoke_test_data_2gePzfM.tar.gz 
```



## 3. Organize directory structure

We recommend the following project structure:

```text
DaTParkinson/
├── data/
│   ├── niftis/
│   │   ├── xaji0y6d.nii.gz
│   │   ├── pbhsahxt.nii.gz
│   │   └── ...
│   └── train_labels.csv
└── smoke_test_data/
    └── niftis/
        ├── ...
        └── *.nii.gz
```

or

```
data/
├── niftis/
├   ├── xaji0y6d.nii.gz
├   ├── pbhsahxt.nii.gz
├   └── ...
└── train_labels.csv
```




## Test set

The test set is withheld and is not available for download. Its images are only accessible from within the runtime container, where they are mounted alongside the training data. Because this is a code execution challenge, you will not see the test examinations directly — your submitted code reads them at inference time and generates predictions for each one.

The test examinations are the same NIfTI format as the training data and follow the same conventions (one 3D volume per file, named <uid>.nii.gz, with varying dimensions and voxel spacing).

For additional debugging, a "smoke test" environment that replicates the test inference runtime has been provided. It runs only on a small set of examinations. In the smoke test environment, the test data structure is the same (`niftis/` and `submission_format.csv`), but the available data represents a small sample drawn from the training set rather than the full test set.
