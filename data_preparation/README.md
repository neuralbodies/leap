# Data preprocessing 
Code to prepare training data from AMASS dataset for the SMPL+H body model.

Download [**AMASS**](https://amass.is.tue.mpg.de/) dataset and save it in a desired directory `${AMASS_ROOT}` and then execute the following command to prepare training files: 

```shell script
python amass.py --src_dataset_path ${AMASS_ROOT} --dst_dataset_path ${TRAINING_DATA_ROOT} --subsets BMLmovi --bm_dir_path ${BODY_MODELS}/smplh
```
Then, copy reduced the training/test split files for the MoVi dataset into the training directory. 
```shell script
cp ./split_movi_training.txt ${TRAINING_DATA_ROOT}/split_movi_training.txt
cp ./split_movi_validation.txt ${TRAINING_DATA_ROOT}/split_movi_validation.txt
```
