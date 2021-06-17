# LEAP: Learning Articulated Occupancy of People
[**Paper**](https://arxiv.org/pdf/2104.06849.pdf) | [**Video**](https://www.youtube.com/watch?v=UVB8A_T5e3c) | [**Project Page**](https://neuralbodies.github.io/LEAP)

<div style="text-align: center">
    <a href="https://neuralbodies.github.io/LEAP"><img src="https://neuralbodies.github.io/LEAP/images/teaser%20figure%20ppl.png" alt="teaser figure"/></a>
</div>

This is the official implementation of the CVPR 2021 submission [**LEAP: Learning Articulated Occupancy of People**](https://neuralbodies.github.io/LEAP)

LEAP is a neural network architecture for representing volumetric animatable human bodies. It follows traditional human body modeling techniques and leverages a statistical human prior to generalize to unseen humans.

If you find our code or paper useful, please consider citing:
```bibtex
@InProceedings{LEAP:CVPR:21,
  title = {{LEAP}: Learning Articulated Occupancy of People},
  author = {Mihajlovic, Marko and Zhang, Yan and Black, Michael J and Tang, Siyu},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2021},
}
```
Contact [Marko Mihajlovic](mailto:markomih@ethz.ch) for questions or open an issue / a pull request.

# Prerequests 
## 1) SMPL body model
Download a SMPL body model ([**SMPL**](https://smpl.is.tue.mpg.de/), [**SMPL+H**](https://mano.is.tue.mpg.de/), [**SMPL+X**](https://smpl-x.is.tue.mpg.de/), [**MANO**](https://mano.is.tue.mpg.de/)) and store it under `${BODY_MODELS}` directory of the following structure:  
```bash
${BODY_MODELS}
├── smpl
│   └── x
├── smplh
│   ├── male
|   │   └── model.npz
│   ├── female
|   │   └── model.npz
│   └── neutral
|       └── model.npz
├── mano
|   └── x
└── smplx
    └── x
```

NOTE: currently only SMPL+H model is supported. Other models will be available soon.  
 
## 2) Installation
Another prerequest is to install python packages specified in the `requirements.txt` file, which can be conveniently 
accomplished by using an [Anaconda](https://www.anaconda.com/) environment:
```bash
# clone the repo
git clone https://github.com/neuralbodies/leap.git
cd ./leap

# create environment
conda env create -f environment.yml
conda activate leap
```
and install the `leap` package via `pip`:
```bash
# note: install the build-essentials package if not already installed (`sudo apt install build-essential`) 
python setup.py build_ext --inplace
pip install -e .
```

## 3) (Optional) Download LEAP pretrained models
Download LEAP pretrained models from [**here**](https://drive.google.com/drive/folders/1HkkH013ErpekedqAEEifQxyMOoVu3ugg?usp=sharing) and extract them under `${LEAP_MODELS}` directory.

## Usage
Check demo code in `examples/query_leap.py` for a demonstration on how to use LEAP for differentiable occupancy checks.    

## Train your own model
Follow instructions specified in `data_preparation/README.md` on how to prepare training data.
Then, replace placeholders for pre-defined path variables in configuration files (`configurations/*.yml`) and execute `training_code/train_leap.py` script to train the neural network modules. 

LEAP consists of two LBS networks and one occupancy decoder. 
```shell script
cd training_code
```
To train the forward LBS network, execute the following command: 
```shell script
python train_leap.py ../configurations/fwd_lbs_training.yml
```

To train the inverse LBS network: 
```shell script
python train_leap.py ../configurations/inv_lbs_training.yml
```
Once the LBS networks are trained, execute the following command to train the occupancy network:
```shell script
python train_leap.py ../configurations/occupancy_training.yml
```

See specified yml configuration files for details about network hyperparameters. 
