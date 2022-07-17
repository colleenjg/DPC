# Representation Learning by Dense Predictive Coding with Gabor stimuli

This repository contains a modified implementation of the Dense Predictive Coding (DPC) algorithm ([Han _et al._, 2019, _ICCV_](https://arxiv.org/abs/1909.04656), hosted [here](https://github.com/TengdaHan/DPC)).  

The Gabor stimulus design is based on the stimuli used in [Gillon _et al._, 2021, _bioRxiv_](https://www.biorxiv.org/content/10.1101/2021.01.15.426915).

## 1. Installation

This code requires: 
- [Anaconda](https://www.anaconda.com/) or [Miniconda](https://conda.io/miniconda.html)
- [pip](https://pip.pypa.io/en/stable/)
- [cuda](https://developer.nvidia.com/cuda-toolkit-archive) (e.g., 10.2)
- [cudnn](https://developer.nvidia.com/rdp/cudnn-archive) (e.g., 7.6.5)

Once these are installed, the required conda environment can be created from `ssl.yml` by running:  
`conda env create -f ssl.yml`  

That environment can then be activated by running:
`source activate ssl`  

The code is written in `Python 3`. 


## 2. Running code

The scripts are developped in `pytorch`, and run optimally on a cuda-enabled machine.  
Bash scripts are provided under `slurm` for running analyses with Slurm on GPU nodes of a specific cluster (Mila cluster). They can be adapted to a different cluster, or used as templates for running local jobs, for example.  


- ### Training DPC model

    To train a ResNet18 model from scratch on the Kinetics400 dataset, run, e.g.:  
    `python run_model.py --model dpc-rnn --dataset k400 --net resnet18 --img_dim 128 --num_epochs 100 --output_dir /path/to/save/directory`

- ### Finetuning a pre-trained DPC model on a classification task
W
    To finetune a ResNet18 model on classifying images from the UCF101 dataset, with weights pre-trained on the Kinetics400 dataset:  
    1. Download the [3D-ResNet18-Kinetics400-128x128](https://drive.google.com/file/d/1jbMg2EAX8armIQA6_0YwfATh_h7rQz4u/view?usp=sharing) pre-trained DPC weights provided on the [DPC GitHub repo](https://github.com/TengdaHan/DPC).
    2. Run `run_model.py`, ensuring that the `pretrained` argument points to the pre-trained weights path, e.g.:  
    `python run_model.py --model lc-rnn --dataset ucf101 --net resnet18 --img_dim 128 --num_epochs 100 --train_what ft --pretrained /path/to/pretrained/model.pth.tar --output_dir /path/to/save/directory`


## 3. Code determinism

The codebase allows for seeding of random processes. However, fully deterministic behaviour consistent across runs is not guaranteed when using a cuda-enabled machine, as the models implement some `pytorch` algorithms that cannot be made fully deterministic (i.e., an error is raised if `torch.use_deterministic_algorithms(True)` is set, as verified with `torch==1.10`).


## 4. Citation

**Original DPC code:** Han T, Xie W, Zisserman A (2019) Video Representation Learning by Dense Predictive Coding. 
_Workshop on Large Scale Holistic Video Understanding (ICCV)_.

**Gabor stimuli:** Gillon CJ _et al._ (2021) Learning from unexpected events in the neocortical microcircuit. _bioRxiv_.


## 5. Additional credit

[Katharina Wilmes](https://github.com/k47h4) and [Luke Y. Prince](https://github.com/lyprince) developped the original code on which the **Gabor stimulus code** is based in this repository, based on [Gillon _et al._, 2021, _bioRxiv_](https://www.biorxiv.org/content/10.1101/2021.01.15.426915).


## 6. Licenses

- Content that is original or derived from the [DPC repository](https://github.com/TengdaHan/DPC) is covered under **LICENSE_DPC**:  
    - `run_model.py`
    - `asset`
    - `dataset`, except `gabor_stimuli.py` and `gabor_stimuli.ipynb`
    - `model`
    - `process_data`
    - `utils`, except `gabor_utils.py`

- The remaining content is covered under **LICENSE**.
