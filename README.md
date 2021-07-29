# Representation Learning by Dense Predictive Coding with Gabor stimuli

This repository contains a modified implementation of the Dense Predictive Coding (DPC) algorithm ([Han _et al._, 2019, _ICCV_](https://arxiv.org/abs/1909.04656), hosted [here](https://github.com/TengdaHan/DPC).  

The stimulus design is based on the Gabor stimuli used in [Gillon _et al._, 2021, _bioRxiv_](https://www.biorxiv.org/content/10.1101/2021.01.15.426915v1).


## 2. Installation

This code requires: 
- [Anaconda](https://www.anaconda.com/) or [Miniconda](https://conda.io/miniconda.html)
- [pip](https://pip.pypa.io/en/stable/)
- [cuda](https://developer.nvidia.com/cuda-toolkit-archive) (e.g., 10.2)
- [cudnn](https://developer.nvidia.com/rdp/cudnn-archive) (e.g., 7.6.5)
- [DPC GitHub repo](https://github.com/TengdaHan/DPC), copied into the same directory (**without** overwriting any files!)

Once these are installed, the required conda environment can be created from `ssl.yml` by running:  
`conda env create -f ssl.yml`  

That environment can then be activated by running:
`source activate ssl`  

The code is written in `Python 3`. 


## 3. Running code

The scripts are designed to run on a cuda-enabled machine. Bash scripts are provided for running analyses with Slurm on a GPU node of a specific cluster (Mila cluster). They can be adapted to a different cluster, or used as templates to run local jobs, for example.  

In all cases, before running, the environment variable `SAVE_DIR` must be updated in each bash script to point to a directory that exists on the machine.


- ### Training DPC model with Gabor stimuli

    To train a ResNET18 model from scratch on the Gabor stimuli, run `bash_dpc`.


- ### Training a pre-trained DPC model with Gabor stimuli

    To train a ResNET18 model with 29 different seeds on the Gabor stimuli that has been pre-trained on the Kinetics400 dataset:
    1. Download the [3D-ResNet18-Kinetics400-128x128](https://drive.google.com/file/d/1jbMg2EAX8armIQA6_0YwfATh_h7rQz4u/view?usp=sharing) pre-trained DPC weights provided on the [DPC GitHub repo](https://github.com/TengdaHan/DPC).
    2. Run `bash_traineddpc`, ensuring that the script points to the location where the pre-trained weights are saved,  
    e.g. by running `sbatch bash_traineddpc` if using Slurm.

    To test the script with just one seed, if using Slurm, run  
    `sbatch --export=ALL,TEST=1 --time=0:45:00 --array=0 bash_traineddpc`.


- ### Plotting results of loss analyses after training the pre-trained DPC model with Gabor stimuli

    To plot the loss analysis results obtained by running `bash_traineddpc`, run `bash_plot_losses`.  


## 4. Seeding

Seeding is partially implemented in the codebase, but does not ensure fully deterministic behavior.

## 5. Citation

**Original DPC code:** Han T, Xie W, Zisserman A. (2019) Video Representation Learning by Dense Predictive Coding. 
_Workshop on Large Scale Holistic Video Understanding (ICCV)_.

**Gabor stimuli:** Gillon C _et al._ (2021) Learning from unexpected events in the neocortical microcircuit. _biorxiv_.

