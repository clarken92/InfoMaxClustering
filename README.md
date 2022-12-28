# Clustering by Maximizing Mutual Information Across Views


This repository contains the official implementation of our paper:
> [**Clustering by Maximizing Mutual Information Across Views**](https://arxiv.org/abs/2107.11635)
>
> [Kien Do](https://twitter.com/kien_do_92), [Truyen Tran](https://twitter.com/truyenoz), Svetha Venkatesh

__Accepted at ICCV 2021.__


## Contents
1. [Requirements](#requirements)
1. [Features](#features)
0. [Repository structure](#repository-structure)
0. [Setup](#setup)
0. [Training](#training)
0. [Citation](#citation)

## Requirements
Pytorch >= 1.8

## Features
- Support model saving
- Support logging
- Support tensorboard visualization

## Repository structure
Our code is organized in 5 main parts:
- `dl_pytorch1/models`: Containing the model CRLC used in our paper.
- `dl_pytorch1/utils`: Containing implementation for the contrastive loss.
- `my_utils`: Containing other utility functions.
- `dl_pytorch1/external_codes`: Containing network architectures I borrowed from other Github repositories.
- `working`: Containing scripts for training our model.

**IMPORTANT NOTE**: Since this repository is organized as a Python project, I strongly encourage you to import it as a project to an IDE (e.g., PyCharm). By doing so, the path to the root folder of this project will be automatically added to PYTHONPATH when you run the code via your IDE. Otherwise, you have to explicitly add it when you run in terminal. Please check `run_cifar10.sh` (or `run_cifar100.sh`, `run_svhn.sh`) to see how it works.

## Setup
The setup for training is **very simple**. All you need to do is opening the `global_settings.py` file and changing the values of the global variables to match yours. The meanings of the global variables are given below:
* `PYTHON_EXE`: Path to your python interpreter.
* `PROJECT_NAME`: Name of the project, which I set to be `'InfoMaxClustering'` - the name of this project.
* `PROJECT_DIR`: Path to the root folder containing the code of this project.
* `RESULTS_DIR`: Path to the root folder that will be used to store results for this project.  
* `RAW_DATA_DIR`: Path to the root folder that contains raw datasets.
* `PYTORCH_DATA_DIR`: Path to the root folder that contains datasets supported by Pytorch.

## Training
Once you have setup everything in `global_settings.py`, you can start training by running the following commands in your terminal:
```shell
export PYTHONPATH="[path to this project]:$PYTHONPATH"
python train.py [required arguments]
```
**IMPORTANT NOTE**: If you run using the commands above, please remember to provide all **required** arguments specified in `train.py` otherwise errors will be raised.

However, if you are too lazy to type arguments in the terminal (like me :sweat_smile:), you can set these arguments in the `run_config` dictionary in `run_train_cifar10.py` and simply run this file:
```shell
export PYTHONPATH="[path to this project]:$PYTHONPATH"
python run_cifar10.py
```

I also provide a `run_train_cifar10.sh` file as an example for you.

## Citation
If you find this repository useful for your research, please consider citing our paper:

```bibtex
@inproceedings{do2021clustering,
  title={Clustering by maximizing mutual information across views},
  author={Do, Kien and Tran, Truyen and Venkatesh, Svetha},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9928--9938},
  year={2021}
}
```
