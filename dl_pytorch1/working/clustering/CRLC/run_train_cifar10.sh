#!/bin/bash

USER="your_name"
PYTHON_EXE="your_path_to_python_exe"
MY_PROJECT="your_path_to_InfoMaxClustering"

export PYTHONPATH="$MY_PROJECT:$PYTHONPATH"
$PYTHON_EXE ./run_train_cifar10.py