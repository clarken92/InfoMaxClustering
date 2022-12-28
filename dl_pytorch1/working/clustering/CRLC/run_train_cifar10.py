import os
from os.path import join, abspath
import subprocess

from my_utils.python_utils.general import get_arg_string
from global_settings import PYTHON_EXE, RESULTS_DIR, PYTORCH_DATA_DIR


dataset = "cifar10"
dataset_dir = join(PYTORCH_DATA_DIR, "ComputerVision", dataset)
output_dir = abspath(join(RESULTS_DIR, "clustering", dataset, "pytorch"))


# Default settings
# =============================== #
DEFAULT_CONFIG = {
    # dataset
    # --------------------- #
    "dataset": dataset,
    "dataset_dir": dataset_dir,
    "output_dir": output_dir,
    "workers": 16,
    "combine_train_test": False,
    # --------------------- #

    # architecture
    # --------------------- #
    "model": "CRLC",
    "net": "resnet34",
    "arch": "wvangansbeke",

    "proj_head_type": "mlp2",
    "class_head_type": "mlp2",
    # --------------------- #

    # training
    # --------------------- #
    "epochs": 2000,
    "batch_size": 512,
    # --------------------- #

    # optimizer
    # --------------------- #
    "optim": "sgd",
    "lr": 0.1,
    "momentum": 0.9,
    "nesterov": False,
    "weight_decay": 5e-4,
    # --------------------- #

    # settings
    # --------------------- #
    # "augmentation_mode": "std_std",
    "num_clusters": 10,
    "num_class_subheads": 10,
    "feat_dim": 128,

    "critic_type": "log_dot_prod",
    "cons_type": "neg_log_dot_prod",
    
    "temperature": 0.1,
    "cluster_temperature": 1.0,

    "smooth_prob": True,
    "smooth_coeff": 0.01,
    "min_rate": 0.7,
    "max_rate": 1.3,
    "max_abs_logit": 25.0,

    "use_pretrained_net": False,
    "pretrained_file": "",
    "freeze_encoder": False,
    # --------------------- #

    # coefficients
    # --------------------- #
    "contrast_cluster": 0.0,
    "contrast_y": 1.0,
    "cons_y": 0.0,
    "contrast_z": 10.0,
    "entropy": 2.0,
    "lower_clamp": 0.0,
    "upper_clamp": 0.0,
    "lower_logit_clamp": 0.01,
    "upper_logit_clamp": 0.01,
    # --------------------- #

    # freq
    # --------------------- #
    "log_freq": -1,
    "save_freq_epoch": 1,
    "eval_freq_epoch": 1,
    # --------------------- #
}
# =============================== #


# Run settings
# =============================== #
run_config = {
    "run": "0_default",

    "force_rm_dir": True,
}
# =============================== #

config = DEFAULT_CONFIG
config.update(run_config)
arg_str = get_arg_string(config)
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

print("Running arguments: [{}]".format(arg_str))
run_command = "{} ./train.py {}".format(PYTHON_EXE, arg_str).strip()
subprocess.call(run_command, shell=True)