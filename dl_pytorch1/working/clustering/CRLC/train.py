from __future__ import print_function

import os
import argparse
from time import time
import math

import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torch.backends.cudnn as cudnn

from my_utils.python_utils.general import make_dir_if_not_exist
from my_utils.python_utils.arg_parsing import save_args, str2bool, str2dict
from my_utils.python_utils.training import BestResultsTracker, SaveDirTracker
from my_utils.python_utils.metrics.clustering import acc as ACC, nmi as NMI, ari as ARI

from my_utils.pytorch1_utils.initialization import kaiming_uniform_
from my_utils.pytorch1_utils.training import ScalarSummarizer, add_scalar_summaries
from my_utils.pytorch1_utils.data.sampler import ContinuousBatchSampler
from my_utils.pytorch1_utils.datasets import DatasetWithMultipleTransforms, \
    DatasetCombined2
from my_utils.pytorch1_utils.modules import CustomModuleList
from my_utils.pytorch1_utils.image.transforms.single.numpy import GaussianBlur
from my_utils.pytorch1_utils.networks.cifar10.resnet.meliketoy import \
    ResNet18 as ResNet18_1, ResNet34 as ResNet34_1, ResNet50 as ResNet50_1
from my_utils.pytorch1_utils.networks.cifar10.resnet.zhirongw import \
    ResNet18 as ResNet18_2, ResNet34 as ResNet34_2, ResNet50 as ResNet50_2

from dl_pytorch1.external_codes.clustering.SCAN_wvangansbeke.data.cifar import CIFAR10, CIFAR20
from dl_pytorch1.external_codes.clustering.SCAN_wvangansbeke.data.stl import STL10
from dl_pytorch1.external_codes.clustering.SCAN_wvangansbeke.models.resnet_cifar import \
    resnet18, resnet34, resnet50

from dl_pytorch1.models.clustering.crlc import CRLC


parser = argparse.ArgumentParser(allow_abbrev=False, description='Settings for all datasets')

# Dataset
# ---------------------------------- #
parser.add_argument('--dataset', type=str, required=True,
                    choices=['cifar10', 'cifar20', 'stl10'])
parser.add_argument('--subset_file', default="", type=str)
parser.add_argument('--dataset_dir', required=True, type=str)
parser.add_argument('--workers', default=16, type=int)
parser.add_argument('--combine_train_test', default='False', type=str2bool)
# ---------------------------------- #

# Model
# ---------------------------------- #
parser.add_argument('--model', default='CRLC', type=str, choices=['CRLC'])
parser.add_argument('--net', default="resnet34", type=str, choices=["resnet18", "resnet34"])
parser.add_argument('--arch', default='wvangansbeke', type=str,
                    choices=['default', 'zhirongw', 'wvangansbeke', 'torchvision'])

parser.add_argument('--class_head_type', default='mlp2', type=str, choices=['linear', 'mlp2'])
parser.add_argument('--proj_head_type', default='mlp2', type=str, choices=['linear', 'mlp2'])
# ---------------------------------- #

# Training
# ---------------------------------- #
parser.add_argument('--epochs', default=2000, type=int)
parser.add_argument('--batch_size', default=512, type=int)
# ---------------------------------- #

# Optimizer
# ---------------------------------- #
parser.add_argument('--optim', default='sgd', type=str, choices=['sgd', 'adam', 'adamw'])
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
# SGD
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--nesterov', default='False', type=str2bool)
# Adam
parser.add_argument('--beta1', default=0.9, type=float)
parser.add_argument('--beta2', default=0.99, type=float)

parser.add_argument('--lr_schedule_type', default='constant', type=str,
                    choices=['cosine', 'step', 'constant'])
parser.add_argument('--lr_decay_rate', default=0.1, type=float)
parser.add_argument('--lr_decay_epochs', default=[], type=int, nargs="+")
# ---------------------------------- #

# Settings
# ---------------------------------- #
parser.add_argument('--num_class_subheads', default=10, type=int)
parser.add_argument('--num_clusters', default=10, type=int)
parser.add_argument('--feat_dim', default=128, type=int)

parser.add_argument('--critic_type', default='log_dot_prod', type=str,
                    choices=['log_dot_prod', 'dot_prod', 'neg_jsd', 'neg_l2'])
parser.add_argument('--cons_type', default='neg_log_dot_prod', type=str,
                    choices=['neg_log_dot_prod', 'neg_dot_prod',
                             'xent', 'jsd', 'l1', 'l2'])

parser.add_argument('--temperature', default=0.1, type=float)
parser.add_argument('--cluster_temperature', default=1.0, type=float)

parser.add_argument('--smooth_prob', default='True', type=str2bool)
parser.add_argument('--smooth_coeff', default=0.1, type=float)
parser.add_argument('--min_rate', default=0.7, type=float)
parser.add_argument('--max_rate', default=1.3, type=float)
parser.add_argument('--max_abs_logit', default=25.0, type=float)

parser.add_argument('--use_pretrained_net', default='False', type=str2bool)
parser.add_argument('--pretrained_file', default='', type=str)
parser.add_argument('--freeze_encoder', default='True', type=str2bool)
# ---------------------------------- #

# Coefficients
# ---------------------------------- #
parser.add_argument('--contrast_z', default=10.0, type=float)
parser.add_argument('--contrast_y', default=1.0, type=float)
parser.add_argument('--cons_y', default=0.0, type=float)
parser.add_argument('--contrast_cluster', default=0.0, type=float)
parser.add_argument('--entropy', default=2.0, type=float)

parser.add_argument('--lower_clamp', default=0.0, type=float)
parser.add_argument('--upper_clamp', default=0.0, type=float)
parser.add_argument('--lower_logit_clamp', default=0.01, type=float)
parser.add_argument('--upper_logit_clamp', default=0.01, type=float)
# ---------------------------------- #

# Log
# ---------------------------------- #
parser.add_argument('--log_freq', type=int, default=-1)
parser.add_argument('--save_freq_epoch', type=int, default=5)
parser.add_argument('--eval_freq_epoch', type=int, default=1)
# ---------------------------------- #

# Save/Resume
# ---------------------------------- #
parser.add_argument('--max_save', type=int, default=2)
parser.add_argument('--max_save_best', type=int, default=2)

parser.add_argument('--resume', type=str2bool, default='False')
parser.add_argument('--resume_best', type=str2bool, default='False')
parser.add_argument('--resume_step', type=int, default=-1)
# New config that will overwrite old config when resuming
parser.add_argument('--new_config', type=str2dict, default='')
# ---------------------------------- #

# Run
# ---------------------------------- #
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--run', type=str, required=True)
parser.add_argument('--force_rm_dir', default='False', type=str2bool)
# ---------------------------------- #


# OK
def get_dataset_manager_CIFAR(args):
    print(f"Load {args.dataset} dataset!")
    args.x_shape = (3, 32, 32)
    args.x_dim = 3 * 32 * 32

    if args.dataset == "cifar10":
        DS_CLASS = CIFAR10
        MEAN = (0.4914, 0.4822, 0.4465)
        STD = (0.2023, 0.1994, 0.2010)
        args.num_classes = 10

    elif args.dataset == "cifar20":
        DS_CLASS = CIFAR20
        MEAN = (0.5071, 0.4867, 0.4408)
        STD = (0.2675, 0.2565, 0.2761)
        args.num_classes = 20

    else:
        raise ValueError("Do not support 'dataset'={}!".format(args.dataset))

    tr = transforms

    train_tf = tr.Compose([
        tr.RandomResizedCrop(size=(32, 32), scale=(0.2, 1.0)),
        tr.RandomHorizontalFlip(),
        tr.RandomApply([tr.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        tr.RandomGrayscale(p=0.2),
        tr.ToTensor(),
        tr.Normalize(MEAN, STD)
    ])

    test_tf = tr.Compose([
        tr.CenterCrop((32, 32)),
        tr.ToTensor(),
        tr.Normalize(MEAN, STD)
    ])

    # Train dataset
    # -------------------------------- #
    if args.combine_train_test:
        print("Combine train and test datasets!")

        _train_dataset = DS_CLASS(
            args.dataset_dir, download=False,
            train=True, transform=None)

        _test_dataset = DS_CLASS(
            args.dataset_dir, download=False,
            train=False, transform=None)

        train_dataset = DatasetCombined2(_train_dataset, _test_dataset)

    else:
        print("Do not combine train and test datasets!")

        train_dataset = DS_CLASS(
            args.dataset_dir, download=False,
            train=True, transform=None)

    train_dataset = DatasetWithMultipleTransforms(
        train_dataset, transforms=[train_tf, train_tf])
    # -------------------------------- #

    # Test dataset
    # -------------------------------- #
    test_dataset = DS_CLASS(
        args.dataset_dir, download=False,
        train=False, transform=test_tf)
    # -------------------------------- #

    # Train dataset (no aug)
    # -------------------------------- #
    train_dataset_noaug = DS_CLASS(
        args.dataset_dir, download=False,
        train=True, transform=test_tf)
    # -------------------------------- #

    # Train loader
    # -------------------------------- #
    train_sampler = ContinuousBatchSampler(len(train_dataset), num_repeats=5,
                                           batch_size=args.batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              num_workers=args.workers, pin_memory=True)
    train_loader = iter(train_loader)
    # -------------------------------- #

    # Test loader
    # -------------------------------- #
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=args.workers, pin_memory=True,
                             shuffle=False, drop_last=False)
    # -------------------------------- #

    # Train loader (no aug)
    # -------------------------------- #
    train_loader_noaug = DataLoader(train_dataset_noaug, batch_size=args.batch_size,
                                    num_workers=args.workers, pin_memory=True,
                                    shuffle=False, drop_last=False)
    # -------------------------------- #

    return {
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'train_dataset_noaug': train_dataset_noaug,
        'train_loader_noaug': train_loader_noaug,
    }


def get_dataset_manager_STL10(args):
    print(f"Load {args.dataset} dataset!")
    args.x_shape = (3, 96, 96)
    args.x_dim = 3 * 96 * 96
    args.num_classes = 10

    tr = transforms

    train_tf = tr.Compose([
        tr.RandomResizedCrop(size=(96, 96), scale=(0.2, 1.0)),
        tr.RandomHorizontalFlip(),
        tr.RandomApply([tr.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        tr.RandomGrayscale(p=0.2),
        GaussianBlur(9, 0.1, 2.0),
        tr.ToTensor(),
    ])

    test_tf = tr.Compose([
        tr.CenterCrop((96, 96)),
        tr.ToTensor(),
    ])

    # Train dataset
    # -------------------------------- #
    if args.combine_train_test:
        print("Combine train and test datasets!")

        _train_dataset = STL10(
            args.dataset_dir, download=False,
            split="train", transform=None)

        _test_dataset = STL10(
            args.dataset_dir, download=False,
            split="test", transform=None)

        train_dataset = DatasetCombined2(_train_dataset, _test_dataset)

    else:
        print("Do not combine train and test datasets!")

        train_dataset = STL10(
            args.dataset_dir, download=False,
            split="train", transform=None)

    train_dataset = DatasetWithMultipleTransforms(
        train_dataset, transforms=[train_tf, train_tf])
    # -------------------------------- #

    # Test dataset
    # -------------------------------- #
    test_dataset = STL10(
        args.dataset_dir, download=False,
        split="test", transform=test_tf)
    # -------------------------------- #

    # Train dataset (no aug)
    # -------------------------------- #
    train_dataset_noaug = STL10(
        args.dataset_dir, download=False,
        split="train", transform=test_tf)
    # -------------------------------- #

    # Train loader
    # -------------------------------- #
    train_sampler = ContinuousBatchSampler(len(train_dataset), num_repeats=5,
                                           batch_size=args.batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              num_workers=args.workers, pin_memory=True)
    train_loader = iter(train_loader)
    # -------------------------------- #

    # Test loader
    # -------------------------------- #
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=args.workers, pin_memory=True,
                             shuffle=False, drop_last=False)
    # -------------------------------- #

    # Train loader (no aug)
    # -------------------------------- #
    train_loader_noaug = DataLoader(train_dataset_noaug, batch_size=args.batch_size,
                                    num_workers=args.workers, pin_memory=True,
                                    shuffle=False, drop_last=False)
    # -------------------------------- #

    return {
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'train_dataset_noaug': train_dataset_noaug,
        'train_loader_noaug': train_loader_noaug,
    }


def get_dataset_manager(args):
    if args.dataset.startswith("cifar"):
        print("Working on CIFAR10/20 dataset!")
        return get_dataset_manager_CIFAR(args)
    elif args.dataset == "stl10":
        print("Working on STL10 dataset!")
        return get_dataset_manager_STL10(args)
    else:
        raise ValueError("Do not support 'dataset'={}!".format(args.dataset))


# OK
def get_components(args):
    net = args.net
    arch = args.arch

    print(f"net={args.net}")
    print(f"arch={args.arch}")

    # Encoder
    # --------------------------------- #
    # ResNet18
    if net == "resnet18":
        if arch == "default":
            encoder = ResNet18_1(feat_dim=-1, in_channels=args.x_shape[0],
                                 base_out_channels=args.base_output_channels,
                                 bn_momentum=args.bn_momentum)
        elif arch == "zhirongw":
            encoder = ResNet18_2(low_dim=-1, in_channels=args.x_shape[0])

        elif arch == "wvangansbeke":
            encoder = resnet18()['backbone']

        elif arch == "torchvision":
            encoder = models.resnet18(num_classes=args.feat_dim)
            encoder.fc = nn.Identity()

        else:
            raise ValueError(f"Do not support arch={arch}!")

        hid_dim = 512

    elif net == "resnet34":
        if arch == "default":
            encoder = ResNet34_1(feat_dim=-1, in_channels=args.x_shape[0],
                                 base_out_channels=args.base_output_channels,
                                 bn_momentum=args.bn_momentum)

        elif arch == "zhirongw":
            encoder = ResNet34_2(low_dim=-1, in_channels=args.x_shape[0])

        elif arch == "wvangansbeke":
            encoder = resnet34()['backbone']

        elif arch == "torchvision":
            encoder = models.resnet34(num_classes=args.feat_dim)
            encoder.fc = nn.Identity()

        else:
            raise ValueError(f"Do not support arch={arch}!")

        hid_dim = 512

    elif net == "resnet50":
        if arch == "default":
            encoder = ResNet50_1(in_channels=-1,
                                 base_out_channels=args.base_output_channels,
                                 bn_momentum=args.bn_momentum)
        elif arch == "zhirongw":
            encoder = ResNet50_2(low_dim=-1, in_channels=args.x_shape[0])

        elif arch == "wvangansbeke":
            encoder = resnet50()['backbone']

        elif arch == "torchvision":
            encoder = models.resnet50(num_classes=args.feat_dim)
            encoder.fc = nn.Identity()

        else:
            raise ValueError(f"Do not support arch={arch}!")

        hid_dim = 2048

    else:
        raise ValueError("Do not support 'net'={}!".format(net))
    # --------------------------------- #

    # Projection head
    # --------------------------------- #
    if args.proj_head_type == "linear":
        proj_head = nn.Linear(hid_dim, args.feat_dim, bias=True)
        kaiming_uniform_(proj_head.weight)
        nn.init.zeros_(proj_head.bias)
    elif args.proj_head_type == "mlp2":
        linear_1 = nn.Linear(hid_dim, hid_dim, bias=True)
        kaiming_uniform_(linear_1.weight)
        nn.init.zeros_(linear_1.bias)

        linear_2 = nn.Linear(hid_dim, args.feat_dim, bias=True)
        kaiming_uniform_(linear_2.weight)
        nn.init.zeros_(linear_2.bias)

        proj_head = nn.Sequential(linear_1, nn.ReLU(), linear_2)
    else:
        raise ValueError(f"Do not support args.proj_head_type={args.proj_head_type}!")
    # --------------------------------- #

    # Classification head
    # --------------------------------- #
    class_head = []
    for i in range(args.num_class_subheads):
        if args.class_head_type == "linear":
            class_subhead = nn.Linear(hid_dim, args.num_clusters, bias=True)
            kaiming_uniform_(class_subhead.weight)
            nn.init.zeros_(class_subhead.bias)

        elif args.class_head_type == "mlp2":
            linear_1 = nn.Linear(hid_dim, hid_dim, bias=True)
            kaiming_uniform_(linear_1.weight)
            nn.init.zeros_(linear_1.bias)

            linear_2 = nn.Linear(hid_dim, args.num_clusters, bias=True)
            kaiming_uniform_(linear_2.weight)
            nn.init.zeros_(linear_2.bias)

            class_subhead = nn.Sequential(linear_1, nn.ReLU(), linear_2)
        else:
            raise ValueError(f"Do not support args.class_head_type={args.class_head_type}!")

        class_head.append(class_subhead)

    class_head = CustomModuleList(class_head)
    # --------------------------------- #

    return encoder, proj_head, class_head


# OK
def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    schedule_type = args.lr_schedule_type

    if schedule_type == "cosine":
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2

    elif schedule_type == "step":
        assert len(args.lr_decay_epochs) > 0
        steps = np.sum(epoch > np.array(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    elif schedule_type == "constant":
        lr = lr

    else:
        raise ValueError('Invalid learning rate schedule type {}'.format(schedule_type))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


# OK
def test(args, test_loader, model):
    assert model.num_clusters == args.num_classes, \
        f"model.num_clusters={model.num_clusters} " \
        f"while args.num_classes={args.num_classes}"

    model.eval()

    num_test = len(test_loader.dataset)
    num_heads = model.num_class_subheads
    device = model.device

    y_true = []
    ys_pred = []

    count = 0
    run_time = 0.0
    with torch.no_grad():
        for b, (x, y) in enumerate(test_loader):
            start_time = time()

            batch_results = model.cluster(x.to(device), stack_results=True)

            ys_pred.append(batch_results['ys_pred'].data.cpu())
            y_true.append(y)

            count += x.size(0)
            run_time += time() - start_time

            print("\rTest [{}/{}] ({:.2f}s)".format(
                count, num_test, (run_time / count),
            ), end="")

    assert count == num_test, f"count={count} while num_test={num_test}!"
    print()

    y_true = torch.cat(y_true, dim=0).numpy()
    ys_pred = torch.cat(ys_pred, dim=0).numpy()

    assert y_true.shape == (num_test,), f"y_true.shape={y_true.shape}"
    assert ys_pred.shape == (num_test, num_heads), f"ys_pred.shape={ys_pred.shape}"

    accs, nmis, aris = [], [], []
    for i in range(num_heads):
        acc = ACC(args.num_classes, ys_pred[:, i], y_true)['acc']
        nmi = NMI(ys_pred[:, i], y_true)
        ari = ARI(ys_pred[:, i], y_true)

        accs.append(acc)
        nmis.append(nmi)
        aris.append(ari)

    best_idx = np.argmax(accs)
    acc_best = accs[best_idx]
    nmi_best = nmis[best_idx]
    ari_best = aris[best_idx]

    acc_avg = np.mean(accs)
    nmi_avg = np.mean(nmis)
    ari_avg = np.mean(aris)

    model.train()

    return (acc_best, nmi_best, ari_best), (acc_avg, nmi_avg, ari_avg)


def main(args):
    # Device and randomness
    # ===================================== #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.set_printoptions(suppress=True, precision=3, threshold=np.inf, linewidth=1000)
    torch.set_num_threads(32)
    torch.set_num_interop_threads(32)
    # ===================================== #

    # Create output directory
    # ===================================== #
    RESUME = args.__dict__.pop('resume', False)
    RESUME_STEP = args.__dict__.pop('resume_step', -1)
    RESUME_BEST = args.__dict__.pop('resume_best', False)
    NEW_CONFIG = args.__dict__.pop('new_config', None)

    args.output_dir = os.path.join(args.output_dir, args.net + "_" + args.arch, args.model, args.run)

    if RESUME:
        assert not args.force_rm_dir, "'force_rm_dir' must be False when 'resume'=True!"
        assert os.path.exists(args.output_dir), "Output directory [{}] does not exist!".format(args.output_dir)

        import json
        config_file = os.path.join(args.output_dir, 'config.json')
        with open(config_file, 'r') as f:
            config = json.load(f)
        args.__dict__.update(config)
        print("Loaded old config from [{}]!".format(config_file))

        if len(NEW_CONFIG) > 0:
            print(f"NEW_CONFIG: {NEW_CONFIG}")
            args.__dict__.update(NEW_CONFIG)

            print("Update args with new config!")
            print("Save args with new config!")
            save_args(os.path.join(args.output_dir, 'config.json'), args)

    else:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        else:
            if args.force_rm_dir:
                import shutil
                shutil.rmtree(args.output_dir, ignore_errors=True)
                print("Removed '{}'".format(args.output_dir))
            else:
                raise ValueError("Output directory '{}' existed. 'force_rm_dir' "
                                 "must be set to True!".format(args.output_dir))
            os.mkdir(args.output_dir)

        save_args(os.path.join(args.output_dir, 'config.json'), args)

    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(args.__dict__)
    # ===================================== #

    # Load data
    # ===================================== #
    dataset_manager = get_dataset_manager(args)

    train_dataset = dataset_manager['train_dataset']
    train_loader = dataset_manager['train_loader']
    test_loader = dataset_manager['test_loader']

    num_train = len(train_dataset)
    steps_per_epoch = num_train // args.batch_size
    print(f"num_train: {num_train}")
    print(f"batch_size: {args.batch_size}")
    print(f"steps_per_epoch: {steps_per_epoch}")

    if args.log_freq < 0:
        args.log_freq = steps_per_epoch

    print(f"log_freq: {args.log_freq}")
    print(f"save_freq_epoch: {args.save_freq_epoch}")
    print(f"eval_freq_epoch: {args.eval_freq_epoch}")
    print()
    # ===================================== #

    # Build model
    # ===================================== #
    encoder, proj_head, class_head = get_components(args)
    model = CRLC(
        num_clusters=args.num_clusters, feat_dim=args.feat_dim,
        num_class_subheads=args.num_class_subheads,
        encoder=encoder, proj_head=proj_head, class_head=class_head,
        batch_size=args.batch_size,

        critic_type=args.critic_type,
        cons_type=args.cons_type,

        freeze_encoder=args.freeze_encoder,
        use_pretrained_net=args.use_pretrained_net,

        temperature=args.temperature,
        cluster_temperature=args.cluster_temperature,
        smooth_prob=args.smooth_prob, smooth_coeff=args.smooth_coeff,

        min_rate=args.min_rate, max_rate=args.max_rate,
        max_abs_logit=args.max_abs_logit,

        device=device)

    # We have to load model before using DataParallel
    if args.use_pretrained_net:
        assert os.path.exists(args.pretrained_file), f"File '{args.pretrained_file}' does not exist!"
        model.load_encoder_n_proj_head(
            args.pretrained_file,
            encoder_key="encoder_state_dict",
            proj_head_key="proj_head_state_dict")

    num_gpus = torch.cuda.device_count()
    print(f"num_gpus: {num_gpus}")
    if num_gpus > 1:
        print("Use data parallel!")
        model.encoder_proj_class = torch.nn.DataParallel(model.encoder_proj_class)
        cudnn.benchmark = True

    model.encoder_proj_class = model.encoder_proj_class.to(device)
    # print(f"encoder_proj_class:\n{model.encoder_proj_class}")

    loss_coeffs = {
        'contrast_cluster': args.contrast_cluster,
        'contrast_y': args.contrast_y,
        'cons_y': args.cons_y,
        'contrast_z': args.contrast_z,
        'entropy': args.entropy,
        'lower_clamp': args.lower_clamp,
        'upper_clamp': args.upper_clamp,
        'lower_logit_clamp': args.lower_logit_clamp,
        'upper_logit_clamp': args.upper_logit_clamp,
    }
    print(f"loss_coeffs: {loss_coeffs}")
    # ===================================== #

    # Build optimizer
    # ===================================== #
    wd_params, non_wd_params = [], []

    for param in model.get_all_train_params():
        if len(param.size()) == 1:
            non_wd_params.append(param)
        else:
            wd_params.append(param)

    # NOTE: Only apply weight decay for 'weights', not 'bias'
    param_list = [{'params': wd_params}, {'params': non_wd_params, 'weight_decay': 0}]

    if args.optim == 'adam':
        print("Use the Adam optimizer!")
        optim = torch.optim.Adam(param_list, lr=args.lr,
                                 betas=(args.beta1, args.beta2),
                                 weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        print("Use the AdamW optimizer!")
        optim = torch.optim.AdamW(param_list, lr=args.lr,
                                  betas=(args.beta1, args.beta2),
                                  weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        print("Use the SGD optimizer!")
        optim = torch.optim.SGD(param_list, lr=args.lr, momentum=args.momentum,
                                nesterov=args.nesterov, weight_decay=args.weight_decay)
    # ===================================== #

    # Create directories
    # ===================================== #
    asset_dir = make_dir_if_not_exist(os.path.join(args.output_dir, "asset"))

    log_dir = make_dir_if_not_exist(os.path.join(args.output_dir, "log"))
    train_log_file = os.path.join(log_dir, "train.log")
    test_log_file = os.path.join(log_dir, "test.log")

    summary_dir = make_dir_if_not_exist(os.path.join(args.output_dir, "summary"))
    model_dir = make_dir_if_not_exist(os.path.join(args.output_dir, "model"))
    # ===================================== #

    # Summarizers and Trackers
    # ===================================== #
    # Summarizer
    summary_writer = SummaryWriter(summary_dir)

    train_fetch_keys_1 = ['loss', 'contrast_cluster', 'contrast_y', 'cons_y', 'contrast_z', 'entropy']
    train_fetch_keys_2 = ['lower_clamp', 'upper_clamp', 'lower_logit_clamp', 'upper_logit_clamp']
    train_fetch_keys = train_fetch_keys_1 + train_fetch_keys_2
    train_summarizer = ScalarSummarizer([(key, 'mean') for key in train_fetch_keys])

    # Tracker
    save_tracker = SaveDirTracker(args.max_save, dir_path_prefix=os.path.join(model_dir, "step_"))
    save_best_tracker = SaveDirTracker(args.max_save_best, dir_path_prefix=os.path.join(model_dir, "best_step_"))

    best_results_tracker = BestResultsTracker([('acc', 'greater')], num_best=args.max_save_best)
    # ===================================== #

    # Resume if needed
    # ===================================== #
    if RESUME:
        if RESUME_BEST:
            save_dir = os.path.join(model_dir, "best_step_{}".format(RESUME_STEP))
        else:
            save_dir = os.path.join(model_dir, "step_{}".format(RESUME_STEP))

        # Reload model
        model.load_dir(save_dir)

        # Reload train state
        train_state = torch.load(os.path.join(save_dir, "train_state.pt"))

        # Reload optimizer
        optim.load_state_dict(train_state['optimizer'])

        # Reload saved steps info
        save_tracker.set_saved_steps(train_state['steps'])
        save_best_tracker.set_saved_steps(train_state['steps_best'])

        # Reload best results info
        best_results_tracker.set_best_results(train_state['best_results'])

        global_step = train_state['global_step']
        start_epoch = train_state['epoch']

        print()
        print("Resume training model at epoch={}, global_step={}!".format(start_epoch, global_step))
        print("saved_steps: {}".format(save_tracker.get_saved_steps()))
        print("saved_steps_best: {}".format(save_best_tracker.get_saved_steps()))
        print("saved_best_results: {}".format(best_results_tracker.get_best_results()))

    else:
        global_step = 0
        start_epoch = 0
    # ===================================== #

    log_time_start = time()
    y_prob_avg = []
    y_logit_min = []
    y_logit_max = []

    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(args, optim, epoch)
        print("\n\n" + "=" * 50)
        print(f"Epoch {epoch}/{args.epochs}, lr={lr:.6f}!")

        model.train()

        for b in range(steps_per_epoch):
            global_step += 1

            (x, xb, y) = next(train_loader)

            # Get data
            # --------------------------------------- #
            x = x.to(device)
            xb = xb.to(device)

            # print(f"[Outside] x.size(), xb.size(): {[x.size(), xb.size()]}")
            # --------------------------------------- #

            # Optimization
            # --------------------------------------- #
            batch_results = model.train_step(
                x, xb, loss_coeffs=loss_coeffs, optimizer=optim)

            # Accumulate results
            # --------------------------------------- #
            train_summarizer.accumulate(batch_results, 1)

            y_prob_avg.append(batch_results['y_prob_avg'])
            y_logit_min.append(batch_results['y_logit_min'])
            y_logit_max.append(batch_results['y_logit_max'])
            # --------------------------------------- #

            # Log
            # ---------------------------------- #
            if global_step % args.log_freq == 0:
                log_time_end = time()
                log_time_gap = (log_time_end - log_time_start)
                log_time_start = log_time_end

                summaries, train_results = train_summarizer.get_summaries_and_reset(summary_prefix='train')
                add_scalar_summaries(summary_writer, summaries, global_step)
                summary_writer.add_scalar('hyper/lr', optim.param_groups[0]['lr'], global_step)

                y_prob_avg = torch.stack(y_prob_avg, dim=0)
                y_prob_avg_ = y_prob_avg.mean(0).data.cpu().numpy()
                y_prob_avg = []

                y_logit_min = torch.stack(y_logit_min, dim=0)
                y_logit_min_ = y_logit_min.mean(0).data.cpu().numpy()
                y_logit_min = []

                y_logit_max = torch.stack(y_logit_max, dim=0)
                y_logit_max_ = y_logit_max.mean(0).data.cpu().numpy()
                y_logit_max = []

                log_str = "\n[Train][{}][{} ({})][{}], Epoch {}/{}, Batch {}/{}, Step {} ({:.2f}s)".format(
                    args.dataset, args.model, args.arch, args.run,
                    epoch, args.epochs, b, steps_per_epoch, global_step, log_time_gap) + \
                        "\n" + ", ".join(["{}: {:.4f}".format(key, train_results[key]) for key in train_fetch_keys_1]) + \
                        "\n" + ", ".join(["{}: {:.4f}".format(key, train_results[key]) for key in train_fetch_keys_2]) + \
                        "\ny_prob_avg: {}".format(y_prob_avg_) + \
                        "\ny_logit_min: {}".format(y_logit_min_) + \
                        "\ny_logit_max: {}".format(y_logit_max_)

                print(log_str)

                with open(train_log_file, "a") as f:
                    f.write(log_str)
                    f.write("\n")
                f.close()
            # ---------------------------------- #

            # Plot
            # ---------------------------------- #
            # if global_step % args.plot_freq == 0:
            #     save_file = os.path.join(inp_img_dir, "step_{}.png".format(global_step))
            #     save_images(save_file, postprocess_images(x), num_cols=16)
            # ---------------------------------- #

        # '''
        # Save
        # ---------------------------------- #
        if (epoch + 1) % args.save_freq_epoch == 0:
            print("Save at global_step={}!".format(global_step))
            save_dir = make_dir_if_not_exist(save_tracker.get_save_dir(global_step))
            model.save_dir(save_dir)
            save_tracker.update_and_delete_old_save(global_step)

            train_state = {
                'global_step': global_step,
                'epoch': epoch + 1,
                'optimizer': optim.state_dict(),
                'steps': save_tracker.get_saved_steps(),
                'steps_best': save_best_tracker.get_saved_steps(),
                'best_results': best_results_tracker.get_best_results(),
            }
            torch.save(train_state, os.path.join(save_dir, "train_state.pt"))
        # ---------------------------------- #
        # '''

        # Test
        # ---------------------------------- #
        if (epoch + 1) % args.eval_freq_epoch == 0:
            (acc, nmi, ari), (acc_avg, nmi_avg, ari_avg) = test(args, test_loader, model)

            test_summaries = [('test/acc', acc), ('test/nmi', nmi), ('test/ari', ari),
                              ('test/acc_avg', acc_avg), ('test/nmi_avg', nmi_avg),
                              ('test/ari_avg', ari_avg)]
            add_scalar_summaries(summary_writer, test_summaries, global_step)

            log_str = "\n[Test] Epoch {}, acc={:.2f}%, nmi={:.2f}%, ari={:.2f}%\n" \
                      "acc_avg={:.2f}%, nmi_avg={:.2f}%, ari_avg={:.2f}%".format(
                epoch, 100 * acc, 100 * nmi, 100 * ari, 100 * acc_avg, 100 * nmi_avg, 100 * ari_avg)
            print(log_str)

            with open(test_log_file, "a") as f:
                f.write(log_str)
                f.write("\n")
            f.close()

            is_better = best_results_tracker.check_and_update({'acc': np.max(acc)}, global_step)
            if is_better['acc']:
                print("The new accuracy is one of the best!")
                print("Save best at global_step={}!".format(global_step))

                save_dir = make_dir_if_not_exist(save_best_tracker.get_save_dir(global_step))
                model.save_dir(save_dir)
                save_best_tracker.update_and_delete_old_save(global_step)

                train_state = {
                    'global_step': global_step,
                    'epoch': epoch + 1,
                    'optimizer': optim.state_dict(),
                    'steps': save_tracker.get_saved_steps(),
                    'steps_best': save_best_tracker.get_saved_steps(),
                    'best_results': best_results_tracker.get_best_results(),
                }
                torch.save(train_state, os.path.join(save_dir, "train_state.pt"))
        # ---------------------------------- #

    print("Save at last step!")
    save_dir = save_tracker.get_save_dir(global_step)
    if not os.path.exists(save_dir):
        print(f"Save at global_step={global_step}!")
        save_dir = make_dir_if_not_exist(save_dir)
        model.save_dir(save_dir)
        save_tracker.update_and_delete_old_save(global_step)

        train_state = {
            'global_step': global_step,
            'epoch': args.epochs,
            'optimizer': optim.state_dict(),
            'steps': save_tracker.get_saved_steps(),
            'steps_best': save_best_tracker.get_saved_steps(),
            'best_results': best_results_tracker.get_best_results(),
        }
        torch.save(train_state, os.path.join(save_dir, "train_state.pt"))
    else:
        print(f"Have already saved at global_step={global_step}!")
    # ===================================== #


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
