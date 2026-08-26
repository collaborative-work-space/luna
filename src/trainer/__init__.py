# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import math
import os
import signal
import sys
import traceback
from pathlib import Path
from random import randint
import datetime
import torch
import wandb
import randomname
from pytorch_lightning.strategies import DDPStrategy
import numpy as np
import matplotlib.cm as cm
import torch.nn as nn
import torch.nn.functional as F

from util.distinct_colors import DistinctColors
from util.misc import visualize_depth, probability_to_normalized_entropy, get_boundary_mask
from util.warmup_scheduler import GradualWarmupScheduler
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from util.filesystem_logger import FilesystemLogger

def print_traceback_handler(sig, _frame):
    print(f'Received signal {sig}')
    bt = ''.join(traceback.format_stack())
    print(f'Requested stack trace:\n{bt}')


def quit_handler(sig, frame):
    print(f'Received signal {sig}, quitting.')
    sys.exit(1)


def register_debug_signal_handlers(sig=signal.SIGUSR1, handler=print_traceback_handler):
    print(f'Setting signal {sig} handler {handler}')
    signal.signal(sig, handler)


def register_quit_signal_handlers(sig=signal.SIGUSR2, handler=quit_handler):
    print(f'Setting signal {sig} handler {handler}')
    signal.signal(sig, handler)


def generate_experiment_name(name, config):
    if config.resume is not None:
        experiment = Path(config.resume).parents[1].name
        os.environ['experiment'] = experiment
    elif not os.environ.get('experiment'):
        experiment = f"{datetime.datetime.now().strftime('%m%d%H%M')}_{name}_{config.experiment}_{randomname.get_name()}"
        os.environ['experiment'] = experiment
    else:
        experiment = os.environ['experiment']
    return experiment


def create_trainer(name, config):
    if not config.wandb_main and config.suffix == '':
        config.suffix = '-dev'
    config.experiment = generate_experiment_name(name, config)
    if config.val_check_interval > 1:
        config.val_check_interval = int(config.val_check_interval)
    if config.seed is None:
        config.seed = randint(0, 999)
    if isinstance(config.image_dim, int):
        config.image_dim = [config.image_dim, config.image_dim]
    assert config.image_dim[0] == config.image_dim[1], "only 1:1 supported"  # TODO: fix dataprocessing bug limiting this

    seed_everything(config.seed)

    register_debug_signal_handlers()
    register_quit_signal_handlers()

    # noinspection PyUnusedLocal
    filesystem_logger = FilesystemLogger(config)

    # use wandb logger instead
    if config.logger == 'wandb':
        logger = WandbLogger(
            project=f'{name}{config.suffix}', 
            name=config.experiment, 
            id=config.experiment, 
            settings=wandb.Settings(
                start_method='thread',
                _disable_stats=True,  # Disable stats collection
                _disable_meta=True,   # Disable meta data collection
                _disable_job_creation=True,  # Disable job creation
                mode='offline' if not config.wandb_online else 'online'  # Allow offline mode
            )
        )
    else:
        logger = TensorBoardLogger(name='tb', save_dir=(Path("runs") / config.experiment))

    checkpoint_callback = ModelCheckpoint(dirpath=(Path("runs") / config.experiment / "checkpoints"),
                                          save_top_k=-1,
                                          verbose=False,
                                          every_n_epochs=config.save_epoch)
    
    accelerator = config.trainer.accelerator
    devices     = config.trainer.devices
    strategy    = config.trainer.strategy

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        num_sanity_val_steps=config.sanity_steps,
        max_epochs=config.trainer.max_epochs,
        callbacks=[checkpoint_callback],
        val_check_interval=float(min(config.val_check_interval, 1)),
        check_val_every_n_epoch=max(1, config.val_check_interval),
        resume_from_checkpoint=config.resume,
        logger=logger,
        benchmark=True
    )
    return trainer

def step(opt, modules):
    for module in modules:
        for param in module.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
    opt.step()


def get_optimizer_and_scheduler(params, config, betas=None):
    opt = torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay, betas=betas)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=config.decay_step, gamma=config.decay_gamma)
    if config.warmup_epochs > 0:
        scheduler = GradualWarmupScheduler(opt, multiplier=config.warmup_multiplier, total_epoch=config.warmup_epochs, after_scheduler=scheduler)
    return opt, scheduler



def visualize_panoptic_outputs(p_rgb, p_semantics, p_instances, p_depth, rgb, semantics, instances, depth, H, W, thing_classes, visualize_entropy=True):
    alpha = 0.65
    distinct_colors = DistinctColors()
    device = torch.device("cuda")

    p_rgb = p_rgb.to(device)
    p_semantics = p_semantics.to(device)
    p_instances = p_instances.to(device)
    p_depth = p_depth.to(device) if p_depth is not None else None
    
    if rgb is not None:
        rgb = rgb.to(device)
    if semantics is not None:
        semantics = semantics.to(device)
    if instances is not None:
        instances = instances.to(device)
    if depth is not None:
        depth = depth.to(device)

    img = p_rgb.view(H, W, 3)
    img = torch.clamp(img, 0, 1).permute(2, 0, 1)

    if visualize_entropy:
        img_sem_entropy = visualize_depth(probability_to_normalized_entropy(torch.nn.functional.softmax(p_semantics, dim=-1)).reshape(H, W), minval=0.0, maxval=1.00, use_global_norm=True).to(device)
    else:
        img_sem_entropy = torch.zeros_like(img)

    if p_depth is not None:
        ren_depth = visualize_depth(p_depth.view(H, W)).to(device)
    else:
        ren_depth = torch.zeros_like(img)

    if len(p_instances.shape) > 1 and len(p_semantics.shape) > 1:
        p_instances = p_instances.argmax(dim=1)
        p_semantics = p_semantics.argmax(dim=1)
    
    img_semantics_colors = distinct_colors.apply_colors_fast_torch(p_semantics)
    img_semantics_colors = img_semantics_colors.to(device)  # in case it was CPU
    
    # Now shape = [H, W, 3], permute, alpha-blend with 'img':
    img_semantics_colors = img_semantics_colors.view(H, W, 3).permute(2, 0, 1)
    img_semantics = img_semantics_colors * alpha + img * (1 - alpha)
    
    # boundary mask on GPU
    boundaries_img_semantics = get_boundary_mask(p_semantics.view(H, W))
    boundaries_img_semantics = boundaries_img_semantics.to(device)
    img_semantics[:, boundaries_img_semantics > 0] = 0
    
    colored_img_instance = distinct_colors.apply_colors_fast_torch(p_instances).float()
    colored_img_instance = colored_img_instance.to(device)
    
    boundaries_img_instances = get_boundary_mask(p_instances.view(H, W))
    boundaries_img_instances = boundaries_img_instances.to(device)
    colored_img_instance[boundaries_img_instances.reshape(-1) > 0, :] = 0
    
    # 'thing_mask' must also be on GPU
    # sum(...) returns GPU since p_semantics is on GPU
    thing_mask = torch.logical_not(sum(p_semantics == s for s in thing_classes).bool())
    colored_img_instance[thing_mask, :] = p_rgb[thing_mask, :]
    
    img_instances = colored_img_instance.view(H, W, 3).permute(2, 0, 1) * alpha + img * (1 - alpha)
    
    # ---------------------------------------------------------------------
    # 8) If ground-truth is available:
    # ---------------------------------------------------------------------
    if rgb is not None and semantics is not None and instances is not None:
        # 'img_gt', 'semantics', 'instances' are on GPU (see above).
        img_gt = rgb.view(H, W, 3).permute(2, 0, 1)
        gt_depth = visualize_depth(depth.view(H, W)).to(device)
        depth_diff = torch.abs(p_depth.view(H, W) - depth.view(H, W))
        depth_diff = visualize_depth(depth_diff).to(device)  # Normalize for visualization

        
        # semantics colors
        sem_gt_colored = distinct_colors.apply_colors_fast_torch(semantics).float()
        sem_gt_colored = sem_gt_colored.to(device)
        sem_gt_colored = sem_gt_colored.view(H, W, 3).permute(2, 0, 1)
        img_semantics_gt = sem_gt_colored * alpha + img_gt * (1 - alpha)
        
        boundaries_img_semantics_gt = get_boundary_mask(semantics.view(H, W)).to(device)
        img_semantics_gt[:, boundaries_img_semantics_gt > 0] = 0
        
        # instances colors
        inst_gt_colored = distinct_colors.apply_colors_fast_torch(instances).float()
        inst_gt_colored = inst_gt_colored.to(device)
        boundaries_img_instances_gt = get_boundary_mask(instances.view(H, W)).to(device)
        
        # If you do any CPU indexing, convert to GPU indexing
        inst_gt_colored[instances == 0, :] = rgb[instances == 0, :]
        
        img_instances_gt = inst_gt_colored.view(H, W, 3).permute(2, 0, 1) * alpha + img_gt * (1 - alpha)
        img_instances_gt[:, boundaries_img_instances_gt > 0] = 0

        
        # final stack must have all on GPU
        left_stack = torch.stack([
            img_gt,
            img_semantics_gt,
            img_instances_gt,
            gt_depth,        # also GPU
            depth_diff
            ])
        right_stack = torch.stack([
            img,
            img_semantics,
            img_instances,
            ren_depth,
            img_sem_entropy
        ])
        stack = torch.cat([left_stack, right_stack], dim=0)
    else:
        # Single stack
        stack = torch.stack([img, img_semantics, img_instances, ren_depth, img_sem_entropy])
    
    return stack
