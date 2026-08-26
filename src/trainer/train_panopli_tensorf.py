# Copyright (c) Meta Platforms, Inc. All Rights Reserved


import math
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from FAMO.famo import FAMO

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing
from pathlib import Path
from torch.utils.data import DataLoader
from torch_scatter import scatter_mean
from torchvision.utils import save_image, make_grid
import hydra
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from tabulate import tabulate



from dataset import get_dataset, get_inconsistent_single_dataset, get_segment_dataset
from model.loss.loss import TVLoss, get_semantic_weights, SCELoss, GeometryAwareLoss
from model.radiance_field.tensoRF import TensorVMSplit
from model.renderer.panopli_tensoRF_renderer import TensoRFRenderer
from util.distinct_colors import DistinctColors
from trainer import create_trainer, get_optimizer_and_scheduler, visualize_panoptic_outputs
from util.metrics import psnr, ConfusionMatrix
from util.panoptic_quality import panoptic_quality

torch.multiprocessing.set_sharing_strategy('file_system') 
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False

class TensoRFTrainer(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        
        self.train_set, self.val_set = get_dataset(config)
        if config.visualized_indices is None:
            config.visualized_indices = list(range(0, len(self.val_set), int(1 / 0.15)))
            if len(config.visualized_indices) < 16:
                config.visualized_indices = list(range(0, len(self.val_set)))[:16]
        config.instance_optimization_epoch = config.instance_optimization_epoch + config.late_semantic_optimization
        config.segment_optimization_epoch = config.segment_optimization_epoch + config.late_semantic_optimization
        self.config = config

        self.current_lambda_dist_reg = 0
        if self.config.segment_grouping_mode != "none":
            self.train_segment_set = get_segment_dataset(self.config)
        self.save_hyperparameters(config)
        total_classes = len(self.train_set.segmentation_data.bg_classes) + len(self.train_set.segmentation_data.fg_classes)
        output_mlp_semantics = torch.nn.Identity() if self.config.semantic_weight_mode != "softmax" else torch.nn.Softmax(dim=-1)
        self.model = TensorVMSplit([config.min_grid_dim, config.min_grid_dim, config.min_grid_dim], num_semantics_comps=(32, 32, 32),
                                   num_semantic_classes=total_classes, dim_feature_instance=self.config.max_instances,
                                   output_mlp_semantics=output_mlp_semantics, use_semantic_mlp=self.config.use_mlp_for_semantics,
                                   use_feature_reg=self.config.use_feature_regularization)
        self.renderer = TensoRFRenderer(self.train_set.scene_bounds, [config.min_grid_dim, config.min_grid_dim, config.min_grid_dim],
                                        semantic_weight_mode=self.config.semantic_weight_mode, stop_semantic_grad=config.stop_semantic_grad)
        semantic_weights = get_semantic_weights(config.reweight_fg, self.train_set.segmentation_data.fg_classes, self.train_set.segmentation_data.num_semantic_classes)
        semantic_weights[0] = config.weight_class_0
        self.loss_rgb = torch.nn.MSELoss(reduction='mean')
        self.loss_feat = torch.nn.L1Loss(reduction='mean')
        self.loss_depth = GeometryAwareLoss()
        self.tv_regularizer = TVLoss()
        if not self.config.use_symmetric_ce:
            self.loss_semantics = torch.nn.CrossEntropyLoss(reduction='none', weight=semantic_weights)
        else:
            self.loss_semantics = SCELoss(self.config.ce_alpha, self.config.ce_beta, semantic_weights)
        self.loss_instances_cluster = torch.nn.CrossEntropyLoss(reduction='none')
        
        self.output_dir_result_images = Path(f'runs/{self.config.experiment}/images')
        self.output_dir_result_images.mkdir(exist_ok=True)
        self.output_dir_result_clusters = Path(f'runs/{self.config.experiment}/instance_clusters')
        self.output_dir_result_clusters.mkdir(exist_ok=True)
        self.automatic_optimization = False
        self.distinct_colors = DistinctColors()

        # Initialize FAMO for main tasks
        self.famo = FAMO(n_tasks=5, device=self.device, 
                         gamma=getattr(self.config.famo, 'gamma'),
                         w_lr=getattr(self.config.famo, 'meta_lr'))
        

    def configure_optimizers(self):
        params = self.model.get_optimizable_parameters(self.config.lr * 20, self.config.lr, weight_decay=self.config.weight_decay)
        optimizer, scheduler = get_optimizer_and_scheduler(params, self.config, betas=(0.9, 0.99))
        param_instance = self.model.get_optimizable_instance_parameters(self.config.lr * 20, self.config.lr)
        optimizer_instance, scheduler_instance = get_optimizer_and_scheduler(param_instance, self.config, betas=(0.9, 0.999))
        
        # FAMO optimizers
        
        return [
            {"optimizer": optimizer, "lr_scheduler": scheduler},
            {"optimizer": optimizer_instance, "lr_scheduler": scheduler_instance},
        ]

    def forward(self, rays, depth=None, is_train=True):
        B = rays.shape[0]
        out_rgb, out_semantics, out_instances, out_depth, out_regfeat, out_dist_regularizer = [], [], [], [], [], []
        for i in range(0, B, self.config.chunk):
            out_rgb_, out_semantics_, out_instances_, out_depth_, out_regfeat_, out_dist_reg_ = self.renderer(self.model, rays[i: i + self.config.chunk], self.config.perturb, self.train_set.white_bg, is_train)
            out_rgb.append(out_rgb_)
            out_semantics.append(out_semantics_)
            out_instances.append(out_instances_)
            out_regfeat.append(out_regfeat_)
            out_depth.append(out_depth_)
            out_dist_regularizer.append(out_dist_reg_.unsqueeze(0))
        out_rgb = torch.cat(out_rgb, 0)
        out_instances = torch.cat(out_instances, 0)
        out_depth = torch.cat(out_depth, 0)
        out_semantics = torch.cat(out_semantics, 0)
        out_regfeat = torch.cat(out_regfeat, 0)
        out_dist_regularizer = torch.mean(torch.cat(out_dist_regularizer, 0))
        return out_rgb, out_semantics, out_instances, out_depth, out_regfeat, out_dist_regularizer

    def forward_instance(self, rays, is_train):
        B = rays.shape[0]
        out_feats_instance = []
        for i in range(0, B, self.config.chunk):
            batch_rays = rays[i: i + self.config.chunk]
            out_feats_instance_ = self.renderer.forward_instance_feature(self.model, batch_rays, self.config.perturb, is_train)
            out_feats_instance.append(out_feats_instance_)
        out_feats_instance = torch.cat(out_feats_instance, 0)
        return out_feats_instance

    def forward_segments(self, rays, is_train):
        B = rays.shape[0]
        out_feats_segments = []
        for i in range(0, B, self.config.chunk_segment):
            batch_rays = rays[i: i + self.config.chunk_segment]
            out_feats_segments_ = self.renderer.forward_segment_feature(self.model, batch_rays, self.config.perturb, is_train)
            out_feats_segments.append(out_feats_segments_)
        out_feats_segments = torch.cat(out_feats_segments, 0)
        return out_feats_segments

    def training_step(self, batch, batch_idx):
        opt_rgb, opt_inst = self.optimizers()[:2]
        opt_rgb.zero_grad(set_to_none=True)
        rays, rgbs, semantics, probs, confs, masks, feats, depth = batch[0]['rays'], batch[0]['rgbs'], batch[0]['semantics'], batch[0]['probabilities'], batch[0]['confidences'], batch[0]['mask'], batch[0]['feats'], batch[0]['depth']
        output_rgb, output_semantics, output_instances, output_depth, output_feats, loss_dist_reg = self(rays, True)
        # create a 3D mask for RGB
        mask3 = masks.unsqueeze(-1)  # shape: [B, 1]
        output_rgb   = output_rgb.clone().masked_fill(~mask3,   0.)
        output_depth = output_depth.clone().masked_fill(~masks, 0.)
        rgbs = rgbs.masked_fill(~mask3, 0.)
        depth = depth.masked_fill(~masks, 0.)
        confs = confs.masked_fill(~masks, 0.)

        # Always compute all losses (no lambda gating)
        loss_rgb = self.loss_rgb(output_rgb, rgbs)
        loss_tv = (self.model.tv_loss_density(self.tv_regularizer) * self.config.lambda_tv_density + self.model.tv_loss_appearance(self.tv_regularizer) * self.config.lambda_tv_appearance + (self.model.tv_loss_semantics(self.tv_regularizer) * self.config.lambda_tv_semantics * (1 if self.current_epoch >= self.config.late_semantic_optimization else 0)))
        loss_feat = (self.loss_feat(output_feats, feats).mean() if self.config.use_feature_regularization else torch.tensor(0., device=self.device))
        rgb_total = loss_rgb + loss_tv + loss_dist_reg * self.current_lambda_dist_reg + loss_feat * self.config.lambda_feat
        self.log("train/loss_rgb", loss_rgb, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/loss_feat", loss_feat, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        # Compute geometry losses based on configuration
        geo = self.loss_depth(output_depth, depth)
        loss_depth = (
                geo["l1"] * self.config.lambda_L1 +
                geo["grad"] * self.config.lambda_grad
            )
        self.log("train/loss_depth", loss_depth,
                on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)


        # Semantics Loss
        if self.config.probabilistic_ce_mode == "TTAConf":
            loss_semantics = (self.loss_semantics(output_semantics, probs) * confs).mean()
        elif self.config.probabilistic_ce_mode == "NoTTAConf":
            loss_semantics = (self.loss_semantics(output_semantics, semantics) * confs).mean()
        else:
            loss_semantics = self.loss_semantics(output_semantics, semantics).mean()

        # Segment Clustering Loss
        loss_seg = torch.tensor(0., device=self.device)
        if self.current_epoch >= self.config.segment_optimization_epoch:
            batch[2]["rays"] = torch.cat(batch[2]['rays'], dim=0)
            batch[2]['group'] = torch.cat(batch[2]['group'], dim=0)
            batch[2]['confidences'] = torch.cat(batch[2]['confidences'], dim=0)
            semantic_features = self.forward_segments(batch[2]["rays"], True)
            batch_target_mean = torch.zeros(self.config.batch_size_segments, semantic_features.shape[-1], device=semantic_features.device)
            scatter_mean(semantic_features, batch[2]['group'], 0, batch_target_mean)
            target = batch_target_mean[batch[2]['group'], :].argmax(-1)
            loss_seg = (self.loss_semantics(semantic_features, target) * batch[2]['confidences']).mean()
            self.log(f"train/loss_segment", loss_seg, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)

        # Instance Clustering Loss
        loss_inst = torch.tensor(0., device=self.device)
        if self.current_epoch >= self.config.instance_optimization_epoch:
            opt_inst.zero_grad(set_to_none=True)
            for img_idx in range(len(batch[1]['rays'])):
                instance_features = self.forward_instance(batch[1]['rays'][img_idx], True)
                loss_inst += self.calculate_instance_clustering_loss(batch[1]['instances'][img_idx], instance_features, batch[1]['confidences'][img_idx])
            self.manual_backward(loss_inst, retain_graph=True)
            opt_inst.step()
            self.log(f"train/loss_clustering", loss_inst, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)

        losses = {
            'rgb':       rgb_total,
            'depth':     loss_depth,
            'semantics': loss_semantics,
        }
        if self.current_epoch >= self.config.segment_optimization_epoch:
            losses['segment'] = loss_seg
        if self.current_epoch >= self.config.instance_optimization_epoch:
            losses['instance'] = loss_inst.detach()

        # —— stack and apply FAMO weighting ——  
        names, vals = zip(*losses.items())
        loss_tensor = torch.stack(vals, 0).to(self.device)
        total_loss, weights = self.famo.get_weighted_loss(loss_tensor)
        self.manual_backward(total_loss)
        opt_rgb.step()
        self.famo.update(loss_tensor)

        for i, name in enumerate(names):
            self.log(f"famo/weight_{name}", float(weights[i].detach().cpu()), on_step=True, sync_dist=True)
        return total_loss

    def calculate_instance_clustering_loss(self, labels_gt, instance_features, confidences):
        virtual_gt_labels = self.create_virtual_gt_with_linear_assignment(labels_gt, instance_features)
        predicted_labels = instance_features.argmax(dim=-1)
        if torch.any(virtual_gt_labels != predicted_labels):
            return (self.loss_instances_cluster(instance_features, virtual_gt_labels) * confidences).mean()
        return instance_features.sum() * 0.0

    @torch.no_grad()
    def create_virtual_gt_with_linear_assignment(self, labels_gt, predicted_scores):
        labels = sorted(torch.unique(labels_gt).cpu().tolist())[:predicted_scores.shape[-1]]
        predicted_probabilities = torch.softmax(predicted_scores, dim=-1)
        cost_matrix = np.zeros([len(labels), predicted_probabilities.shape[-1]])
        for lidx, label in enumerate(labels):
            cost_matrix[lidx, :] = -(predicted_probabilities[labels_gt == label, :].sum(dim=0) / ((labels_gt == label).sum() + 1e-4)).cpu().numpy()
        assignment = scipy.optimize.linear_sum_assignment(np.nan_to_num(cost_matrix))
        new_labels = torch.zeros_like(labels_gt)
        for aidx, lidx in enumerate(assignment[0]):
            new_labels[labels_gt == labels[lidx]] = assignment[1][aidx]
        return new_labels

    def calculate_segment_clustering_loss(self, sem_features, confidences):
        target = torch.mean(torch.softmax(sem_features, dim=-1), dim=0).detach().unsqueeze(0).expand(sem_features.shape[0], -1)
        if self.config.segment_grouping_mode == "argmax_noconf":
            return self.loss_semantics(sem_features, target.argmax(-1)).mean()
        elif self.config.segment_grouping_mode == "argmax_conf":
            return (self.loss_semantics(sem_features, target.argmax(-1)) * confidences).mean()
        elif self.config.segment_grouping_mode == "prob_noconf":
            return self.loss_semantics(sem_features, target).mean()
        elif self.config.segment_grouping_mode == "prob_conf":
            return (self.loss_semantics(sem_features, target) * confidences).mean()
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        rays, rgbs, semantics, instances, mask, depth = batch['rays'].squeeze(), batch['rgbs'].squeeze(), batch['semantics'].squeeze(), batch['instances'].squeeze(), batch['mask'].squeeze(), batch['depth'].squeeze()
        rs_semantics, rs_instances = batch['rs_semantics'].squeeze(), batch['rs_instances'].squeeze()
        probs, confs = batch['probabilities'].squeeze(), batch['confidences'].squeeze()
        output = self(rays, depth=depth, is_train=False)
        output_rgb, output_semantics, output_instances, output_depth, _, _ = output
        mask3 = mask.unsqueeze(-1)  # [H*W,1] for RGB channels
        output_rgb = output_rgb.masked_fill(~mask3, 0.)
        rgbs = rgbs.masked_fill(~mask3, 0.)
        output_depth = output_depth.masked_fill(~mask, 0.)
        depth = depth.masked_fill(~mask, 0.)
        loss_rgb = self.loss_rgb(output_rgb, rgbs)

        if self.train_set.normscene_scale is not None:
            output_depth = output_depth * self.train_set.normscene_scale
        geo = self.loss_depth(output_depth, depth)
        loss_depth = (
            geo["l1"] * self.config.lambda_L1 +
            geo["grad"] * self.config.lambda_grad
            )
        
        self.log("val/loss_rgb", loss_rgb, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        metric_psnr = psnr(output_rgb, rgbs)
        self.log("val/psnr", metric_psnr, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        
        output_semantics = output_semantics.masked_fill(~mask3, 0.)
        semantics = semantics.masked_fill(~mask, 0)
        pred_sem = output_semantics.argmax(dim=1)
        pred_sem = pred_sem.masked_fill(~mask, 0)
        if self.config.probabilistic_ce_mode == "TTAConf":
            loss_semantics = (self.loss_semantics(output_semantics, probs) * confs).mean()
        elif self.config.probabilistic_ce_mode == "NoTTAConf":
            loss_semantics = (self.loss_semantics(output_semantics, semantics) * confs).mean()
        else:
            loss_semantics = self.loss_semantics(output_semantics, semantics).mean()

        val_cm = ConfusionMatrix(num_classes=self.model.num_semantic_classes, ignore_class=[0])
        metric_iou = val_cm.add_batch(pred_sem.cpu().numpy(), semantics.cpu().numpy(), return_miou=True)
        pano_pred = torch.cat([pred_sem.unsqueeze(1), output_instances.argmax(dim=1).unsqueeze(1)], dim=1)
        pano_target = torch.cat([semantics.unsqueeze(1), instances.unsqueeze(1)], dim=1)
        metric_pq, metric_sq, metric_rq = panoptic_quality(pano_pred, pano_target, self.train_set.things_filtered, self.train_set.stuff_filtered, allow_unknown_preds_category=True)
        
        self.log("val/iou", metric_iou, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("val/pq", metric_pq, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("val/sq", metric_sq, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("val/rq", metric_rq, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("val/loss_semantics", loss_semantics, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        
        have_rs = False
        if self.current_epoch >= self.config.segment_optimization_epoch:
            val_rs_cm = ConfusionMatrix(num_classes=self.model.num_semantic_classes,
                                        ignore_class=list(self.train_set.faulty_classes))
            output_semantics_with_invalid = output_semantics.detach().argmax(dim=1)
            metric_rs_iou = val_rs_cm.add_batch(
                output_semantics_with_invalid.cpu().numpy(),
                rs_semantics.cpu().numpy(),
                return_miou=True
            )
            pano_rs_pred = torch.cat(
                [output_semantics_with_invalid.unsqueeze(1),
                output_instances.argmax(dim=1).unsqueeze(1)], dim=1)
            pano_rs_target = torch.cat(
                [rs_semantics.unsqueeze(1), rs_instances.unsqueeze(1)], dim=1)
            metric_rs_pq, metric_rs_sq, metric_rs_rq = panoptic_quality(
                pano_rs_pred, pano_rs_target,
                self.train_set.things_filtered, self.train_set.stuff_filtered,
                allow_unknown_preds_category=True
            )
            self.log("val_rs/iou", metric_rs_iou, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log("val_rs/pq",  metric_rs_pq,  on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log("val_rs/sq",  metric_rs_sq,  on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log("val_rs/rq",  metric_rs_rq,  on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            have_rs = True

        inst_iou = None
        if self.current_epoch >= self.config.instance_optimization_epoch:
            pred_inst = output_instances.argmax(dim=1).masked_fill(~mask, 0)
            inst_cm = ConfusionMatrix(num_classes=self.config.max_instances, ignore_class=[])
            inst_iou = inst_cm.add_batch(pred_inst.cpu().numpy(),
                                     instances.cpu().numpy(),
                                     return_miou=True)
            self.log("val/instance_iou", inst_iou,
                 on_step=False, on_epoch=True, sync_dist=True)


        result = {
            "loss_rgb":   float(loss_rgb.item()),
            "loss_depth": float(loss_depth.item()),
            "loss_sem":   float(loss_semantics.item()),
            "psnr":       float(metric_psnr.item()),
            "iou":        float(metric_iou),
            "pq":         float(metric_pq),
            "sq":         float(metric_sq),
            "rq":         float(metric_rq),
            }

        if have_rs:
            result.update({
                "val_rs_iou": float(metric_rs_iou),
                "val_rs_pq":  float(metric_rs_pq),
                "val_rs_sq":  float(metric_rs_sq),
                "val_rs_rq":  float(metric_rs_rq),
            })

        # attach instance metric only if computed
        if inst_iou is not None:
            result["instance_iou"] = float(inst_iou)

        return result
                
    @rank_zero_only
    def validation_epoch_end(self, val_step_outputs):
        print()
        headers = ['loss_rgb','loss_depth','loss_sem','psnr','iou','pq','sq','rq']

        # Add optional columns only if present
        if any('val_rs_iou' in o for o in val_step_outputs):
            headers += ['val_rs_iou','val_rs_pq','val_rs_sq','val_rs_rq']
        if any('instance_iou' in o for o in val_step_outputs):
            headers += ['instance_iou']

        # Mean over available items; skip missing keys
        row = []
        for h in headers:
            vals = [o[h] for o in val_step_outputs if h in o]
            row.append(np.mean(vals) if len(vals) > 0 else float('nan'))

        table = [tuple(headers), tuple(row)]
        print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

        H, W = self.config.image_dim[0], self.config.image_dim[1]
        (self.output_dir_result_clusters / f"{self.global_step:06d}").mkdir(exist_ok=True)
        self.renderer.export_instance_clusters(self.model, self.output_dir_result_clusters / f"{self.global_step:06d}")
        for batch_idx, batch in enumerate(self.val_dataloader()):
            if batch_idx in self.config.visualized_indices:
                
                rays, rgbs, semantics, instances, depth = batch['rays'].squeeze().to(self.device), batch['rgbs'].squeeze().to(self.device), \
                                                   batch['semantics'].squeeze().to(self.device), batch['instances'].squeeze().to(self.device), \
                                                   batch['depth'].squeeze().to(self.device)
                rs_semantics, rs_instances = batch['rs_semantics'].squeeze().to(self.device), batch['rs_instances'].squeeze().to(self.device)
                mask = batch['mask'].squeeze().to(self.device)
                output = self(rays, depth=depth, is_train=False)
                output_rgb, output_semantics, output_instances, output_depth, _, _ = output
                mask3 = mask.unsqueeze(-1)
                output_rgb   = output_rgb.clone().masked_fill(~mask3,   0.)
                output_depth = output_depth.clone().masked_fill(~mask,  0.)
                output_semantics = output_semantics.clone().masked_fill(~mask3, 0.)
                output_instances = output_instances.clone().masked_fill(~mask3, 0.)
                rgbs         = rgbs.clone().masked_fill(~mask3, 0.)
                stack = visualize_panoptic_outputs(output_rgb, output_semantics, output_instances, output_depth, rgbs, rs_semantics, rs_instances, depth, H, W, thing_classes=self.train_set.segmentation_data.fg_classes)
                save_image(stack, self.output_dir_result_images / f"{self.global_step:06d}_{batch_idx:04d}.jpg", value_range=(0, 1), nrow=5, normalize=True)
                if self.config.logger == 'wandb':
                    self.logger.log_image(key=f"images/{batch_idx:04d}", images=[make_grid(stack, value_range=(0, 1), nrow=5, normalize=True)])
                else:
                    self.logger.experiment.add_image(f'visuals/{batch_idx:04d}', make_grid(stack, value_range=(0, 1), nrow=5, normalize=True), global_step=self.global_step)

    def train_dataloader(self):
        loaders = {
            0: DataLoader(self.train_set, self.config.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=self.config.num_workers)
        }
        train_instance_set = get_inconsistent_single_dataset(self.config)
        assert len(train_instance_set) > 0, f"Warning: Empty instance dataset"
        loaders[1] = DataLoader(train_instance_set, self.config.batch_size_contrastive, shuffle=True, drop_last=True, collate_fn=train_instance_set.collate_fn, num_workers=0)
        if self.config.segment_grouping_mode != "none":
            loaders[2] = DataLoader(self.train_segment_set, self.config.batch_size_segments, shuffle=False, drop_last=True, collate_fn=self.train_segment_set.collate_fn, num_workers=0)
        return loaders

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=1, shuffle=False, drop_last=False, num_workers=0)

    def on_train_end(self):
        self.loss_log_file.close()  # Close the file when training ends

    def on_train_epoch_start(self):
        self.current_lambda_dist_reg = self.config.lambda_dist_reg * (1 - math.exp(-0.25 * self.current_epoch))
        
        super().on_train_epoch_start()
        if self.current_epoch in self.config.bbox_aabb_reset_epochs:
            self.renderer.update_bbox_aabb_and_shrink(self.model)
        if self.current_epoch in self.config.grid_upscale_epochs:
            num_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(self.config.min_grid_dim**3), np.log(self.config.max_grid_dim**3), len(self.config.grid_upscale_epochs)+1))).long()).tolist()[1:]
            target_num_voxels = num_voxel_list[self.config.grid_upscale_epochs.index(self.current_epoch)]
            target_resolution = self.renderer.get_target_resolution(target_num_voxels)
            self.config.weight_decay = 0
            self.model.upsample_volume_grid(target_resolution)
            self.renderer.update_step_size(target_resolution)
            self.trainer.strategy.setup_optimizers(self.trainer)
        if self.config.segment_grouping_mode != "none" and self.current_epoch >= self.config.segment_optimization_epoch:
            self.train_segment_set.enabled = True

    def on_load_checkpoint(self, checkpoint):
        for epoch in self.config.grid_upscale_epochs[::-1]:
            if checkpoint['epoch'] >= epoch:
                grid_dim = checkpoint["state_dict"]["renderer.grid_dim"].cpu()
                self.model.upsample_volume_grid(grid_dim)
                self.renderer.update_step_size(grid_dim)
                self.config.weight_decay = 0
                self.trainer.strategy.setup_optimizers(self.trainer)
                break


@hydra.main(config_path='../config', config_name='panopli', version_base='1.2')
def main(config):
    trainer = create_trainer("PanopLi", config)
    model = TensoRFTrainer(config)
    trainer.fit(model)


if __name__ == '__main__':
    
    main()
