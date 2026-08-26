# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch
from torch import nn
import torch.nn.functional as F

#from util.camera import project_3d_to_2d 


class NeRFMSELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, inputs, targets):
        loss_coarse = torch.zeros([1], device=targets.device)
        loss_fine = torch.zeros([1], device=targets.device)
        import ipdb; ipdb.set_trace()
        if 'rgb_coarse' in inputs:
            loss_coarse = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss_fine = self.loss(inputs['rgb_fine'], targets)
        return loss_coarse, loss_fine


class GeometryAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.silog_loss = lambda x, y: torch.mean((torch.log(x + 1e-6) - torch.log(y + 1e-6)) ** 2)

    def forward(self, predicted_depth, gt_depth):
        assert predicted_depth.dim() == 1 and gt_depth.dim() == 1, "Depth must be 1D (per-ray)."
        valid_mask = ((gt_depth > 0) & (gt_depth < 10)).float()

        l1_loss_value = self.l1_loss(predicted_depth * valid_mask, gt_depth * valid_mask)
        silog_loss_value = self.silog_loss(predicted_depth * valid_mask, gt_depth * valid_mask)
        grad_loss = self.depth_gradient_loss(predicted_depth, gt_depth, valid_mask)
        normal_loss = self.normal_consistency_loss(predicted_depth, gt_depth, valid_mask)

        return {
            "l1": l1_loss_value,
            "grad": grad_loss
        }

    def get_l1_loss(self, predicted_depth, gt_depth):
        """Get only the L1 loss component."""
        valid_mask = ((gt_depth > 0) & (gt_depth < 10)).float()
        return self.l1_loss(predicted_depth * valid_mask, gt_depth * valid_mask)
    
    def get_silog_loss(self, predicted_depth, gt_depth):
        """Get only the SILog loss component."""
        valid_mask = ((gt_depth > 0) & (gt_depth < 10)).float()
        return self.silog_loss(predicted_depth * valid_mask, gt_depth * valid_mask)
    
    def get_l1_silog_loss(self, predicted_depth, gt_depth):
        """Get the combined L1 and SILog loss component."""
        l1 = self.get_l1_loss(predicted_depth, gt_depth)
        silog = self.get_silog_loss(predicted_depth, gt_depth)
        return l1 + 0.5 * silog
    
    def get_grad_loss(self, predicted_depth, gt_depth):
        """Get only the gradient loss component."""
        valid_mask = ((gt_depth > 0) & (gt_depth < 10)).float()
        return self.depth_gradient_loss(predicted_depth, gt_depth, valid_mask)
    
    def get_normal_loss(self, predicted_depth, gt_depth):
        """Get only the normal consistency loss component."""
        valid_mask = ((gt_depth > 0) & (gt_depth < 10)).float()
        return self.normal_consistency_loss(predicted_depth, gt_depth, valid_mask)

    def compute_normals(self, depth_map):
        """Approximates surface normals from per-ray depth."""
        assert depth_map.dim() == 1, "Depth must be 1D (per-ray)."

        dzdx = depth_map[:-1] - depth_map[1:]  # Finite difference

        # 🔥 Manual Padding Instead of F.pad()
        dzdx = torch.cat([dzdx, dzdx[-1:]], dim=0)  # Repeat last value

        normals = torch.cat([-dzdx.unsqueeze(1), torch.ones_like(dzdx).unsqueeze(1)], dim=1)
        normals = F.normalize(normals, dim=1)

        return normals


    def normal_consistency_loss(self, predicted_depth, gt_depth, valid_mask):
        """Enforces normal similarity between predicted and ground truth depth."""

        # 🔥 Compute normals
        pred_normals = self.compute_normals(predicted_depth)
        gt_normals = self.compute_normals(gt_depth)

        # 🔥 Ensure valid_mask shape matches normals
        valid_mask = valid_mask.unsqueeze(1).expand_as(pred_normals)

        return F.l1_loss(pred_normals * valid_mask, gt_normals * valid_mask)

    def depth_gradient_loss(self, predicted_depth, gt_depth, valid_mask):
        """Computes gradient loss for per-ray depth."""
        pred_dx = predicted_depth[:-1] - predicted_depth[1:]
        gt_dx = gt_depth[:-1] - gt_depth[1:]

        grad_loss_x = F.l1_loss(pred_dx * valid_mask[:-1], gt_dx * valid_mask[:-1])
        return grad_loss_x

class NeRFSemanticsLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, inputs, targets, key):
        loss_coarse = torch.zeros([1], device=targets.device)
        loss_fine = torch.zeros([1], device=targets.device)
        if f'{key}_coarse' in inputs:
            loss_coarse = self.loss(inputs[f'{key}_coarse'], targets)
        if f'{key}_fine' in inputs:
            loss_fine = self.loss(inputs[f'{key}_fine'], targets)
        return loss_coarse, loss_fine


class TVLoss(nn.Module):

    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.size_tensor(x[:, :, 1:, :]) + 1e-4
        count_w = self.size_tensor(x[:, :, :, 1:]) + 1e-4
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def size_tensor(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def get_semantic_weights(reweight_classes, fg_classes, num_semantic_classes):
    weights = torch.ones([num_semantic_classes]).float()
    if reweight_classes:
        weights[fg_classes] = 2
    return weights


class InstanceMSELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, rgb_all_instance, targets, instances):
        targets = targets.unsqueeze(1).expand(-1, rgb_all_instance.shape[-1], -1)
        instances = instances.unsqueeze(-1).expand(-1, rgb_all_instance.shape[-1])
        instance_values = torch.tensor(list(range(rgb_all_instance.shape[-1]))).to(instances.device).unsqueeze(0).expand(instances.shape[0], -1)
        instance_mask = instances == instance_values
        loss = self.loss(rgb_all_instance.permute((0, 2, 1)).reshape(-1, 3)[instance_mask.view(-1), :], targets.reshape(-1, 3)[instance_mask.view(-1), :])
        return loss


class MaskedNLLLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = torch.nn.NLLLoss(reduction='mean')

    def forward(self, output_instances, instances, semantics, invalid_class):
        if invalid_class is None:
            return self.loss(output_instances, instances)
        mask = semantics != invalid_class
        return self.loss(output_instances[mask, :], instances[mask])


class SCELoss(torch.nn.Module):

    def __init__(self, alpha, beta, class_weights):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.class_weights = class_weights
        self.cross_entropy = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='none')

    def forward(self, pred, labels_probabilities):
        # CCE
        ce = self.cross_entropy(pred, labels_probabilities)

        # RCE
        weights = torch.tensor(self.class_weights, device=pred.device).unsqueeze(0)
        pred = F.softmax(pred * weights, dim=1)
        pred = torch.clamp(pred, min=1e-8, max=1.0)
        label_clipped = torch.clamp(labels_probabilities, min=1e-8, max=1.0)

        rce = torch.sum(-1 * (pred * torch.log(label_clipped) * weights), dim=1)

        # Loss
        loss = self.alpha * ce + self.beta * rce
        return loss 