import os

import numpy as np
import skimage.measure
import torch
import torch.nn.functional as F
from torchmetrics import JaccardIndex
from torcheval.metrics import PeakSignalNoiseRatio
from scipy.spatial import cKDTree as KDTree


def mean_squared_error(model_output, gt):
    if 'img' in gt.keys():
        return {'loss': ((model_output['model_out'] - gt['img']) ** 2).mean()}
    elif 'func' in gt.keys():
        return {'loss': ((model_output['model_out'] - gt['func']) ** 2).mean()}

structural_similarity = skimage.metrics.structural_similarity
peak_signal_noise_ratio = skimage.metrics.peak_signal_noise_ratio

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def sdf_loss(model_output, gt):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    gradient_res = gradient(pred_sdf, coords)

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
    inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
    normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient_res, gt_normals, dim=-1)[..., None],
                                    torch.zeros_like(gradient_res[..., :1]))
    grad_constraint = torch.abs(gradient_res.norm(dim=-1) - 1)
    # Exp      # Lapl
    # -----------------
    return {'sdf': torch.abs(sdf_constraint).mean() * 3e3,  # 1e4      # 3e3
            'inter': inter_constraint.mean() * 1e2,  # 1e2                   # 1e3
            'normal_constraint': normal_constraint.mean() * 1e2,  # 1e2
            'grad_constraint': grad_constraint.mean() * 5e1}  # 1e1      # 5e1

def peak_signal_to_noise_ratio(model_output, gt):
    """Simple wrapper around skimage.metrics.peak_signal_noise_ratio.
    This function calculates the Peak Signal to Noise Ratio (PSNR) between the model output and the ground truth.
    Args:
        model_output (dict): The model output.
        gt (dict): The ground truth.
    Returns:
        psnr (float): The Peak Signal to Noise Ratio.
    """
    psnr = PeakSignalNoiseRatio()
    if 'img' in gt.keys():
        psnr.update(model_output['model_out'], gt['img'])
    elif 'func' in gt.keys():
        psnr.update(model_output['model_out'], gt['func'])
    return psnr.compute().detach().cpu().item()

def intersection_over_union(model_output, gt, n_classes=2):
    """Recommended standard approach for IoU with PyTorch.
    Can be found here: https://lightning.ai/docs/torchmetrics/stable/classification/jaccard_index.html
    This function calculates the Intersection over Union (IoU) between the model output and the ground truth.
    Args:
        model_output (dict): The model output.
        gt (dict): The ground truth.
        n_classes (int): The number of classes.
    Returns:
        iou (float): The Intersection over Union.
    """
    jaccard = JaccardIndex(num_classes=n_classes, task='multiclass')
    return jaccard(model_output['model_out'], gt['sdf']).detach().cpu().item()


def chamfer_hausdorff_distance(recon_points, gt_points, eval_type="Default"):
    """Borrowed from the StEik paper repository.
    Can be found here: https://github.com/sunyx523/StEik
    This function can be found in /surface_reconstruction/compute_metrics_shapenet.py
    This function calculates the Chamfer distance and Hausdorff distance between two point clouds.
    Args:
        recon_points (np.ndarray): The reconstructed point cloud.
        gt_points (np.ndarray): The ground truth point cloud.
        eval_type (str): The evaluation type. Can be 'DeepSDF' or 'Default'.
    Returns:
        chamfer_dist (float): The Chamfer distance.
        hausdorff_distance (float): The Hausdorff distance.
        cd_re2gt (float): The Chamfer distance from the reconstructed to the ground truth point cloud.
        cd_gt2re (float): The Chamfer distance from the ground truth to the reconstructed point cloud.
        hd_re2gt (float): The Hausdorff distance from the reconstructed to the ground truth point cloud.
        hd_gt2re (float): The Hausdorff distance from the ground truth to the reconstructed point cloud.
    """
    recon_kd_tree = KDTree(recon_points)
    gt_kd_tree = KDTree(gt_points)
    re2gt_distances, re2gt_vertex_ids = recon_kd_tree.query(gt_points, workers=4)
    gt2re_distances, gt2re_vertex_ids = gt_kd_tree.query(recon_points, workers=4)
    if eval_type == 'DeepSDF':
        cd_re2gt = np.mean(re2gt_distances ** 2)
        cd_gt2re = np.mean(gt2re_distances ** 2)
        hd_re2gt = np.max(re2gt_distances)
        hd_gt2re = np.max(gt2re_distances)
        chamfer_dist = cd_re2gt + cd_gt2re
        hausdorff_distance = np.max((hd_re2gt, hd_gt2re))
    else:
        cd_re2gt = np.mean(re2gt_distances)
        cd_gt2re = np.mean(gt2re_distances)
        hd_re2gt = np.max(re2gt_distances)
        hd_gt2re = np.max(gt2re_distances)
        chamfer_dist = 0.5 * (cd_re2gt + cd_gt2re)
        hausdorff_distance = np.max((hd_re2gt, hd_gt2re))
    return chamfer_dist, hausdorff_distance, cd_re2gt, cd_gt2re, hd_re2gt, hd_gt2re
