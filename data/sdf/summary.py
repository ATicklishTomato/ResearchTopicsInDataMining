import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt

from data.metrics import chamfer_hausdorff_distance, intersection_over_union
from data.utils import get_mgrid, lin2img

def make_contour_plot(array_2d,mode='log'):
    fig, ax = plt.subplots(figsize=(2.75, 2.75), dpi=300)

    levels = None
    colors = None

    if mode== 'log':
        num_levels = 6
        levels_pos = np.logspace(-2, 0, num=num_levels) # logspace
        levels_neg = -1. * levels_pos[::-1]
        levels = np.concatenate((levels_neg, np.zeros((0)), levels_pos), axis=0)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels*2+1))
    elif mode== 'lin':
        num_levels = 10
        levels = np.linspace(-.5,.5,num=num_levels)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels))

    sample = np.flipud(array_2d)
    CS = ax.contourf(sample, levels=levels, colors=colors)
    cbar = fig.colorbar(CS)

    ax.contour(sample, levels=levels, colors='k', linewidths=0.1)
    ax.contour(sample, levels=[0], colors='k', linewidths=0.3)
    ax.axis('off')
    return fig



def sdf_summary(model, ground_truth, predicted_distance, total_steps):
    # Calculate the PSNR between the ground truth and predicted distance
    iou = intersection_over_union(predicted_distance, ground_truth)
    chamfer, hausdorff, _, _, _, _ = chamfer_hausdorff_distance(ground_truth["sdf"], predicted_distance["model_out"])

    if wandb.run is not None:
        wandb.log({'iou': iou, 'chamfer': chamfer, 'hausdorff': hausdorff, 'total_steps': total_steps})

    slice_coords_2d = get_mgrid(512)

    yz_slice_coords = torch.cat((torch.zeros_like(slice_coords_2d[:, :1]), slice_coords_2d), dim=-1)
    yz_slice_model_input = {'coords': yz_slice_coords.cuda()[None, ...]}

    yz_model_out = model(yz_slice_model_input)
    sdf_values = yz_model_out['model_out']
    sdf_values = lin2img(sdf_values).squeeze().cpu().numpy()
    fig = make_contour_plot(sdf_values)

    if wandb.run is not None:
        wandb.log({model.__class__.__name__ + "_" + total_steps + "_" + 'yz_sdf_slice': wandb.Image(fig)})

    xz_slice_coords = torch.cat((slice_coords_2d[:, :1],
                                 torch.zeros_like(slice_coords_2d[:, :1]),
                                 slice_coords_2d[:, -1:]), dim=-1)
    xz_slice_model_input = {'coords': xz_slice_coords.cuda()[None, ...]}

    xz_model_out = model(xz_slice_model_input)
    sdf_values = xz_model_out['model_out']
    sdf_values = lin2img(sdf_values).squeeze().cpu().numpy()
    fig = make_contour_plot(sdf_values)

    if wandb.run is not None:
        wandb.log({model.__class__.__name__ + "_" + total_steps + "_" + 'xz_sdf_slice': wandb.Image(fig)})

    xy_slice_coords = torch.cat((slice_coords_2d[:, :2],
                                 -0.75 * torch.ones_like(slice_coords_2d[:, :1])), dim=-1)
    xy_slice_model_input = {'coords': xy_slice_coords.cuda()[None, ...]}

    xy_model_out = model(xy_slice_model_input)
    sdf_values = xy_model_out['model_out']
    sdf_values = lin2img(sdf_values).squeeze().cpu().numpy()
    fig = make_contour_plot(sdf_values)

    if wandb.run is not None:
        wandb.log({model.__class__.__name__ + "_" + total_steps + "_" + 'xy_sdf_slice': wandb.Image(fig)})