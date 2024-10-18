import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import cv2
import cmapy
import os

import data.images.utils as utils
import data.images.differential_operators as differential_operators
import data.images.metrics as metrics

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def summary(
    image_resolution,
    ground_truth,
    model_output, 
    writer: SummaryWriter, 
    total_steps, 
    prefix='train_'
):
    """
    Analyze the current model quality by creating images for the output, the gradient of the output,
    and the laplacian of the output.

    Persist this analysis using Tensorboard (see readme for more info on this).  
    """
    ground_truth_img = utils.lin2img(ground_truth['img'], image_resolution)
    predicted_img = utils.lin2img(model_output['model_out'], image_resolution)

    img_gradient = differential_operators.gradient(model_output['model_out'], model_output['model_in'])
    img_laplace = differential_operators.laplace(model_output['model_out'], model_output['model_in'])

    output_vs_gt = torch.cat((ground_truth_img, predicted_img), dim=-1)
    writer.add_image(
        prefix + 'gt_vs_pred', 
        make_grid(output_vs_gt, scale_each=False, normalize=True),
        global_step=total_steps
    )

    # Rescale and handle multiple channels for predicted image
    pred_img_vis = utils.rescale_img((predicted_img+1)/2, mode='clamp').permute(0,2,3,1).squeeze(0).detach().cpu().numpy()
    if pred_img_vis.shape[-1] > 3:  # If there are more than 3 channels, take the first 3 for visualization
        pred_img_vis = pred_img_vis[:, :, :3]

    img = utils.to_uint8(
        utils.rescale_img(
            utils.lin2img(img_laplace), 
            perc=2
        ).permute(0,2,3,1).squeeze(0).detach().cpu().numpy()
    )

    print(img.shape)

    pred_grad = utils.grads2img(utils.lin2img(img_gradient)).permute(1,2,0).squeeze().detach().cpu().numpy()
    pred_lapl = cv2.cvtColor(
        cv2.applyColorMap(
            utils.to_uint8(
                utils.rescale_img(
                    utils.lin2img(img_laplace), 
                    perc=2
                ).permute(0,2,3,1).squeeze(0).detach().cpu().numpy()
            ),
            cmapy.cmap('RdBu')
        ), cv2.COLOR_BGR2RGB)

    # Rescale and handle multiple channels for ground truth image
    ground_truth_img_vis = utils.rescale_img((ground_truth_img+1) / 2, mode='clamp').permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()
    if ground_truth_img_vis.shape[-1] > 3:  # If there are more than 3 channels, take the first 3 for visualization
        ground_truth_img_vis = ground_truth_img_vis[:, :, :3]

    ground_truth_gradient = utils.grads2img(utils.lin2img(ground_truth['gradients'])).permute(1, 2, 0).squeeze().detach().cpu().numpy()
    ground_truth_laplacian = cv2.cvtColor(cv2.applyColorMap(utils.to_uint8(utils.rescale_img(
        utils.lin2img(ground_truth['laplace']), perc=2).permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()), cmapy.cmap('RdBu')), cv2.COLOR_BGR2RGB)

    # Write images to Tensorboard
    writer.add_image(prefix + 'pred_img', torch.from_numpy(pred_img_vis).permute(2, 0, 1), global_step=total_steps)
    writer.add_image(prefix + 'pred_grad', torch.from_numpy(pred_grad).permute(2, 0, 1), global_step=total_steps)
    writer.add_image(prefix + 'pred_lapl', torch.from_numpy(pred_lapl).permute(2,0,1), global_step=total_steps)
    writer.add_image(prefix + 'gt_img', torch.from_numpy(ground_truth_img_vis).permute(2,0,1), global_step=total_steps)
    writer.add_image(prefix + 'gt_grad', torch.from_numpy(ground_truth_gradient).permute(2, 0, 1), global_step=total_steps)
    writer.add_image(prefix + 'gt_lapl', torch.from_numpy(ground_truth_laplacian).permute(2, 0, 1), global_step=total_steps)

    write_metrics(
        utils.lin2img(model_output['model_out'], image_resolution),
        utils.lin2img(ground_truth['img'], image_resolution), 
        writer, 
        total_steps, 
        prefix+'img_'
    )


def write_metrics(pred_img, gt_img, writer, iter, prefix):
    """
    Compute peak signal to noise ratio and structural similarity for each channel and write to tensorboard.
    """
    batch_size = pred_img.shape[0]
    channels = pred_img.shape[1]


    pred_img = pred_img.detach().cpu().numpy()
    gt_img = gt_img.detach().cpu().numpy()

    psnrs, ssims = list(), list()
    
    for i in range(batch_size):
        psnr_channel, ssim_channel = list(), list()

        for ch in range(channels):
            p = pred_img[i, ch].transpose(0, 1)
            trgt = gt_img[i, ch].transpose(0, 1)

            # Normalize the images to [0, 1] range for SSIM and PSNR computation
            p = (p / 2.) + 0.5
            p = np.clip(p, a_min=0., a_max=1.)

            trgt = (trgt / 2.) + 0.5

            ssim = metrics.structural_similarity(im1=p, im2=trgt, data_range=1)
            psnr = metrics.peak_signal_noise_ratio(p, trgt, data_range=1)

            ssim_channel.append(ssim)
            psnr_channel.append(psnr)

        # Average PSNR and SSIM over all channels
        psnrs.append(np.mean(psnr_channel))
        ssims.append(np.mean(ssim_channel))

    writer.add_scalar(prefix + "psnr", np.mean(psnrs), iter)
    writer.add_scalar(prefix + "ssim", np.mean(ssims), iter)