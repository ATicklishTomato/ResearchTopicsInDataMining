import numpy as np
import torch
from torchvision.utils import make_grid
import cv2
import cmapy
import os
import wandb
from matplotlib import pyplot as plt

import data.utils as utils
import data.metrics as metrics
from data.images import differential_operators


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def make_image(tensor, filename, squeeze=True, permute=None):
    if squeeze:
        tensor = tensor.squeeze()
    if permute is not None:
        tensor = tensor.permute(permute)
    plt.imshow(tensor.permute(1,2,0).detach().cpu().numpy())
    plt.axis('off')
    if wandb.run is not None:
        plt.savefig(os.path.join(wandb.run.dir, filename))
        wandb.log({filename: wandb.Image(plt)})
    else:
        plt.savefig('./out/' + filename)


def image_summary(
    image_resolution,
    ground_truth,
    model_output,
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

    # Make the image to save it to W&B
    make_image(make_grid(output_vs_gt, scale_each=False, normalize=True), 'gt_vs_pred_epoch_%04d.png' % total_steps,
               squeeze=False, permute=None)

    # Rescale and handle multiple channels for predicted image
    pred_img_vis = utils.rescale_img((predicted_img + 1) / 2, mode='clamp').permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()
    if pred_img_vis.shape[-1] > 3:  # If there are more than 3 channels, take the first 3 for visualization
        pred_img_vis = pred_img_vis[:, :, :3]

    img = utils.to_uint8(
        utils.rescale_img(
            utils.lin2img(img_laplace), 
            perc=2
        ).permute(0,2,3,1).squeeze(0).detach().cpu().numpy()
    )

    print(img.shape)

    pred_grad = utils.grads2img(utils.lin2img(img_gradient)).permute(1, 2, 0).squeeze().detach().cpu().numpy()
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
    ground_truth_img_vis = utils.rescale_img((ground_truth_img + 1) / 2, mode='clamp').permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()
    if ground_truth_img_vis.shape[-1] > 3:  # If there are more than 3 channels, take the first 3 for visualization
        ground_truth_img_vis = ground_truth_img_vis[:, :, :3]

    ground_truth_gradient = utils.grads2img(utils.lin2img(ground_truth['gradients'])).permute(1, 2, 0).squeeze().detach().cpu().numpy()
    ground_truth_laplacian = cv2.cvtColor(cv2.applyColorMap(utils.to_uint8(utils.rescale_img(
        utils.lin2img(ground_truth['laplace']), perc=2).permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()), cmapy.cmap('RdBu')), cv2.COLOR_BGR2RGB)

    # Create images to save to W&B
    make_image(torch.from_numpy(pred_img_vis), 'pred_img_epoch_%04d.png' % total_steps,
               squeeze=False, permute=(2, 0, 1))
    make_image(torch.from_numpy(pred_grad), 'pred_grad_epoch_%04d.png' % total_steps,
               squeeze=False, permute=(2, 0, 1))
    make_image(torch.from_numpy(pred_lapl), 'pred_lapl_epoch_%04d.png' % total_steps,
               squeeze=False, permute=(2, 0, 1))
    make_image(torch.from_numpy(ground_truth_img_vis), 'gt_img_epoch_%04d.png' % total_steps,
               squeeze=False, permute=(2, 0, 1))
    make_image(torch.from_numpy(ground_truth_gradient), 'gt_grad_epoch_%04d.png' % total_steps,
               squeeze=False, permute=(2, 0, 1))
    make_image(torch.from_numpy(ground_truth_laplacian), 'gt_lapl_epoch_%04d.png' % total_steps,
               squeeze=False, permute=(2, 0, 1))


    write_metrics(
        utils.lin2img(model_output['model_out'], image_resolution),
        utils.lin2img(ground_truth['img'], image_resolution),
        total_steps, 
        prefix+'img_'
    )


def write_metrics(pred_img, gt_img, iter, prefix):
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

    if wandb.run is not None:
        wandb.log({prefix + "channel_avg_psnr": np.mean(psnrs),
                   prefix + "channel_avg_ssim": np.mean(ssims),
                   'total_steps': iter})