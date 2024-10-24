import matplotlib.pyplot as plt
import torch
import scipy.io.wavfile as wavfile
import os
import wandb

def min_max_summary(name, tensor, total_steps):
    if wandb.run is not None:
        wandb.log({name + '_min': tensor.min().detach().cpu().numpy(),
                   name + '_max': tensor.max().detach().cpu().numpy()})

def audio_summary(
    model_input,
    ground_truth,
    model_output,
    total_steps,
    prefix='train_'
):
    if prefix == 'train_':
        prefix = f'train_{total_steps}_'

    gt_func = torch.squeeze(ground_truth['func'])
    gt_rate = torch.squeeze(ground_truth['rate']).detach().cpu().numpy()
    pred_func = torch.squeeze(model_output['model_out'])
    coords = torch.squeeze(model_output['model_in'].clone()).detach().cpu().numpy()

    fig, axes = plt.subplots(3,1)

    strt_plot, fin_plot = int(0.05*len(coords)), int(0.95*len(coords))
    coords = coords[strt_plot:fin_plot]
    gt_func_plot = gt_func.detach().cpu().numpy()[strt_plot:fin_plot]
    pred_func_plot = pred_func.detach().cpu().numpy()[strt_plot:fin_plot]

    axes[1].plot(coords, pred_func_plot)
    axes[0].plot(coords, gt_func_plot)
    axes[2].plot(coords, gt_func_plot - pred_func_plot)

    axes[0].get_xaxis().set_visible(False)
    axes[1].axes.get_xaxis().set_visible(False)
    axes[2].axes.get_xaxis().set_visible(False)

    if wandb.run is not None:
        wandb.log({prefix + 'gt_vs_pred': wandb.Image(fig)})
    else:
        plt.savefig(f"./out/{prefix}gt_vs_pred.png")

    min_max_summary(prefix + 'coords', model_input['coords'], total_steps)
    min_max_summary(prefix + 'pred_func', pred_func, total_steps)
    min_max_summary(prefix + 'gt_func', gt_func, total_steps)

    # write audio files:
    if wandb.run is not None:
        wavfile.write(os.path.join(wandb.run.dir, 'gt.wav'), gt_rate, gt_func.detach().cpu().numpy())
        wavfile.write(os.path.join(wandb.run.dir, 'pred.wav'), gt_rate, pred_func.detach().cpu().numpy())
        wandb.log({prefix + 'gt_audio': wandb.Audio(os.path.join(wandb.run.dir, 'gt.wav'), caption='Ground Truth'),
                     prefix + 'pred_audio': wandb.Audio(os.path.join(wandb.run.dir, 'pred.wav'), caption='Prediction')})
    else:
        wavfile.write(f"./out/{prefix}gt.wav", gt_rate, gt_func.detach().cpu().numpy())
        wavfile.write(f"./out/{prefix}pred.wav", gt_rate, pred_func.detach().cpu().numpy())