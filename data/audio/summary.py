import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import scipy.io.wavfile as wavfile
import os

def min_max_summary(name, tensor, writer, total_steps):
    writer.add_scalar(name + '_min', tensor.min().detach().cpu().numpy(), total_steps)
    writer.add_scalar(name + '_max', tensor.max().detach().cpu().numpy(), total_steps)

def summary(
    logging_root_path,
    model_input,
    ground_truth,
    model_output, 
    writer: SummaryWriter, 
    total_steps, 
    prefix='train_'
):
    gt_func = torch.squeeze(ground_truth['func'])
    gt_rate = torch.squeeze(ground_truth['rate']).detach().cpu().numpy()
    gt_scale = torch.squeeze(ground_truth['scale']).detach().cpu().numpy()
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

    writer.add_figure(prefix + 'gt_vs_pred', fig, global_step=total_steps)

    min_max_summary(prefix + 'coords', model_input['coords'], writer, total_steps)
    min_max_summary(prefix + 'pred_func', pred_func, writer, total_steps)
    min_max_summary(prefix + 'gt_func', gt_func, writer, total_steps)

    # write audio files:
    wavfile.write(os.path.join(logging_root_path, 'gt.wav'), gt_rate, gt_func.detach().cpu().numpy())
    wavfile.write(os.path.join(logging_root_path, 'pred.wav'), gt_rate, pred_func.detach().cpu().numpy())