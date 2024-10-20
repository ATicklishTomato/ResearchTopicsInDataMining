import os

import wandb
from matplotlib import pyplot as plt

from data.metrics import peak_signal_to_noise_ratio


def audio_summary(ground_truth, predicted_audio, total_steps):
    # Calculate the PSNR between the ground truth and predicted audio
    psnr = peak_signal_to_noise_ratio(ground_truth, predicted_audio)

    # Log the PSNR to W&B
    if wandb.run is not None:
        wandb.log({'psnr': psnr, 'total_steps': total_steps})

    ground_truth = ground_truth['func']
    predicted_audio = predicted_audio['model_out']

    # Graph the ground truth and predicted audio, and store it in the W&B directory
    plt.figure(figsize=(12, 6))
    plt.plot(ground_truth.squeeze(0).detach().cpu().numpy(), label='Ground Truth')
    plt.plot(predicted_audio.squeeze(0).detach().cpu().numpy(), label='Predicted Audio')
    plt.legend()
    plt.title('Ground Truth vs Predicted Audio')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    if wandb.run is not None:
        plt.savefig(os.path.join(wandb.run.dir, 'gt_vs_pred_epoch_%04d.png' % total_steps))
        wandb.log({'gt_vs_pred_epoch_%04d.png' % total_steps: wandb.Image(plt)})
    else:
        plt.savefig('./out/gt_vs_pred_epoch_%04d.png' % total_steps)
    plt.close()