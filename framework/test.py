import logging
import torch
import wandb
from matplotlib import pyplot as plt

from data.images import utils, metrics

logger = logging.getLogger(__name__)

def test(model,
         test_dataloader,
         model_dir: str,
         config: dict,
         device,
         log_level,
         use_wandb=False):
    logger.setLevel(log_level)
    logger.info(f"Testing {model.__class__.__name__}")

    model.to(device)
    model.eval()

    with torch.no_grad():
        for (model_input, ground_truth) in test_dataloader:
            model_input = {key: value.to(device) for key, value in model_input.items()}
            ground_truth = {key: value.to(device) for key, value in ground_truth.items()}
            model_output = model(model_input)

            losses = config["loss_fn"](model_output, ground_truth)

            if use_wandb:
                wandb.log({'total_loss': sum(losses.values()),
                           "avg_loss": config["loss_fn"](model_output, ground_truth)['img_loss'],
                           'psnr': metrics.peak_signal_to_noise_ratio(model_output, ground_truth)
                           })

            if config["resolution"] is not None:
                # Get the numpy arrays for the images
                ground_truth = utils.lin2img(ground_truth['img'], config["resolution"])
                predicted = utils.lin2img(model_output['model_out'], config["resolution"])

                # Normalize the image values to [0, 1]
                ground_truth = utils.rescale_img(ground_truth, mode='scale').detach().cpu().numpy()
                predicted = utils.rescale_img(predicted, mode='scale').detach().cpu().numpy()

                # Images are in the format (3, height, width) because 3 colours
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(ground_truth[0].transpose(1, 2, 0))
                ax[0].set_title("Ground Truth")
                ax[0].axis('off')
                ax[1].imshow(predicted[0].transpose(1, 2, 0))
                ax[1].set_title("Predicted")
                ax[1].axis('off')

                if use_wandb:
                    wandb.log({
                        "test_image": wandb.Image(fig)
                    })
                else:
                    plt.savefig(f"{model_dir}/test_image.png")
                plt.close(fig)

    logger.info("Testing complete")
