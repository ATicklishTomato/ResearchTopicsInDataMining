import torch
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import logging
import wandb

from data.images import metrics

logger = logging.getLogger(__name__)

def train(
    model,
    dataloader,
    epochs,
    lr,
    config: dict,
    device,
    log_level,
    use_wandb=False
):
    # Set up logging, checkpoints and summaries
    logger.setLevel(log_level)
    logger.info(f"Training {model.__class__.__name__}")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if use_wandb:
        wandb.watch(model, log='all', log_freq=250)
        logger.info("Model watched by Weights and Biases")

    total_steps = 0
    with tqdm(total=len(dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            if not epoch % 25 and epoch and use_wandb:
                # Make a model checkpoint.
                torch.save(
                    model.state_dict(),
                    os.path.join(wandb.run.dir, 'model_epoch_%04d.pth' % epoch)
                )
                wandb.save(os.path.join(wandb.run.dir, 'model_epoch_%04d.pth' % epoch))
                np.savetxt(
                    os.path.join(wandb.run.dir, 'train_losses_epoch_%04d.txt' % epoch),
                    np.array(train_losses)
                )
                wandb.save(os.path.join(wandb.run.dir, 'train_losses_epoch_%04d.txt' % epoch))

            for step, (model_input, ground_truth) in enumerate(dataloader):
                start_time = time.time()

                model_input = {key: value.to(device) for key, value in model_input.items()}
                ground_truth = {key: value.to(device) for key, value in ground_truth.items()}

                logging.debug(f"Model input: {model_input}")
                model_output = model(model_input)
                losses = config["loss_fn"](model_output, ground_truth)

                if use_wandb:
                    wandb.log({'total_loss': sum(losses.values()),
                               "avg_loss": config["loss_fn"](model_output, ground_truth)['img_loss'],
                               'psnr': metrics.peak_signal_to_noise_ratio(model_output, ground_truth)
                               })


                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()
                    if use_wandb:
                        wandb.log({loss_name: single_loss, 'total_steps': total_steps})
                    train_loss += single_loss

                train_losses.append(train_loss.item())
                if use_wandb:
                    wandb.log({'total_loss': train_loss, 'total_steps': total_steps})

                if not total_steps % 500 and use_wandb:
                    # Make a model summary
                    torch.save(
                        model.state_dict(),
                        os.path.join(wandb.run.dir, 'model_current.pth')
                    )
                    wandb.save(os.path.join(wandb.run.dir, 'model_current.pth'))
                    config["summary_fn"](ground_truth, model_output, total_steps)
                elif not total_steps % 500:
                    torch.save(
                        model.state_dict(),
                        './out/model_current.pth'
                    )
                    config["summary_fn"](ground_truth, model_output, total_steps)

                # Backpropagation
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                pbar.update(1)

                if not total_steps % 500:
                    tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

                total_steps += 1

        if use_wandb:
            torch.save(
                model.state_dict(),
                os.path.join(wandb.run.dir, 'model_final.pth')
            )
            wandb.save(os.path.join(wandb.run.dir, 'model_final.pth'))
            np.savetxt(
                os.path.join(wandb.run.dir, 'train_losses_final.txt'),
                np.array(train_losses)
            )
            wandb.save(os.path.join(wandb.run.dir, 'train_losses_final.txt'))
        else:
            torch.save(model.state_dict(), './out/model_final.pth')
            np.savetxt('./out/train_losses_final.txt', np.array(train_losses))

        if use_wandb:
            # On Windows and this doesn't work? Go to Settings -> Update & Security -> For developers
            # and enable Developer Mode
            torch.save(
                model.state_dict(),
                os.path.join(wandb.run.dir, 'model_final.pth')
            )
            wandb.save(os.path.join(wandb.run.dir, 'model_final.pth'))

        logger.info("Training complete")