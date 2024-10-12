import framework.utils as utils
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil
import logging

logger = logging.getLogger(__name__)

def train(
    model,
    train_dataloader,
    epochs,
    lr,
    model_dir: str,
    config: dict,
    device,
    log_level
):
    # Set up logging, checkpoints and summaries
    logger.setLevel(log_level)
    logger.info(f"Training {model.__class__.__name__}")
    if os.path.exists(model_dir):
        val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
        if val == 'y':
            shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)
    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)
    writer = SummaryWriter(summaries_dir)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            if not epoch % 25 and epoch:
                # Make a model checkpoint.
                torch.save(
                    model.state_dict(),
                    os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch)
                )
                np.savetxt(
                    os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                    np.array(train_losses)
                )

            for step, (model_input, ground_truth) in enumerate(train_dataloader):
                start_time = time.time()
            
                model_input = {key: value.to(device) for key, value in model_input.items()}
                ground_truth = {key: value.to(device) for key, value in ground_truth.items()}

                model_output = model(model_input)
                losses = config["loss_fn"](model_output, ground_truth)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                train_losses.append(train_loss.item())
                writer.add_scalar("total_train_loss", train_loss, total_steps)

                if not total_steps % 500:
                    # Make a model summary
                    torch.save(
                        model.state_dict(),
                        os.path.join(checkpoints_dir, 'model_current.pth')
                    )
                    config["summary_fn"](ground_truth, model_output, writer, total_steps)

                # Backpropagation
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                pbar.update(1)

                if not total_steps % 500:
                    tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

                total_steps += 1

        torch.save(
            model.state_dict(),
            os.path.join(checkpoints_dir, 'model_final.pth')
        )
        np.savetxt(
            os.path.join(checkpoints_dir, 'train_losses_final.txt'),
            np.array(train_losses)
        )