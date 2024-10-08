import logging
import torch
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)

def test(model, data, test_dataloader, device, log_level):
    logger.setLevel(log_level)
    logger.info(f"Testing {model.__class__.__name__}")

    model.to(device)
    model.eval()

    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)

            if data == "images":
                logger.debug("Plotting images")
                fig, axes = plt.subplots(1, 2)
                axes[0].imshow(y_pred.view(256, 256).cpu().numpy())
                axes[0].set_title("Prediction")
                axes[0].axis("off")
                axes[1].imshow(y.view(256, 256).cpu().numpy())
                axes[1].set_title("Ground truth")
                axes[1].axis("off")
                plt.show()
            else:
                logger.debug("No known data type. Fallback calculating accuracy")
                accuracy = (y_pred.argmax(dim=1) == y).float().mean()
                logger.info(f"Accuracy: {accuracy.item()}")
