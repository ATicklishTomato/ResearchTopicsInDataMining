import logging
import torch

logger = logging.getLogger(__name__)

def test(model, test_dataloader, device, log_level):
    logger.setLevel(log_level)
    logger.info(f"Testing {model.__class__.__name__}")

    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    accuracy = correct / total
    logger.info(f"Accuracy: {accuracy}")

    return accuracy