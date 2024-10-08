from collections.abc import Iterable
import logging
import torch

logger = logging.getLogger(__name__)

def train(model, data, train_dataloader, val_dataloader, epochs, lr, device, log_level):
    logger.setLevel(log_level)
    logger.info(f"Training {model.__class__.__name__}")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        logger.debug(f"Epoch {epoch + 1}")
        model.train()
        for i, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            logger.debug(f"Batch {i + 1}, Loss: {loss.item()}")

        if isinstance(val_dataloader, Iterable) and data == "images":
            model.eval()
            with torch.no_grad():
                for x, y in val_dataloader:
                    x, y = x.to(device), y.to(device)
                    y_pred = model(x)

                    fig, axes = plt.subplots(1, 2)
                    axes[0].imshow(y_pred.view(256, 256).cpu().numpy())
                    axes[0].set_title("Prediction")
                    axes[0].axis("off")
                    axes[1].imshow(y.view(256, 256).cpu().numpy())
                    axes[1].set_title("Ground truth")
                    axes[1].axis("off")
                    plt.show()

    logger.info(f"Training complete")

    return model