import logging
import torch

logger = logging.getLogger(__name__)

def train(model, train_dataloader, val_dataloader, epochs, lr, device, log_level):
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

        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for x, y in val_dataloader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                _, predicted = torch.max(y_pred.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        accuracy = correct / total
        logger.info(f"Epoch {epoch + 1}, Accuracy: {accuracy}")

    logger.info(f"Training complete")

    return model