import torch
from torch import nn
from torchmetrics import Metric


def train_step(
    model: nn.Module,
    data_loader: torch.utils.data.Dataset,
    loss_fn: nn.Module,
    acc_metric: Metric,
    optimizer: torch.optim.Optimizer,
    device,
):

    model.train()
    train_loss = 0
    acc_metric.reset(),

    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        y_logits = model(X)
        y_preds = torch.softmax(y_logits, dim=1)

        loss = loss_fn(y_logits, y)
        train_loss += loss
        acc_metric.update(y_preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc = acc_metric.compute()

    return train_loss, train_acc


def test_step(
    model: nn.Module,
    data_loader: torch.utils.data.Dataset,
    loss_fn: nn.Module,
    acc_metric: Metric,
    device,
):

    model.eval()
    test_loss = 0
    acc_metric.reset(),

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_logits = model(X)
            y_preds = torch.softmax(y_logits, dim=1)

            loss = loss_fn(y_logits, y)
            test_loss += loss
            acc_metric.update(y_preds, y)

        total_loss = test_loss / len(data_loader)
        test_acc = acc_metric.compute()

    return total_loss, test_acc


def train_model(
    model: nn.Module,
    train_dataloader: torch.utils.data.Dataset,
    test_dataloader: torch.utils.data.Dataset,
    loss_fn: nn.Module,
    acc_metric: Metric,
    optimizer: torch.optim.Optimizer,
    device,
    epochs=5,
):
    model.to(device)
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(epochs):
        print(f"\nEpoch: {epoch}")
        train_loss, train_acc = train_step(
            model, train_dataloader, loss_fn, acc_metric, optimizer, device
        )
        test_loss, test_acc = test_step(
            model, test_dataloader, loss_fn, acc_metric, device
        )

        results["train_loss"].append(train_loss.item())
        results["train_acc"].append(train_acc.item())
        results["test_loss"].append(test_loss.item())
        results["test_acc"].append(test_acc.item())

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    return results
