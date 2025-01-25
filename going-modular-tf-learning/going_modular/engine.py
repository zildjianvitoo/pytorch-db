import torch
from torch import nn
from torchmetrics import Metric
from tqdm.auto import tqdm

from torch.utils.tensorboard import SummaryWriter


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

    progress_bar = tqdm(data_loader, desc="Training", leave=False)

    for X, y in progress_bar:
        X, y = X.to(device), y.to(device)
        y_logits = model(X)
        y_preds = torch.softmax(y_logits, dim=1)

        loss = loss_fn(y_logits, y)
        train_loss += loss
        acc_metric.update(y_preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix({"Batch Loss": loss.item()})

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

    progress_bar = tqdm(data_loader, desc="Testing", leave=False)

    with torch.inference_mode():
        for X, y in progress_bar:
            X, y = X.to(device), y.to(device)
            y_logits = model(X)
            y_preds = torch.softmax(y_logits, dim=1)

            loss = loss_fn(y_logits, y)
            test_loss += loss
            acc_metric.update(y_preds, y)

            progress_bar.set_postfix({"Batch Loss": loss.item()})

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
    writer: SummaryWriter,
    device,
    epochs=5,
):
    model.to(device)
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    writer.add_graph(
        model=model, input_to_model=torch.randn(32, 3, 224, 224).to(device)
    )

    for epoch in range(epochs):
        print(f"\nEpoch: {epoch}")
        train_loss, train_acc = train_step(
            model, train_dataloader, loss_fn, acc_metric, optimizer, device
        )
        test_loss, test_acc = test_step(
            model, test_dataloader, loss_fn, acc_metric, device
        )

        writer.add_scalars(
            main_tag="Loss",
            tag_scalar_dict={"train_loss": train_loss, "test_loss": test_loss},
            global_step=epoch,
        )
        writer.add_scalars(
            main_tag="Accuracy",
            tag_scalar_dict={"train_acc": train_acc, "test_acc": test_acc},
            global_step=epoch,
        )

        results["train_loss"].append(train_loss.item())
        results["train_acc"].append(train_acc.item())
        results["test_loss"].append(test_loss.item())
        results["test_acc"].append(test_acc.item())

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    writer.close()
    return results
