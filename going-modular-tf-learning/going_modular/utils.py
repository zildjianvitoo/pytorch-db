import torch
from torch import nn
from pathlib import Path

import matplotlib.pyplot as plt


def save_model(model: nn.Module, target_dir: str, model_name: str):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith("pth") or model_name.endswith(
        "pt"
    ), "Ekstensi model harus .pt atau .pth"
    model_save_path = target_dir_path / model_name

    torch.save(obj=model.state_dict(), f=model_save_path)


def plot_results(results):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    ax[0].set_title("Accuracy")
    ax[0].plot(
        range(len(results["train_acc"])), results["train_acc"], label="Train acc"
    )
    ax[0].plot(range(len(results["test_acc"])), results["test_acc"], label="Test acc")
    ax[0].legend()

    ax[1].set_title("Loss")
    ax[1].plot(
        range(len(results["train_loss"])), results["train_loss"], label="Train loss"
    )
    ax[1].plot(
        range(len(results["test_loss"])), results["test_loss"], label="Test loss"
    )
    ax[1].legend()
