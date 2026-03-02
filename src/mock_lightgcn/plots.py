import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


def plot_training_curves(
    history: dict[str, list[float]],
    save_path: str | None = None,
) -> None:
    """Plot training loss and validation metrics over epochs.

    Args:
        history: Dict with 'train_loss' and optional metric keys (e.g. 'val_precision@10').
        save_path: If provided, save figure to this path instead of showing.
    """
    train_losses = history.get("train_loss", [])
    val_metrics = {k: v for k, v in history.items() if k != "train_loss"}

    epochs = range(1, len(train_losses) + 1)

    num_plots = 1 + len(val_metrics)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 4))
    if num_plots == 1:
        axes = [axes]

    # Training loss
    axes[0].plot(epochs, train_losses, label="Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()

    # Validation metrics
    for ax, (metric_name, values) in zip(axes[1:], val_metrics.items()):
        ax.plot(range(1, len(values) + 1), values, label=metric_name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.legend()

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()
