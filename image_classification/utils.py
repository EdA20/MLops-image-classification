import importlib
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 15})
data_dir = Path(__file__).resolve().parent.parent / "data"


# Source: https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
def instantiate(obj_path: str, default_obj_path: str = "") -> Any:
    """
    Extract an object from a given path.
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")
    return getattr(module_obj, obj_name)


def plot_losses(
    train_losses,
    test_losses,
    train_accuracies,
    test_accuracies,
    train_precision,
    test_precision,
):
    fig, axs = plt.subplots(3, figsize=(13, 20))
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label="train")
    axs[0].plot(range(1, len(test_losses) + 1), test_losses, label="test")
    axs[0].set_ylabel("loss")

    axs[1].plot(range(1, len(train_accuracies) + 1), train_accuracies, label="train")
    axs[1].plot(range(1, len(test_accuracies) + 1), test_accuracies, label="test")
    axs[1].set_ylabel("accuracy")

    axs[2].plot(range(1, len(train_precision) + 1), train_precision, label="train")
    axs[2].plot(range(1, len(test_precision) + 1), test_precision, label="test")
    axs[2].set_ylabel("precision")

    for ax in axs:
        ax.set_xlabel("epoch")
        ax.legend()

    file_path = data_dir / "loss.png"
    fig.savefig(file_path)

    return fig, file_path
