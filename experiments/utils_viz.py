from matplotlib.ticker import ScalarFormatter
from mpltools import special

import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns


def plot_hinton(
    W, method=None, _max_value=None, xticklabels=None, yticklabels=None, path=None
):
    """
    Draws a Hinton diagram for visualizing a weight matrix.
    Temporarily disables matplotlib interactive mode if it is on,
    otherwise this takes forever.
    """

    reenable = False
    if plt.isinteractive():
        plt.ioff()
    plt.clf()

    if reenable:
        plt.ion()

    special.hinton(W, max_value=_max_value)
    if xticklabels is not None:
        plt.xticks(np.arange(len(xticklabels)), xticklabels, rotation=90, fontsize=13)
    if yticklabels is not None:
        plt.yticks(np.arange(len(xticklabels)), yticklabels, fontsize=13)

    plt.xlabel("true type", fontsize=17)
    plt.ylabel("predicted type", fontsize=17)

    if path is not None:
        plt.savefig(path, dpi=1000, bbox_inches="tight")
    else:
        plt.show()


def plot_hintons(df, path, methods):
    if not (os.path.exists(path)):
        os.mkdir(path)
    type_indices = {
        "currency": 0,
        "data storage": 1,
        "length": 2,
        "mass": 3,
        "volume": 4,
        "no unit": 5,
    }
    xticklabels = [
        item[0] for item in sorted(type_indices.items(), key=lambda kv: (kv[1], kv[0]))
    ]
    true_values = df["annotation"].tolist()
    for method in methods:
        confusion_matrix = np.zeros((len(type_indices), len(type_indices)), dtype=float)
        predicted_values = df[method].tolist()
        for true_value, predicted_value in zip(true_values, predicted_values):
            true_index = type_indices[true_value]
            if true_value in type_indices and predicted_value in type_indices:
                predicted_index = type_indices[predicted_value]
            elif predicted_value in ["no unit", "unknown"]:
                predicted_index = type_indices["no unit"]

            confusion_matrix[predicted_index, true_index] += 1

        sum_confusion_matrix = confusion_matrix.sum(axis=0)
        indices = np.where(sum_confusion_matrix != 0)[0]
        normalized_confusion_matrix = confusion_matrix.copy()
        normalized_confusion_matrix[:, indices] = (
            confusion_matrix[:, indices] / sum_confusion_matrix[indices]
        )
        plot_hinton(
            normalized_confusion_matrix,
            method=method,
            xticklabels=xticklabels,
            yticklabels=xticklabels,
            path=path + "figures/hinton_" + method + ".png",
        )


def plot_runtimes(df_stack, output_path):
    fig = plt.figure()
    ax = sns.violinplot(x="Method", y="Runtime (sec.)", data=df_stack, color="white")
    yticks = ax.get_yticks()
    # note that this may mess up things when yticks are close to each other (e.g., -0.5  0.   0.5)
    # by mapping them to the same integer (0 in this case)
    yticks = [int(ytick) for ytick in yticks]
    ax.set_yticks(ax.get_yticks())  # just get and reset whatever you already have
    ax.set_yticklabels(yticks)

    labels = []
    for label in ax.get_yticklabels():
        pwr = label.get_text()
        labels.append(r"$10^{{{:}}}$".format(pwr))
    ax.set_yticks(ax.get_yticks())  # just get and reset whatever you already have
    ax.set_yticklabels(labels)
    plt.ylabel("Runtime (s)", fontsize=13)
    plt.xlabel("Method", fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=11)
    plt.savefig(output_path + "figures/runtimes.png", dpi=1000)
