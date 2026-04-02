"""Utility functions for RNNs"""

# pylint: disable=unused-variable, invalid-name, too-many-locals
import math
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

font = {"size": 12}
matplotlib.rc("font", **font)

def plot_pred(data, scaler=None, rmse=True, plotmarkers=False, show=True, **kw):
    """Plot prediction and original data"""
    ticks = kw.pop("ticks", None)
    labels = kw.pop("labels", None)
    fig, ax = plt.subplots(**kw)
    legend = []
    markers = ["*", "x", "o"]
    colors = ["black", "steelblue", "darkred", "green"]
    x = []
    y = []
    shift = 0
    for k, v in data.items():
        if v is None:
            continue
        Ypred, Y, Y_indices = v
        X = np.arange(len(Y)) + shift
        shift = shift + len(X)
        if scaler is not None:
            Y = scaler.inverse_transform(Y.reshape(-1, 1)).flatten()
            Ypred = scaler.inverse_transform(Ypred.reshape(-1, 1)).flatten()
        e = math.sqrt(mean_squared_error(Y[Y_indices], Ypred))
        if rmse:
            k = f"{k} (RMSE: {e:.4f})"
        legend.append(k)
        col = colors.pop()
        ax.plot(X[Y_indices], Ypred, color=col)
        if plotmarkers:
            ax.plot(X[Y_indices], Ypred, markers.pop(), color=col)
        x.extend(X)
        y.extend(Y)
    legend.append("Data")
    ax.plot(x, y, "-", color=colors.pop())
    ax.set_title("Model prediction")
    if ticks is not None and labels is not None:
        ax.set_xticks(ticks, labels=labels)
    ax.legend(legend)
    if show:
        plt.show()


def plot_loss(metrics, fig_size=(12, 6)):
    """Plot loss metrics"""
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(metrics["loss"], label="train loss", color="red", alpha=0.8)
    ax.plot(metrics["val_loss"], label="val loss", color="orange", alpha=0.8)
    ax.set_title("model loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    plt.show()

        
def plot_loss_acc(metrics, fig_width=12, fig_height=6):
    """Plot loss and accuracy metrics"""
    accuracy = "accuracy"
    val_accuracy = "val_accuracy"
    if accuracy not in metrics:
        accuracy = "acc"
        val_accuracy = "val_acc"

    fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))
    ax2 = ax1.twinx()

    ax1.plot(metrics["loss"], label="train loss", color="red", alpha=0.8)
    ax1.plot(metrics["val_loss"], label="val loss", color="orange", alpha=0.8)
    ax1.set_ylabel("loss")
    ax1.set_title("model accuracy and loss")
    ax1.set_xlabel("epoch")

    ax2.plot([x*100.0 for x in metrics[accuracy]], label="train acc", color="blue", alpha=0.8)
    ax2.plot([x*100.0 for x in metrics[val_accuracy]], label="val acc", color="green", alpha=0.8)
    ax2.set_ylabel("accuracy (%)")
    ax2.set_ylim(0, 100)

    ax1.legend(["train loss", "val loss"], loc="upper left")
    ax2.legend(["train acc", "val acc"], loc="upper right")
    plt.show()


def plot_history(history, show=True, xlim=None, ylim=None, **kw):
    """Plot history - plot training and/or test accuracy or loss values"""
    datalabels = ["Training", "Validation"]
    metrics_labels = {
        "loss": "loss",
        "acc": "accuracy",
        "accuracy": "accuracy",
        "mse": "mse",
        "recall": "recall",
    }
    if not isinstance(history, dict):
        history = history.history
    hkeys = history.keys()
    h = np.array([history[k] for k in hkeys])
    labels = [
        f"{x} {y}"
        for x, y in zip(
            [datalabels[u.startswith("val_")] for u in hkeys],
            [metrics_labels[v.replace("val_", "")] for v in hkeys],
        )
    ]
    fig, ax = plt.subplots(**kw)
    ax.plot(np.array(range(0, h.shape[1])), h.T)
    ax.set_title("Model metrics")
    ax.set_ylabel("Metric")
    ax.set_xlabel("Epoch")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(list(labels), loc="upper left")
    if show:
        plt.show()
