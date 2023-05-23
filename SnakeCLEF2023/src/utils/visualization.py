import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch


def create_fig(*, ntotal=None, ncols=1, nrows=1, colsize=8, rowsize=6, **kwargs):
    if ntotal is not None:
        nrows = int(np.ceil(ntotal / ncols))
    fig, ax_mat = plt.subplots(
        nrows, ncols, figsize=(colsize*ncols, rowsize*nrows), **kwargs)
    axs = np.array(ax_mat).flatten()
    if ntotal is not None and len(axs) > ntotal:
        for ax in axs[ntotal:]:
            ax.axis('off')
    if len(axs) == 1:
        axs = axs[0]
    return fig, axs


def plot_bbox(bbox, ax=None, c='red', linestyle='-', ms=10, **kwargs):
    if ax is None:
        fig, ax = create_fig(colsize=3, rowsize=3)

    xmin, ymin, xmax, ymax = bbox
    ax.plot(
        [xmin, xmax, xmax, xmin, xmin],
        [ymin, ymin, ymax, ymax, ymin],
        c=c, linestyle=linestyle, ms=ms, **kwargs)  # marker='.',

    return ax


def imshow(img, *, bbox=None, ax=None,
           title=None, axis_off=True, **kwargs):
    if isinstance(img, torch.Tensor):
        assert len(img.shape) == 3
        img = img.detach().cpu().numpy().transpose(1, 2, 0)

    if ax is None:
        fig, ax = create_fig(colsize=3, rowsize=3)

    ax.imshow(img, **kwargs)
    if bbox is not None:
        plot_bbox(bbox, ax=ax)
    if title is not None:
        ax.set(title=title)
    if axis_off:
        ax.axis('off')

    return ax
