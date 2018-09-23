import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_side_by_side(images: list, cmap: str = 'gray', figsize: tuple = None, labels: list=[]) -> None:
    """Pass a list of images to display them side by side"""
    fig, axes = plt.subplots(ncols=len(images), nrows=1)

    if figsize:
        fig.set_size_inches(*figsize)

    for i, im in enumerate(images):
        axes[i].imshow(im, cmap=cmap)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    if len(labels) > 0:
        for i, label in enumerate(labels):
            axes[i].set_title(label)

    plt.tight_layout()


def load_image_as_array(path: str) -> np.ndarray:
    """Load image from disk into numpy array"""
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return img