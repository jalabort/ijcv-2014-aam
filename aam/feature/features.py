import itertools
import numpy as np

from skimage.feature import daisy as skimage_daisy
from cyvlfeat.sift.dsift import dsift as cyvlfeat_dsift

from .base import ndfeature, winitfeature
from .cython import compute_gradient

scipy_gaussian_filter = None  # expensive


@ndfeature
def gradient(pixels):
    r"""
    Calculates the gradient of an input image. The image is assumed to have
    channel information on the last axis. In the case of multiple channels,
    it returns the gradient over each axis over each channel as the last axis.

    Parameters
    ----------
    pixels : `ndarray`, shape (X, Y, ..., Z, C)
        An array where the last dimension is interpreted as channels. This
        means an N-dimensional image is represented by an N+1 dimensional
        array.

    Returns
    -------
    gradient : ndarray, shape (X, Y, ..., Z, C * length([X, Y, ..., Z]))
        The gradient over each axis over each channel. Therefore, the
        last axis of the gradient of a 2D, single channel image, will have
        length `2`. The last axis of the gradient of a 2D, 3-channel image,
        will have length `6`, he ordering being [Rd_x, Rd_y, Gd_x, Gd_y,
        Bd_x, Bd_y].

    """
    grad_per_dim_per_channel = [np.gradient(g) for g in pixels]
    # Flatten out the separate dims
    grad_per_channel = list(itertools.chain.from_iterable(
        grad_per_dim_per_channel))
    # Add a channel axis for broadcasting
    grad_per_channel = [g[None, ...] for g in grad_per_channel]
    # Concatenate gradient list into an array (the new_image)
    return np.concatenate(grad_per_channel, axis=0)


@ndfeature
def fast_gradient(pixels):
    return compute_gradient(pixels)


@ndfeature
def gaussian_filter(pixels, sigma):
    global scipy_gaussian_filter
    if scipy_gaussian_filter is None:
        from scipy.ndimage import gaussian_filter as scipy_gaussian_filter
    output = np.empty(pixels.shape)
    for dim in range(pixels.shape[0]):
        scipy_gaussian_filter(pixels[dim, ...], sigma, output=output[dim, ...])
    return output


@ndfeature
def daisy(pixels, step=4, radius=15, rings=3, histograms=8, orientations=8,
          normalization='l1', sigmas=None, ring_radii=None):
    pixels = skimage_daisy(pixels[0, ...], step=step, radius=radius,
                           rings=rings, histograms=histograms,
                           orientations=orientations,
                           normalization=normalization, sigmas=sigmas,
                           ring_radii=ring_radii)

    return np.rollaxis(pixels, -1)


@winitfeature
def dsift(pixels, step=1, size=3, bounds=None, window_size=2, norm=False,
          fast=False, float_descriptors=False, geometry=(4, 4, 8)):
    centers, output = cyvlfeat_dsift(np.rot90(pixels[0, ..., ::-1]),
                                     step=step, size=size, bounds=bounds,
                                     window_size=window_size, norm=norm,
                                     fast=fast,
                                     float_descriptors=float_descriptors,
                                     geometry=geometry)
    shape = pixels.shape[1:] - 2 * centers[:, 0]
    return (np.require(output.reshape((-1, shape[0], shape[1])),
                       dtype=np.double),
            np.require(centers.T[..., ::-1].reshape((shape[0], shape[1], 2)),
                       dtype=np.int))



@ndfeature
def no_op(image_data):
    r"""
    A no operation feature - does nothing but return a copy of the pixels
    passed in.
    """
    return image_data.copy()
