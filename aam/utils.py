from __future__ import division
import numpy as np

from skimage.filter import gaussian_filter

from menpo.shape import TriMesh
from menpo.transform import Translation

from .cython import extract_patches_cython
from .image import Image, MaskedImage


def build_reference_frame(landmarks, boundary=3, group='source', trilist=None):
    r"""
    Builds a reference frame from a particular set of landmarks.

    Parameters
    ----------
    landmarks : :map:`PointCloud`
        The landmarks that will be used to build the reference frame.

    boundary : `int`, optional
        The number of pixels to be left as a safe margin on the boundaries
        of the reference frame (has potential effects on the gradient
        computation).

    group : `string`, optional
        Group that will be assigned to the provided set of landmarks on the
        reference frame.

    trilist : ``(t, 3)`` `ndarray`, optional
        Triangle list that will be used to build the reference frame.

        If ``None``, defaults to performing Delaunay triangulation on the
        points.

    Returns
    -------
    reference_frame : :map:`Image`
        The reference frame.
    """
    reference_frame = _build_reference_frame(landmarks, boundary=boundary,
                                             group=group)
    if trilist is not None:
        reference_frame.landmarks[group] = TriMesh(
            reference_frame.landmarks[group].lms.points, trilist=trilist)

    # TODO: revise kwarg trilist in method constrain_mask_to_landmarks,
    # perhaps the trilist should be directly obtained from the group landmarks
    reference_frame.constrain_mask_to_landmarks(group=group, trilist=trilist)

    return reference_frame


def build_patch_reference_frame(landmarks, boundary=3, group='source',
                                patch_shape=(16, 16)):
    r"""
    Builds a reference frame from a particular set of landmarks.

    Parameters
    ----------
    landmarks : :map:`PointCloud`
        The landmarks that will be used to build the reference frame.

    boundary : `int`, optional
        The number of pixels to be left as a safe margin on the boundaries
        of the reference frame (has potential effects on the gradient
        computation).

    group : `string`, optional
        Group that will be assigned to the provided set of landmarks on the
        reference frame.

    patch_shape : tuple of ints, optional
        Tuple specifying the shape of the patches.

    Returns
    -------
    patch_based_reference_frame : :map:`Image`
        The patch based reference frame.
    """
    boundary = np.max(patch_shape) + boundary
    reference_frame = _build_reference_frame(landmarks, boundary=boundary,
                                             group=group)

    # mask reference frame
    reference_frame.build_mask_around_landmarks(patch_shape, group=group)

    return reference_frame


def _build_reference_frame(landmarks, boundary=3, group='source'):
    # translate landmarks to the origin
    minimum = landmarks.bounds(boundary=boundary)[0]
    landmarks = Translation(-minimum).apply(landmarks)

    resolution = landmarks.range(boundary=boundary)
    reference_frame = MaskedImage.blank(resolution)
    reference_frame.landmarks[group] = landmarks

    return reference_frame


def build_parts_image(image, centres, parts_shape, normalize_parts=False):

    # call cython module
    parts = extract_patches_cython(image.pixels, centres.points,
                                   np.array(parts_shape), np.array([[0, 0]]))

    # build parts image
    img = Image(parts[..., 0, :, :])

    if normalize_parts:
        # normalize parts
        img.normalize_norm_inplace()

    return img


def _build_parts_image(image, centres, parts_shape, out=None):

    if out is not None:
        parts = out
    else:
        parts = np.zeros(
             parts_shape + (centres.n_points,) + (image.n_channels,))

    image_size = np.array(image.shape, dtype=np.int)
    parts_shape = np.array(parts_shape, dtype=np.int)
    centres = np.require(centres.points, dtype=np.int)
    half_parts_shape = np.require(np.floor(parts_shape / 2), dtype=np.int)
    # deal with odd parts shapes
    # - add_to_parts[axis] = 0 if parts_shape[axis] is odd
    # - add_to_parts[axis] = 1 if parts_shape[axis] is even
    add_to_parts = np.mod(parts_shape, 2)

    # 1. Compute the extents
    c_min = centres - half_parts_shape
    c_max = centres + half_parts_shape + add_to_parts
    out_min_min = c_min < 0
    out_min_max = c_min > image_size
    out_max_min = c_max < 0
    out_max_max = c_max > image_size

    # 2. Build the extraction slices
    ext_s_min = c_min.copy()
    ext_s_max = c_max.copy()
    # clamp the min to 0
    ext_s_min[out_min_min] = 0
    ext_s_max[out_max_min] = 0
    # clamp the max to image bounds across each dimension
    for i in xrange(image.n_dims):
        ext_s_max[out_max_max[:, i], i] = image_size[i] - 1
        ext_s_min[out_min_max[:, i], i] = image_size[i] - 1

    # 3. Build the insertion slices
    ins_s_min = ext_s_min - c_min
    ins_s_max = np.maximum(ext_s_max - c_max + parts_shape, (0, 0))

    for i, (e_a, e_b, i_a, i_b) in enumerate(zip(ext_s_min, ext_s_max,
                                                 ins_s_min, ins_s_max)):
        # build a list of insertion slices and extraction slices
        i_slices = [slice(a, b) for a, b in zip(i_a, i_b)]
        e_slices = [slice(a, b) for a, b in zip(e_a, e_b)]
        # get a view onto the parts we are on
        part = parts[..., i, :]
        # apply the slices to map
        part[i_slices] = image.pixels[e_slices]

    # build and return parts image
    return Image(parts)


def flatten_out(list_of_lists):
    return [i for l in list_of_lists for i in l]


fsmooth = lambda x, sigma: gaussian_filter(x, sigma, mode='constant')
