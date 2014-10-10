from __future__ import division
import abc
import numpy as np

from hdf5able import HDF5able, SerializableCallable

from menpo.shape import TriMesh
from menpo.transform.thinplatesplines import ThinPlateSplines

from .utils import  build_reference_frame, build_patch_reference_frame


# Abstract Interface for AAM Objects ------------------------------------------

class AAM(object):

    __metaclass__ = abc.ABCMeta

    @property
    def n_levels(self):
        """
        The number of scale levels of the AAM.

        :type: `int`
        """
        return len(self.scales)

    def instance(self, shape_weights=None, appearance_weights=None, level=-1):
        r"""
        Generates a novel AAM instance given a set of shape and appearance
        weights. If no weights are provided, the mean AAM instance is
        returned.

        Parameters
        -----------
        shape_weights : ``(n_weights,)`` `ndarray` or `float` list
            Weights of the shape model that will be used to create
            a novel shape instance. If ``None``, the mean shape
            ``(shape_weights = [0, 0, ..., 0])`` is used.

        appearance_weights : ``(n_weights,)`` `ndarray` or `float` list
            Weights of the appearance model that will be used to create
            a novel appearance instance. If ``None``, the mean appearance
            ``(appearance_weights = [0, 0, ..., 0])`` is used.

        level : `int`, optional
            The pyramidal level to be used.

        Returns
        -------
        image : :map:`Image`
            The novel AAM instance.
        """
        sm = self.shape_models[level]
        am = self.appearance_models[level]

        # TODO: this bit of logic should to be transferred down to PCAModel
        if shape_weights is None:
            shape_weights = [0]
        if appearance_weights is None:
            appearance_weights = [0]
        n_shape_weights = len(shape_weights)
        shape_weights *= sm.eigenvalues[:n_shape_weights] ** 0.5
        shape_instance = sm.instance(shape_weights)
        n_appearance_weights = len(appearance_weights)
        appearance_weights *= am.eigenvalues[:n_appearance_weights] ** 0.5
        appearance_instance = am.instance(appearance_weights)

        return self._instance(level, shape_instance, appearance_instance)

    def random_instance(self, level=-1):
        r"""
        Generates a novel random instance of the AAM.

        Parameters
        -----------
        level : `int`, optional
            The pyramidal level to be used.

        Returns
        -------
        image : :map:`Image`
            The novel AAM instance.
        """
        sm = self.shape_models[level]
        am = self.appearance_models[level]

        # TODO: this bit of logic should to be transferred down to PCAModel
        shape_weights = (np.random.randn(sm.n_active_components) *
                         sm.eigenvalues[:sm.n_active_components]**0.5)
        shape_instance = sm.instance(shape_weights)
        appearance_weights = (np.random.randn(am.n_active_components) *
                              am.eigenvalues[:am.n_active_components]**0.5)
        appearance_instance = am.instance(appearance_weights)

        return self._instance(level, shape_instance, appearance_instance)

    def h5_dict_to_serializable_dict(self):
        import menpo.transform
        d = self.__dict__.copy()
        transform = d.pop('transform')
        d['transform'] = SerializableCallable(transform, [menpo.transform])

        import aam.feature
        features = d.pop('features')
        if self.scale_features:
            # features is a single callable
            d['features'] = SerializableCallable(features, [aam.feature])
        else:
            # features is a list of callables
            d['features'] = [SerializableCallable(f, [aam.feature])
                             for f in features]
        return d


# Concrete Implementations of AAM Objects -------------------------------------

class GlobalAAM(AAM, HDF5able):

    def __init__(self, shape_models, appearance_models, reference_shape,
                 transform, features, sigma, scales, scale_shapes,
                 scale_features):
        self.shape_models = shape_models
        self.appearance_models = appearance_models
        self.transform = transform
        self.features = features
        self.reference_shape = reference_shape
        self.sigma = sigma
        self.scales = scales
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features

    def _instance(self, level, shape_instance, appearance_instance):
        template = self.appearance_models[level].mean
        landmarks = template.landmarks['source'].lms

        reference_frame = self._build_reference_frame(
            shape_instance, landmarks)

        transform = self.transform(
            reference_frame.landmarks['source'].lms, landmarks)

        instance = appearance_instance.warp_to_mask(
            reference_frame.mask, transform)
        instance.landmarks = reference_frame.landmarks

        return instance

    def _build_reference_frame(self, reference_shape, landmarks):
        if type(landmarks) == TriMesh:
            trilist = landmarks.trilist
        else:
            trilist = None
        return build_reference_frame(reference_shape, trilist=trilist)


class PatchAAM(AAM, HDF5able):

    def __init__(self, shape_models, appearance_models, reference_shape,
                 patch_shape, features, sigma, scales, scale_shapes,
                 scale_features):

        self.shape_models = shape_models
        self.appearance_models = appearance_models
        self.transform = ThinPlateSplines
        self.patch_shape = patch_shape
        self.features = features
        self.reference_shape = reference_shape
        self.sigma = sigma
        self.scales = scales
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features

    def _instance(self, level, shape_instance, appearance_instance):
        template = self.appearance_models[level].mean
        landmarks = template.landmarks['source'].lms

        reference_frame = self._build_reference_frame(
            shape_instance, landmarks)

        transform = self.transform(
            reference_frame.landmarks['source'].lms, landmarks)

        instance = appearance_instance.warp_to_mask(reference_frame.mask,
                                                transform)
        instance.landmarks = reference_frame.landmarks

        return instance

    def _build_reference_frame(self, reference_shape, _):
        return build_patch_reference_frame(
            reference_shape, patch_shape=self.patch_shape)


class LinearGlobalAAM(AAM, HDF5able):

    def __init__(self, shape_models, appearance_models, reference_shape,
                 transform, features, sigma, scales, scale_shapes,
                 scale_features, n_landmarks):

        self.shape_models = shape_models
        self.appearance_models = appearance_models
        self.transform = transform
        self.features = features
        self.reference_shape = reference_shape
        self.sigma = sigma
        self.scales = scales
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features
        self.n_landmarks = n_landmarks


class LinearPatchAAM(AAM, HDF5able):

    def __init__(self, shape_models, appearance_models, reference_shape,
                 patch_shape, features, sigma, scales, scale_shapes,
                 scale_features, n_landmarks):

        self.shape_models = shape_models
        self.appearance_models = appearance_models
        self.transform = ThinPlateSplines
        self.patch_shape = patch_shape
        self.features = features
        self.reference_shape = reference_shape
        self.sigma = sigma
        self.scales = scales
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features
        self.n_landmarks = n_landmarks


class PartsAAM(AAM, HDF5able):

    def __init__(self, shape_models, appearance_models, reference_shape,
                 parts_shape, features, sigma, scales, scale_shapes,
                 scale_features):

        self.shape_models = shape_models
        self.appearance_models = appearance_models
        self.parts_shape = parts_shape
        self.features = features
        self.sigma = sigma
        self.reference_shape = reference_shape
        self.scales = scales
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features

    def h5_dict_to_serializable_dict(self):
        import aam.feature
        d = self.__dict__.copy()

        features = d.pop('features')
        if self.scale_features:
            # features is a single callable
            d['features'] = SerializableCallable(features, [aam.feature])
        else:
            # features is a list of callables
            d['features'] = [SerializableCallable(f, [aam.feature])
                             for f in features]
        return d
