import scipy
import numpy as np

from menpo.base import DP, Targetable, Vectorizable
from menpo.shape import PointCloud
from menpo.transform import Transform, AlignmentSimilarity
from menpo.model import Similarity2dInstanceModel
from menpo.model.modelinstance import ModelInstance


# Point Distribution Models ---------------------------------------------------

class PDM(ModelInstance, DP):
    r"""Specialization of :map:`ModelInstance` for use with spatial data.
    """

    def __init__(self, model, sigma2=1):
        super(PDM, self).__init__(model)
        self._set_prior(sigma2)

    def _set_prior(self, sigma2):
        self.j_prior = sigma2 / self.model.eigenvalues
        self.h_prior = np.diag(self.j_prior)

    @property
    def n_dims(self):
        r"""
        The number of dimensions of the spatial instance of the model

        :type: int
        """
        return self.model.template_instance.n_dims

    def d_dp(self, points):
        """
        Returns the Jacobian of the PCA model reshaped to have the standard
        Jacobian shape:

            n_points    x  n_params      x  n_dims

            which maps to

            n_features  x  n_components  x  n_dims

            on the linear model

        Returns
        -------
        jacobian : (n_features, n_components, n_dims) ndarray
            The Jacobian of the model in the standard Jacobian shape.
        """
        d_dp = self.model.d_dp.T.reshape(self.model.n_active_components,
                                         -1, self.n_dims)
        return d_dp.swapaxes(0, 1)


class GlobalPDM(PDM):
    r"""
    """
    def __init__(self, model, global_transform_cls, sigma2=1):
        # Start the global_transform as an identity (first call to
        # from_vector_inplace() or set_target() will update this)
        self.global_transform = global_transform_cls(model.mean, model.mean)
        super(GlobalPDM, self).__init__(model, sigma2)

    def _set_prior(self, sigma2):
        sim_prior = np.ones((4,))
        pdm_prior = sigma2 / self.model.eigenvalues
        self.j_prior = np.hstack((sim_prior, pdm_prior))
        self.h_prior = np.diag(self.j_prior)

    @property
    def n_global_parameters(self):
        r"""
        The number of parameters in the `global_transform`

        :type: int
        """
        return self.global_transform.n_parameters

    @property
    def global_parameters(self):
        r"""
        The parameters for the global transform.

        :type: (`n_global_parameters`,) ndarray
        """
        return self.global_transform.as_vector()

    def _new_target_from_state(self):
        r"""
        Return the appropriate target for the model weights provided,
        accounting for the effect of the global transform


        Returns
        -------

        new_target: :class:`menpo.shape.PointCloud`
            A new target for the weights provided
        """
        return self.global_transform.apply(self.model.instance(self.weights))

    def _weights_for_target(self, target):
        r"""
        Return the appropriate model weights for target provided, accounting
        for the effect of the global transform. Note that this method
        updates the global transform to be in the correct state.

        Parameters
        ----------

        target: :class:`menpo.shape.PointCloud`
            The target that the statistical model will try to reproduce

        Returns
        -------

        weights: (P,) ndarray
            Weights of the statistical model that generate the closest
            PointCloud to the requested target
        """

        self._update_global_transform(target)
        projected_target = self.global_transform.pseudoinverse.apply(target)
        # now we have the target in model space, project it to recover the
        # weights
        new_weights = self.model.project(projected_target)
        # TODO investigate the impact of this, could be problematic
        # the model can't perfectly reproduce the target we asked for -
        # reset the global_transform.target to what it CAN produce
        #refined_target = self._target_for_weights(new_weights)
        #self.global_transform.target = refined_target
        return new_weights

    def _update_global_transform(self, target):
        self.global_transform.set_target(target)

    def _as_vector(self):
        r"""
        Return the current parameters of this transform - this is the
        just the linear model's weights

        Returns
        -------
        params : (`n_parameters`,) ndarray
            The vector of parameters
        """
        return np.hstack([self.global_parameters, self.weights])

    def from_vector_inplace(self, vector):
        # First, update the global transform
        global_parameters = vector[:self.n_global_parameters]
        self._update_global_weights(global_parameters)
        # Now extract the weights, and let super handle the update
        weights = vector[self.n_global_parameters:]
        PDM.from_vector_inplace(self, weights)

    def _update_global_weights(self, global_weights):
        r"""
        Hook that allows for overriding behavior when the global weights are
        set. Default implementation simply asks global_transform to
        update itself from vector.
        """
        self.global_transform.from_vector_inplace(global_weights)

    def d_dp(self, points):
        # d_dp is always evaluated at the mean shape
        points = self.model.mean.points

        # compute dX/dp

        # dX/dq is the Jacobian of the global transform evaluated at the
        # current target
        # (n_points, n_global_params, n_dims)
        dX_dq = self._global_transform_d_dp(points)

        # by application of the chain rule dX/db is the Jacobian of the
        # model transformed by the linear component of the global transform
        # (n_points, n_weights, n_dims)
        dS_db = PDM.d_dp(self, [])
        # (n_points, n_dims, n_dims)
        dX_dS = self.global_transform.d_dx(points)
        # (n_points, n_weights, n_dims)
        dX_db = np.einsum('ilj, idj -> idj', dX_dS, dS_db)

        # dX/dp is simply the concatenation of the previous two terms
        # (n_points, n_params, n_dims)
        return np.hstack((dX_dq, dX_db))

    def _global_transform_d_dp(self, points):
        return self.global_transform.d_dp(points)


class OrthoPDM(GlobalPDM):
    r"""
    """
    def __init__(self, model, sigma2=1):
        # 1. Construct similarity model from the mean of the model
        self.similarity_model = Similarity2dInstanceModel(model.mean)
        # 2. Orthonormalize model and similarity model
        model_cpy = model.copy()
        model_cpy.orthonormalize_against_inplace(self.similarity_model)
        self.similarity_weights = self.similarity_model.project(model_cpy.mean)
        super(OrthoPDM, self).__init__(model_cpy, AlignmentSimilarity, sigma2)

    @property
    def global_parameters(self):
        r"""
        The parameters for the global transform.

        :type: (`n_global_parameters`,) ndarray
        """
        return self.similarity_weights

    def _update_global_transform(self, target):
        self.similarity_weights = self.similarity_model.project(target)
        self._update_global_weights(self.similarity_weights)

    def _update_global_weights(self, global_weights):
        self.similarity_weights = global_weights
        new_target = self.similarity_model.instance(global_weights)
        self.global_transform.set_target(new_target)

    def _global_transform_d_dp(self, points):
        return self.similarity_model.d_dp.T.reshape(
            self.n_global_parameters, -1, self.n_dims).swapaxes(0, 1)


# Linear Warps ----------------------------------------------------------------

class LinearWarp(OrthoPDM, Transform):

    def __init__(self, model, n_landmarks, sigma2=1):
        super(LinearWarp, self).__init__(model, sigma2)
        self.n_landmarks = n_landmarks

        self.W = np.vstack((self.similarity_model.components,
                            self.model.components))
        V = self.W[:, :self.n_dims*self.n_landmarks]
        self.pinv_V = scipy.linalg.pinv(V)

    @property
    def dense_target(self):
        return PointCloud(self.target.points[self.n_landmarks:])

    @property
    def sparse_target(self):
        return PointCloud(self.target.points[:self.n_landmarks])

    def set_target(self, target):
        if target.n_points == self.n_landmarks:
            # densify target
            target = np.dot(np.dot(target.as_vector(), self.pinv_V), self.W)
            target = PointCloud(np.reshape(target, (-1, self.n_dims)))
        OrthoPDM.set_target(self, target)

    def _apply(self, _, **kwargs):
        return self.target.points[self.n_landmarks:]

    def d_dp(self, _):
        return OrthoPDM.d_dp(self, _)[self.n_landmarks:, ...]


# Non-Linear Warps ------------------------------------------------------------

class ModelDrivenTransform(Transform, Targetable, Vectorizable, DP):
    r"""
    A transform that couples a traditional landmark-based transform to a
    statistical model such that source points of the alignment transform
    are the points of the model. The weights of the transform are just
    the weights of statistical model.

    If no source is provided, the mean of the model is defined as the
    source landmarks of the transform.

    Parameters
    ----------
    model : :class:`menpo.model.base.StatisticalModel`
        A linear statistical shape model.
    transform_cls : :class:`menpo.transform.AlignableTransform`
        A class of :class:`menpo.transform.base.AlignableTransform`
        The align constructor will be called on this with the source
        and target landmarks. The target is
        set to the points generated from the model using the
        provide weights - the source is either given or set to the
        model's mean.
    source : :class:`menpo.shape.base.PointCloud`
        The source landmarks of the transform. If None, the mean of the model
         is used.

        Default: None

    """
    def __init__(self, model, transform_cls, source=None, sigma2=1):
        self.pdm = PDM(model, sigma2=sigma2)
        self._cached_points, self.dW_dl = None, None
        self.transform = transform_cls(source, self.target)

    @property
    def n_dims(self):
        r"""
        The number of dimensions that the transform supports.

        :type: int
        """
        return self.pdm.n_dims

    def _apply(self, x, **kwargs):
        r"""
        Apply this transform to the given object. Uses the internal transform.

        Parameters
        ----------
        x : (N, D) ndarray or a transformable object
            The object to be transformed.
        kwargs : dict
            Passed through to transforms `apply_inplace` method.

        Returns
        --------
        transformed : (N, D) ndarray or object
            The transformed object
        """
        return self.transform._apply(x, **kwargs)

    @property
    def target(self):
        return self.pdm.target

    def _target_setter(self, new_target):
        r"""
        On a new target being set, we need to:

        Parameters
        ----------

        new_target: :class:`PointCloud`
            The new_target that we want to set.
        """
        self.pdm.set_target(new_target)

    def _new_target_from_state(self):
        # We delegate to PDM to handle all our Targetable duties. As a
        # result, *we* never need to call _sync_target_for_state, so we have
        # no need for an implementation of this method. Of course the
        # interface demands it, so the stub is here. Contrast with
        # _target_setter, which is required, because we will have to handle
        # external calls to set_target().
        pass

    def _sync_state_from_target(self):
        # Let the pdm update its state
        self.pdm._sync_state_from_target()
        # and update our transform to the new state
        self.transform.set_target(self.target)

    @property
    def n_parameters(self):
        r"""
        The total number of parameters.

        Simply ``n_weights``.

        :type: int
        """
        return self.pdm.n_parameters

    def _as_vector(self):
        r"""
        Return the current weights of this transform - this is the
        just the linear model's weights

        Returns
        -------
        params : (`n_parameters`,) ndarray
            The vector of weights
        """
        return self.pdm.as_vector()

    def from_vector_inplace(self, vector):
        r"""
        Updates the ModelDrivenTransform's state from it's
        vectorized form.
        """
        self.pdm.from_vector_inplace(vector)
        # By here the pdm has updated our target state, we just need to
        # update the transform
        self.transform.set_target(self.target)

    def d_dp(self, points):
        r"""
        The derivative of this MDT wrt parametrization changes evaluated at
        points.

        This is done by chaining the derivative of points wrt the
        source landmarks on the transform (dW/dL) together with the Jacobian
        of the linear model wrt its weights (dX/dp).

        Parameters
        ----------

        points: ndarray shape (n_points, n_dims)
            The spatial points at which the derivative should be evaluated.

        Returns
        -------

        ndarray shape (n_points, n_params, n_dims)
            The jacobian wrt parameterization

        """
        # check if re-computation of dW/dl can be avoided
        if not np.array_equal(self._cached_points, points):
            # recompute dW/dl, the derivative each point wrt
            # the source landmarks
            self.dW_dl = self.transform.d_dl(points)
            # cache points
            self._cached_points = points

        # dX/dp is simply the Jacobian of the PDM
        dX_dp = self.pdm.d_dp(points)

        # PREVIOUS
        # dW_dX:  n_points x n_centres x n_dims
        # dX_dp:  n_centres x n_params x n_dims

        # dW_dl:  n_points x (n_dims) x n_centres x n_dims
        # dX_dp:  (n_points x n_dims) x n_params
        dW_dp = np.einsum('ild, lpd -> ipd', self.dW_dl, dX_dp)
        # dW_dp:  n_points x n_params x n_dims

        return dW_dp

    def jp(self):
        r"""

        References
        ----------

        .. [1] G. Papandreou and P. Maragos, "Adaptive and Constrained
               Algorithms for Inverse Compositional Active Appearance Model
               Fitting", CVPR08
        """
        # the incremental warp is always evaluated at p=0, ie the mean shape
        points = self.pdm.model.mean.points

        # compute:
        #   - dW/dp when p=0
        #   - dW/dp when p!=0
        #   - dW/dx when p!=0 evaluated at the source landmarks

        # dW/dp when p=0 and when p!=0 are the same and simply given by
        # the Jacobian of the model
        # (n_points, n_params, n_dims)
        dW_dp_0 = self.pdm.d_dp(points)
        # (n_points, n_params, n_dims)
        dW_dp = dW_dp_0

        # (n_points, n_dims, n_dims)
        dW_dx = self.transform.d_dx(points)

        # (n_points, n_params, n_dims)
        dW_dx_dW_dp_0 = np.einsum('ijk, ilk -> eilk', dW_dx, dW_dp_0)

        # (n_params, n_params)
        J = np.einsum('ijk, ilk -> jl', dW_dp, dW_dx_dW_dp_0)
        # (n_params, n_params)
        H = np.einsum('ijk, ilk -> jl', dW_dp, dW_dp)
        # (n_params, n_params)
        Jp = np.linalg.solve(H, J)

        return Jp

    @property
    def j_prior(self):
        return self.pdm.j_prior

    @property
    def h_prior(self):
        return self.pdm.h_prior


class GlobalMDTransform(ModelDrivenTransform):
    r"""
    A transform that couples an alignment transform to a
    statistical model together with a global similarity transform,
    such that the weights of the transform are fully specified by
    both the weights of statistical model and the weights of the
    similarity transform. The model is assumed to
    generate an instance which is then transformed by the similarity
    transform; the result defines the target landmarks of the transform.
    If no source is provided, the mean of the model is defined as the
    source landmarks of the transform.

    Parameters
    ----------
    model : :class:`menpo.model.base.StatisticalModel`
        A linear statistical shape model.
    transform_cls : :class:`menpo.transform.AlignableTransform`
        A class of :class:`menpo.transform.base.AlignableTransform`
        The align constructor will be called on this with the source
        and target landmarks. The target is
        set to the points generated from the model using the
        provide weights - the source is either given or set to the
        model's mean.
    global_transform : :class:`menpo.transform.AlignableTransform`
        A class of :class:`menpo.transform.base.AlignableTransform`
        The global transform that should be applied to the model output.
        Doesn't have to have been constructed from the .align() constructor.
        Note that the GlobalMDTransform isn't guaranteed to hold on to the
        exact object passed in here - so don't expect external changes to
        the global_transform to be reflected in the behavior of this object.
    source : :class:`menpo.shape.base.PointCloud`, optional
        The source landmarks of the transform. If no `source` is provided the
        mean of the model is used.
    weights : (P,) ndarray, optional
        The reconstruction weights that will be fed to the model in order to
        generate an instance of the target landmarks.
    composition: 'both', 'warp' or 'model', optional
        The composition approximation employed by this
        ModelDrivenTransform.

        Default: `both`
    """
    def __init__(self, model, transform_cls, global_transform, source=None,
                 sigma2=1):
        self.pdm = GlobalPDM(model, global_transform, sigma2=sigma2)
        self._cached_points = None
        self.transform = transform_cls(source, self.target)

    def jp(self):
        r"""
        Composes two ModelDrivenTransforms together based on the
        first order approximation proposed by Papandreou and Maragos in [1].

        Parameters
        ----------
        delta : (N,) ndarray
            Vectorized :class:`ModelDrivenTransform` to be applied **before**
            self

        Returns
        --------
        transform : self
            self, updated to the result of the composition


        References
        ----------

        .. [1] G. Papandreou and P. Maragos, "Adaptive and Constrained
               Algorithms for Inverse Compositional Active Appearance Model
               Fitting", CVPR08
        """
        # the incremental warp is always evaluated at p=0, ie the mean shape
        points = self.pdm.model.mean.points

        # compute:
        #   - dW/dp when p=0
        #   - dW/dp when p!=0
        #   - dW/dx when p!=0 evaluated at the source landmarks

        # dW/dq when p=0 and when p!=0 are the same and given by the
        # Jacobian of the global transform evaluated at the mean of the
        # model
        # (n_points, n_global_params, n_dims)
        dW_dq = self.pdm._global_transform_d_dp(points)

        # dW/db when p=0, is the Jacobian of the model
        # (n_points, n_weights, n_dims)
        dW_db_0 = PDM.d_dp(self.pdm, points)

        # dW/dp when p=0, is simply the concatenation of the previous
        # two terms
        # (n_points, n_params, n_dims)
        dW_dp_0 = np.hstack((dW_dq, dW_db_0))

        # by application of the chain rule dW_db when p!=0,
        # is the Jacobian of the global transform wrt the points times
        # the Jacobian of the model: dX(S)/db = dX/dS *  dS/db
        # (n_points, n_dims, n_dims)
        dW_dS = self.pdm.global_transform.d_dx(points)
        # (n_points, n_weights, n_dims)
        dW_db = np.einsum('ilj, idj -> idj', dW_dS, dW_db_0)

        # dW/dp is simply the concatenation of dW_dq with dW_db
        # (n_points, n_params, n_dims)
        dW_dp = np.hstack((dW_dq, dW_db))

        # dW/dx is the jacobian of the transform evaluated at the source
        # landmarks
        # (n_points, n_dims, n_dims)
        dW_dx = self.transform.d_dx(points)

        # (n_points, n_params, n_dims)
        dW_dx_dW_dp_0 = np.einsum('ijk, ilk -> ilk', dW_dx, dW_dp_0)

        # (n_params, n_params)
        J = np.einsum('ijk, ilk -> jl', dW_dp, dW_dx_dW_dp_0)
        # (n_params, n_params)
        H = np.einsum('ijk, ilk -> jl', dW_dp, dW_dp)
        # (n_params, n_params)
        Jp = np.linalg.solve(H, J)

        return Jp


class OrthoMDTransform(GlobalMDTransform):
    r"""
    A transform that couples an alignment transform to a
    statistical model together with a global similarity transform,
    such that the weights of the transform are fully specified by
    both the weights of statistical model and the weights of the
    similarity transform. The model is assumed to
    generate an instance which is then transformed by the similarity
    transform; the result defines the target landmarks of the transform.
    If no source is provided, the mean of the model is defined as the
    source landmarks of the transform.

    This transform (in contrast to the :class:`GlobalMDTransform`)
    additionally orthonormalizes both the global and the model basis against
    each other, ensuring that orthogonality and normalization is enforced
    across the unified bases.

    Parameters
    ----------
    model : :class:`menpo.model.base.StatisticalModel`
        A linear statistical shape model.
    transform_cls : :class:`menpo.transform.AlignableTransform`
        A class of :class:`menpo.transform.base.AlignableTransform`
        The align constructor will be called on this with the source
        and target landmarks. The target is
        set to the points generated from the model using the
        provide weights - the source is either given or set to the
        model's mean.
    global_transform : :class:`menpo.transform.Aligna
    bleTransform`
        A class of :class:`menpo.transform.base.AlignableTransform`
        The global transform that should be applied to the model output.
        Doesn't have to have been constructed from the .align() constructor.
        Note that the GlobalMDTransform isn't guaranteed to hold on to the
        exact object passed in here - so don't expect external changes to
        the global_transform to be reflected in the behavior of this object.
    source : :class:`menpo.shape.base.PointCloud`, optional
        The source landmarks of the transform. If no `source` is provided the
        mean of the model is used.
    """
    def __init__(self, model, transform_cls, source=None, sigma2=1):
        self.pdm = OrthoPDM(model, sigma2=sigma2)
        self._cached_points = None
        self.transform = transform_cls(source, self.target)


