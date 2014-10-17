import numpy as np
from scipy.signal import cosine


class mcf(object):
    r"""
    Multi-channel Correlation Filter
    """
    def __init__(self, X, Y, l=0, cosine_mask=False):

        if X[0].shape[1:] != (len(Y),) + Y[0].shape:
            raise ValueError('')

        n_channels, n_offsets, height, width = X[0].shape

        self._cosine_mask = 1
        if cosine_mask:
            self._cosine_mask = np.sum(np.meshgrid(cosine(height),
                                                   cosine(width)), axis=0)

        X_hat = self._compute_fft2s(X)
        Y_hat = self._compute_fft2s(Y)

        self.f = np.zeros((n_channels, height, width), dtype=np.complex64)
        for i in xrange(n_channels):
            for j in xrange(height):
                for k in xrange(width):
                    H_hat = 0
                    J_hat = 0
                    for x_hat in X_hat:
                        for o, y_hat in enumerate(Y_hat):
                            H_hat += (np.conj(x_hat[i, o, j, k]) *
                                      x_hat[i, o, j, k])
                            J_hat += (np.conj(x_hat[i, o, j, k]) *
                                      y_hat[j, k])
                    H_hat += l
                    self.f[i, j, k] = J_hat / H_hat

    def _compute_fft2s(self, X):
        X_hat = []
        for x in X:
            x_hat = np.require(np.fft.fft2(self._cosine_mask * x),
                               dtype=np.complex64)
            X_hat.append(x_hat)
        return X_hat

    def __call__(self, x):
        return np.real(
            np.fft.ifft2(self.f * np.fft.fft2(self._cosine_mask * x)))


