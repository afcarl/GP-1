"""The RationalQuadratic kernel."""
import tensorflow as tf
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.positive_semidefinite_kernels import positive_semidefinite_kernel as psd_kernel
from tensorflow_probability.python.positive_semidefinite_kernels.internal import util

__all__ = [
    'RationalQuadratic',
]


def _validate_arg_if_not_none(arg, assertion, validate_args):
    if arg is None:
        return arg
    with tf.control_dependencies([assertion(arg)] if validate_args else []):
        result = tf.identity(arg)
    return result


class RationalQuadratic(psd_kernel.PositiveSemidefiniteKernel):

    def __init__(self,
                 amplitude=None,
                 length_scale=None,
                 scale_mixture=None,
                 feature_ndims=1,
                 validate_args=False,
                 name='RationalQuadratic'):

        with tf.name_scope(name, values=[amplitude, length_scale]) as name:
            dtype = dtype_util.common_dtype([amplitude, length_scale], tf.float32)
            if amplitude is not None:
                amplitude = tf.convert_to_tensor(
                    amplitude, name='amplitude', dtype=dtype)
            self._amplitude = _validate_arg_if_not_none(
                amplitude, tf.assert_positive, validate_args)
            if length_scale is not None:
                length_scale = tf.convert_to_tensor(
                    length_scale, name='length_scale', dtype=dtype)
            self._length_scale = _validate_arg_if_not_none(
                length_scale, tf.assert_positive, validate_args)
            if scale_mixture is not None:
                scale_mixture = tf.convert_to_tensor(
                    scale_mixture, name='scale_mixture', dtype=dtype)
            self._scale_mixture = _validate_arg_if_not_none(
                scale_mixture, tf.assert_positive, validate_args)
            tf.assert_same_float_dtype([self._amplitude, self._length_scale, self._scale_mixture])
        super(RationalQuadratic, self).__init__(
            feature_ndims, dtype=dtype, name=name)

    @property
    def amplitude(self):
        """Amplitude parameter."""
        return self._amplitude

    @property
    def length_scale(self):
        """Length scale parameter."""
        return self._length_scale

    @property
    def scale_mixture(self):
        """Scale mixture parameter."""
        return self._scale_mixture

    def _batch_shape(self):
        scalar_shape = tf.TensorShape([])
        return tf.broadcast_static_shape(
            tf.broadcast_static_shape(
                scalar_shape if self.amplitude is None else self.amplitude.shape,
                scalar_shape if self.length_scale is None else self.length_scale.shape),
            scalar_shape if self.scale_mixture is None else self.scale_mixture.shape)

    def _batch_shape_tensor(self):
        return tf.broadcast_dynamic_shape(
            tf.broadcast_dynamic_shape(
                [] if self.amplitude is None else tf.shape(self.amplitude),
                [] if self.length_scale is None else tf.shape(self.length_scale)),
            [] if self.scale_mixture is None else tf.shape(self.scale_mixture))

    def _apply(self, x1, x2, param_expansion_ndims=0):
        kernel = 0.5 * util.sum_rightmost_ndims_preserving_shape(
            tf.squared_difference(x1, x2), self.feature_ndims)
        if self.length_scale is not None:
            length_scale = util.pad_shape_right_with_ones(
                self.length_scale, param_expansion_ndims)
            kernel /= length_scale ** 2
        if self.scale_mixture is not None:
            scale_mixture = util.pad_shape_right_with_ones(
                self.scale_mixture, param_expansion_ndims)
            kernel /= scale_mixture
            kernel += 1.
            kernel **= -scale_mixture
        else:
            kernel += 1.
        if self.amplitude is not None:
            amplitude = util.pad_shape_right_with_ones(
                self.amplitude, param_expansion_ndims)
            kernel += amplitude ** 2
        return kernel
