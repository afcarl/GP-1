import tensorflow as tf
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.positive_semidefinite_kernels import positive_semidefinite_kernel as psd_kernel
from tensorflow_probability.python.positive_semidefinite_kernels.internal import util

__all__ = [
    'DotProd',
]


def _validate_arg_if_not_none(arg, assertion, validate_args):
    if arg is None:
        return arg
    with tf.control_dependencies([assertion(arg)] if validate_args else []):
        result = tf.identity(arg)
    return result


class DotProd(psd_kernel.PositiveSemidefiniteKernel):
    def __init__(self,
                 amplitude=None,
                 bias=None,
                 power=None,
                 feature_ndims=1,
                 validate_args=False,
                 name='DotProd'):
        with tf.name_scope(name, values=[amplitude, bias, power]) as name:
            dtype = dtype_util.common_dtype([amplitude, bias, power], tf.float32)
            if amplitude is not None:
                amplitude = tf.convert_to_tensor(
                    amplitude, name='amplitude', dtype=dtype)
            self._amplitude = _validate_arg_if_not_none(
                amplitude, tf.assert_positive, validate_args)
            if bias is not None:
                bias = tf.convert_to_tensor(
                    bias, name='bias', dtype=dtype)
            self._bias = _validate_arg_if_not_none(
                bias, tf.assert_positive, validate_args)
            if power is not None:
                power = tf.convert_to_tensor(
                    power, name='power', dtype=dtype)
            self._power = _validate_arg_if_not_none(
                power, tf.assert_positive, validate_args)

            tf.assert_same_float_dtype(
                [self._amplitude, self._bias, self._power])
        super(DotProd, self).__init__(
            feature_ndims, dtype=dtype, name=name)


    @property
    def amplitude(self):
        """Amplitude parameter."""
        return self._amplitude

    @property
    def bias(self):
        """Bias parameter."""
        return self._bias

    @property
    def power(self):
        """Power parameter."""
        return self._power

    def _batch_shape(self):
        scalar_shape = tf.TensorShape([])
        return tf.broadcast_static_shape(
            tf.broadcast_static_shape(
                scalar_shape if self.amplitude is None else self.amplitude.shape,
                scalar_shape if self.bias is None else self.bias.shape),
            scalar_shape if self.power is None else self.power.shape)

    def _batch_shape_tensor(self):
        return tf.broadcast_dynamic_shape(
            tf.broadcast_dynamic_shape(
                [] if self.amplitude is None else tf.shape(self.amplitude),
                [] if self.bias is None else tf.shape(self.bias)),
            [] if self.power is None else tf.shape(self.power))

    def _apply(self, x1, x2, param_expansion_ndims=0):
        dot_prod = tf.tensordot(x1, tf.transpose(x2), axes=1)
        dot_prod = tf.reshape(dot_prod, [x1.shape[0], x2.shape[1]])
        if self.bias is not None:
            bias = util.pad_shape_right_with_ones(
                self.bias, param_expansion_ndims)
            dot_prod += bias ** 2.

        if self.power is not None:
            power = util.pad_shape_right_with_ones(
                self.power, param_expansion_ndims)
            dot_prod **= power

        if self.amplitude is not None:
            amplitude = util.pad_shape_right_with_ones(
                self.amplitude, param_expansion_ndims)
            dot_prod *= amplitude ** 2.


        return dot_prod
