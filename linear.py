import tensorflow as tf
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.positive_semidefinite_kernels import positive_semidefinite_kernel as psd_kernel
from tensorflow_probability.python.positive_semidefinite_kernels.internal import util

__all__ = [
    'Linear',
]


def _validate_arg_if_not_none(arg, assertion, validate_args):
    if arg is None:
        return arg
    with tf.control_dependencies([assertion(arg)] if validate_args else []):
        result = tf.identity(arg)
    return result


class Linear(psd_kernel.PositiveSemidefiniteKernel):
    def __init__(self,
                 amplitude=None,
                 bias=None,
                 origin=None,
                 feature_ndims=1,
                 validate_args=False,
                 name='Linear'):
        with tf.name_scope(name, values=[amplitude, bias, origin]) as name:
            dtype = dtype_util.common_dtype([amplitude, bias, origin], tf.float32)
            print('dtype: ', dtype)
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
            if origin is not None:
                origin = tf.convert_to_tensor(
                    origin, name='origin', dtype=dtype)
            self._origin = tf.identity(origin)

            tf.assert_same_float_dtype(
                [self._amplitude, self._bias, self._origin])
        super(Linear, self).__init__(
            feature_ndims, dtype=dtype, name=name)
        print('self.origin: ', type(self.origin), self.origin.dtype, self.origin.shape)
        print('self.bias: ', type(self.bias), self.bias.dtype, self.bias.shape)
        print('self.amplitude: ', type(self.amplitude), self.amplitude.dtype, self.amplitude.shape)

    @property
    def amplitude(self):
        """Amplitude parameter."""
        return self._amplitude

    @property
    def bias(self):
        """Bias parameter."""
        return self._bias

    @property
    def origin(self):
        """Origin parameter."""
        return self._origin

    def _batch_shape(self):
        scalar_shape = tf.TensorShape([])
        return tf.broadcast_static_shape(
            tf.broadcast_static_shape(
                scalar_shape if self.amplitude is None else self.amplitude.shape,
                scalar_shape if self.bias is None else self.bias.shape),
            scalar_shape if self.origin is None else self.origin.shape)

    def _batch_shape_tensor(self):
        return tf.broadcast_dynamic_shape(
            tf.broadcast_dynamic_shape(
                [] if self.amplitude is None else tf.shape(self.amplitude),
                [] if self.bias is None else tf.shape(self.bias)),
            [] if self.origin is None else tf.shape(self.origin))

    def _apply(self, x1, x2, param_expansion_ndims=0):
        print('\nLinear._apply')
        print('x1: ', type(x1), x1.dtype, x1.shape)
        print('x2: ', type(x2), x2.dtype, x2.shape)
        if self.origin is not None:
            x1 = x1 - self.origin
            x2 = x2 - self.origin
        print('x1: ', type(x1), x1.dtype, x1.shape)
        print('x2: ', type(x2), x2.dtype, x2.shape)
        print('x2.T: ', type(tf.transpose(x2)), tf.transpose(x2).dtype, tf.transpose(x2).shape)
        dot_prod = tf.tensordot(x1, tf.transpose(x2), axes=1)
        dot_prod = tf.reshape(dot_prod, [x1.shape[0], x2.shape[1]])
        # dot_prod = tf.matmul(x1, x2, transpose_b=True)
        print('dot_prod: ', type(dot_prod), dot_prod.dtype, dot_prod.shape)
        if self.bias is not None:
            bias = util.pad_shape_right_with_ones(
                self.bias, param_expansion_ndims)
            dot_prod += bias ** 2
            print('dot_prod: ', type(dot_prod), dot_prod.dtype, dot_prod.shape)

        if self.amplitude is not None:
            amplitude = util.pad_shape_right_with_ones(
                self.amplitude, param_expansion_ndims)
            dot_prod *= amplitude**2
            print('dot_prod: ', type(dot_prod), dot_prod.dtype, dot_prod.shape)

        return dot_prod
