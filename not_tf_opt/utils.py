import tensorflow as tf

__all__ = [
    "VariableError",
    "sigmoid_inverse",
    "map_to_bounded_interval",
    "map_from_bounded_interval"
]


class VariableError(Exception):
    """
    Base error thrown by modules in the core
    """


def sigmoid_inverse(x):
    if tf.reduce_any(x < 0.) or tf.reduce_any(x > 1.):
        raise ValueError(f"x = {x} was not in the sigmoid function's range ([0, 1])!")
    x = tf.clip_by_value(x, 1e-10, 1 - 1e-10)

    return tf.math.log(x) - tf.math.log(1. - x)


def map_to_bounded_interval(x, lower, upper):
    return (upper - lower) * tf.nn.sigmoid(x) + lower


def map_from_bounded_interval(x, lower, upper, eps=1e-12):
    if tf.reduce_any(x < lower) or tf.reduce_any(x > upper):
        raise VariableError(f"All values must be in the range {(upper, lower)}! (Got x = {x})")

    return sigmoid_inverse((x - lower) / (upper - lower + eps))
