import tensorflow as tf


def assert_near(x, y, atol=None, rtol=None):
    assert tf.debugging.assert_near(x, y, rtol=rtol, atol=atol) is None
