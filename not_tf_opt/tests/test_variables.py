import pytest
import tensorflow as tf

from not_tf_opt import sigmoid_inverse, AbstractVariable, UnconstrainedVariable, PositiveVariable, BoundedVariable


class TestSigmoidInverse:
    def test_sigmoid_inverse(self):

        # Define some test inputs
        test_xs = tf.linspace(1e-10, 1 - 1e-10, 1000)
        test_reparam_xs = tf.linspace(-9., 9., 1000)

        inner_composition = tf.nn.sigmoid(sigmoid_inverse(test_xs))
        outer_composition = sigmoid_inverse(tf.nn.sigmoid(test_reparam_xs))

        inner_max_error = tf.reduce_max(tf.abs(inner_composition - test_xs))
        outer_max_error = tf.reduce_max(tf.abs(outer_composition - test_reparam_xs))

        # Check that the maximum error is within appropriate bounds
        assert inner_max_error <= 1e-7

        # Note that the sigmoid inverse is quite unstable around the boundary
        assert outer_max_error <= 1e-3

    def test_sigmoid_inverse_invalid_range(self):
        """
        The range of sigmoid is [0, 1], therefore we shouldn't be
        able to invert anything outside that range.
        """

        with pytest.raises(ValueError):
            sigmoid_inverse(-1.)

        with pytest.raises(ValueError):
            sigmoid_inverse(-1e-7)

        with pytest.raises(ValueError):
            sigmoid_inverse(1 + 1e-7)

        with pytest.raises(ValueError):
            sigmoid_inverse(10.)

    def test_sigmoid_inverse_invalid_input(self):
        """
        sigmoid inverse only takes float types
        :return:
        """

        with pytest.raises(TypeError):
            sigmoid_inverse(1)

        with pytest.raises(TypeError):
            sigmoid_inverse("haha")


class TestVariables:
    pass
