import pytest
import numpy as np
import tensorflow as tf

from not_tf_opt import sigmoid_inverse, AbstractVariable, UnconstrainedVariable, PositiveVariable, BoundedVariable
from .utils import assert_near

inf = tf.cast(float("inf"), tf.float64)


class TestSigmoidInverse:
    def test_sigmoid_inverse(self):

        # Define some test inputs
        test_xs = tf.linspace(1e-10, 1 - 1e-10, 1000)
        test_reparam_xs = tf.linspace(-9., 9., 1000)

        inner_composition = tf.nn.sigmoid(sigmoid_inverse(test_xs))
        outer_composition = sigmoid_inverse(tf.nn.sigmoid(test_reparam_xs))

        print(tf.reduce_max(tf.abs(outer_composition - test_reparam_xs)))

        # Check that the maximum error is within appropriate bounds
        assert_near(inner_composition, test_xs)
        assert_near(outer_composition, test_reparam_xs, atol=1e-3, rtol=1e-5)

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

    dtype = tf.float64

    @pytest.mark.parametrize(
        'var, init, assign',
        [(UnconstrainedVariable, 5, np.pi),
         (UnconstrainedVariable, tf.zeros(5, dtype=tf.float64), tf.ones(5, dtype=tf.float64)),
         (UnconstrainedVariable, np.ones([5, 3], dtype=np.float64), 0.1 * np.ones([5, 3], dtype=np.float64)),
         (PositiveVariable, 5, np.pi),
         (PositiveVariable, tf.zeros(5, dtype=tf.float64) + 0.1, tf.ones(5, dtype=tf.float64)),
         (PositiveVariable, np.ones([5, 3], dtype=np.float64), 0.1 * np.ones([5, 3], dtype=np.float64)),
         (lambda x: BoundedVariable(x, 0, 3), np.pi - 1, np.pi - 2),
         (lambda x: BoundedVariable(x, 0, 3), tf.zeros(5, dtype=tf.float64) + 0.1, tf.ones(5, dtype=tf.float64)),
         (lambda x: BoundedVariable(x, 0, 3), np.ones([5, 3], dtype=np.float64), 0.1 * np.ones([5, 3], dtype=np.float64)),
         ]
    )
    def test_init_and_assign(self, var, init, assign):
        v = var(init)
        assert_near(v(), init, 1e-7)

        v.assign(assign)
        assert_near(v(), assign, 1e-7)

    @pytest.mark.parametrize(
        'var, valid_range',
        [(UnconstrainedVariable, (-inf, inf)),
         (PositiveVariable, (0, inf)),
         (lambda x: BoundedVariable(x, -9, 10), (-9, 10))
         ]
    )
    def test_valid_ranges(self, var, valid_range):
        v = var(5)
        assert v.valid_range() == valid_range
