import pytest

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from not_tf_opt.variables import UnconstrainedVariable, PositiveVariable, BoundedVariable
import not_tf_opt.optimize as ntfo
from .utils import assert_near


def test_minimize_arguments():
    # The variables passed to ntfo.minimize must always be in an iterable!
    with pytest.raises(ntfo.OptimizationError):
        ntfo.minimize(lambda x: x,
                      vs=UnconstrainedVariable(1.))

    # Optimizer must be in AVAILABLE_OPTIMIZERS
    with pytest.raises(ntfo.OptimizationError):
        ntfo.minimize(lambda x: x,
                      vs=[UnconstrainedVariable(1.)],
                      optimizer="blabla")


@pytest.mark.parametrize(
    'variable',
    [UnconstrainedVariable,
     PositiveVariable,
     lambda x: BoundedVariable(x, -100., 100.)]
)
def test_high_dimensional_minimize(variable):
    """
    Example taken from
    https://www.tensorflow.org/probability/api_docs/python/tfp/optimizer/lbfgs_minimize

    """

    # A high-dimensional quadratic bowl.
    ndims = 60
    minimum = 10 * tf.ones([ndims], dtype=tf.float64)
    scales = tf.range(ndims, dtype=tf.float64) + 1.0

    def quadratic_loss(x):
        return tf.reduce_sum(scales * tf.math.squared_difference(x, minimum), axis=-1)

    init = tf.ones([ndims], dtype=tf.float64)
    v = variable(init)

    result, converged, diverged = ntfo.minimize(quadratic_loss,
                                                vs=[v],
                                                tolerance=1e-8)

    # Check that the search converged
    assert converged

    # Check that the solution is good
    assert_near(v(), minimum, atol=1e-8, rtol=1e-8)


def test_positive_minimize():
    """
    Checks optimization of PositiveVariable
    """

    def get_loss(mean):
        return lambda x: (x - mean) ** 2

    loss_pos = get_loss(5.)
    loss_neg = get_loss(-5.)

    v1 = PositiveVariable(10.)
    v2 = PositiveVariable(10.)

    # -------------------------------------------------------------------------
    # Test the case when the minimum is within the constrained domain
    # -------------------------------------------------------------------------
    _, converged, _ = ntfo.minimize(loss_pos,
                                    vs=[v1],
                                    tolerance=1e-8)

    # Check that the search converged
    assert converged

    # Check that the solution is good
    assert_near(v1(), 5., atol=1e-8, rtol=1e-8)

    # -------------------------------------------------------------------------
    # Test the case when the minimum is NOT within the constrained domain
    # -------------------------------------------------------------------------
    _, converged, _ = ntfo.minimize(loss_neg,
                                    vs=[v2],
                                    tolerance=1e-8)

    # Check that the search converged
    assert converged

    # Check that the solution is good:
    # The closest we can get to -5 is 0!
    assert_near(v2(), 0., atol=1e-8, rtol=1e-8)


def test_bounded_minimize():
    """
    Checks optimization of BoundedVariable
    """

    def get_loss(mean):
        return lambda x: (x - mean) ** 2

    loss_pos = get_loss(5.)
    loss_0 = get_loss(0.)
    loss_neg = get_loss(-5.)

    v1 = BoundedVariable(np.pi - 1, -3, 3)
    v2 = BoundedVariable(np.pi - 1, -3, 3)
    v3 = BoundedVariable(np.pi - 1, -3, 3)

    # -------------------------------------------------------------------------
    # Test the case when the minimum is within the constrained domain
    # -------------------------------------------------------------------------
    _, converged, _ = ntfo.minimize(loss_0,
                                    vs=[v1],
                                    tolerance=1e-8)

    # Check that the search converged
    assert converged

    # Check that the solution is good
    assert_near(v1(), 0., atol=1e-8, rtol=1e-8)

    # -------------------------------------------------------------------------
    # Test the case when the minimum is BELOW the constrained range
    # -------------------------------------------------------------------------
    _, converged, _ = ntfo.minimize(loss_neg,
                                    vs=[v2],
                                    tolerance=1e-8)

    # Check that the search converged
    assert converged

    # Check that the solution is good:
    # The closest we can get to -5 is -3!
    assert_near(v2(), -3, atol=1e-8, rtol=1e-8)

    # -------------------------------------------------------------------------
    # Test the case when the minimum is ABOVE the constrained range
    # -------------------------------------------------------------------------
    _, converged, _ = ntfo.minimize(loss_pos,
                                    vs=[v3],
                                    tolerance=1e-8)

    # Check that the search converged
    assert converged

    # Check that the solution is good:
    # The closest we can get to 5 is 3!
    assert_near(v3(), 3, atol=1e-8, rtol=1e-8)

