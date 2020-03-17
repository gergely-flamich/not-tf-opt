import pytest

import tensorflow as tf
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


def test_nested_optimization():
    """
    Checks if the optimizer runs correctly when the
    arguments are given in a nested tuple structure.
    """

    def get_nested_loss(mean):

        def loss(x, xs, xss):

            x = x + tf.reduce_mean(xs)

            for xs_ in xss:
                x = x + tf.reduce_sum(xs_)

            return (x - mean) ** 2

        return loss

    # Fix random seed
    tf.random.set_seed(42)

    # Get loss function
    loss_fn = get_nested_loss(0.)

    # Define variables
    x = BoundedVariable(tf.random.uniform(shape=()), -3., 3.)
    xs = [BoundedVariable(tf.random.uniform(shape=(4,)), 0., 1.) for _ in range(3)]
    xss = [[BoundedVariable(tf.random.uniform(shape=(4,)), -1., 1.) for _ in range(3)]
           for _ in range(2)]

    total_loss, converged, _ = ntfo.minimize(loss_fn,
                                             vs=[x, xs, xss])

    assert converged

    # Check if the result is good enough
    assert_near(total_loss, 0.)


def test_invalid_shallow_optimization_arguments():
    """
    Checks if the appropriate exception is raised if
    we provide invalid optimization target variables in a tuple.
    """

    def get_loss(mean):
        return lambda x: (x - mean) ** 2

    loss = get_loss(3.)

    with pytest.raises(ntfo.OptimizationError):
        ntfo.minimize(loss,
                      vs=[None])


def test_function_arguments_mismatch():
    """
    Test if the arguments given can be passed into the function
    """

    def get_loss(mean):
        return lambda x: (x - mean) ** 2

    loss = get_loss(0.)

    v1 = PositiveVariable(5.)
    v2 = BoundedVariable(4, -3, 10)

    # Check if we get an error if we have too few arguments passed
    with pytest.raises(ntfo.OptimizationError) as exception:
        ntfo.minimize(loss,
                      vs=[])

    assert str(exception.value) == "Optimization target takes 1 argument(s) " \
                                   "but 0 were given!"

    # Check if we get an error if we have too many arguments passed
    with pytest.raises(ntfo.OptimizationError) as exception:
        ntfo.minimize(loss,
                      vs=[v1, v2])

    assert str(exception.value) == "Optimization target takes 1 argument(s) " \
                                   "but 2 were given!"


def test_invalid_nested_optimization():
    """
    Checks if the appropriate exception is raised if we provide
    invalid optimization targets in a nested tuple structure.
    """

    def get_nested_loss(mean):

        def loss(x, xs, xss):

            x = x + tf.reduce_mean(xs)

            for xs_ in xss:
                x = x + tf.reduce_sum(xs_)

            return (x - mean) ** 2

        return loss

    loss = get_nested_loss(0.)

    v = PositiveVariable(5.)
    vs = [None, "haha"]
    vss = ["Noone", 12.]

    with pytest.raises(ntfo.OptimizationError):
        ntfo.minimize(loss,
                      vs=[v, vs, vss])


def test_mixed_optimization():
    """
    Checks if variables with mixed constraints are optimized correctly together.
    """

    def get_loss(mean):

        def loss(x, y, z):
            return (x + y + z - mean) ** 2

        return loss

    # Fix seed
    tf.random.set_seed(42)

    # Create variables
    v1 = UnconstrainedVariable(tf.random.uniform(shape=()))
    v2 = PositiveVariable(1 + 3 * tf.random.uniform(shape=()))
    v3 = BoundedVariable(3 + tf.random.uniform(shape=()), lower=3, upper=4)

    # Get loss
    loss = get_loss(0.)

    total_loss, converged, _ = ntfo.minimize(loss,
                                             vs=[v1, v2, v3])

    # Test if the optimization converges
    assert converged

    assert_near(total_loss, 0.)


def test_implicit_optimization():
    """
    Checks if the optimizer works correctly in the case when the variables are
    not passed explicitly to the function.
    """

    def get_implicit_loss(mean, var):

        def loss():
            return (var() - mean) ** 2

        return loss

    v = BoundedVariable(9,
                        lower=-3,
                        upper=10)

    implicit_loss = get_implicit_loss(2, v)

    _, converged, _ = ntfo.minimize(implicit_loss,
                                    vs=[v],
                                    explicit=False)

    assert converged

    # Check if the solution is good
    assert_near(v(), 2., atol=1e-5, rtol=1e-5)


def test_implicit_optimization_dependency_check():
    """
    Checks if the optimizer can detect if the function doesn't depend on the
    specified targets.
    """

    def get_implicit_loss(mean, v):

        def loss():
            return (v() - mean) ** 2

        return loss

    v = BoundedVariable(1, 0, 3)
    v2 = BoundedVariable(3, 0, 10)

    implicit_loss = get_implicit_loss(1, v)

    with pytest.raises(ntfo.OptimizationError) as error:
        ntfo.minimize(implicit_loss,
                      vs=[v2],
                      explicit=False)

    assert "Given function does not depend on some of the given variables!" in str(error.value)

