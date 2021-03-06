from collections.abc import Iterable
from inspect import signature

import tensorflow as tf
import tensorflow_probability as tfp

from numpy import finfo, float64

from .variables import AbstractVariable

__all__ = [
    "minimize",
    "get_reparametrizations",
    "OptimizationError"
]

AVAILABLE_OPTIMIZERS = {
    "l-bfgs": tfp.optimizer.lbfgs_minimize,
    "bfgs": tfp.optimizer.bfgs_minimize
}


class OptimizationError(Exception):
    pass


def minimize(function,
             vs,
             explicit=True,
             num_correction_pairs=10,
             tolerance=1e-05,
             x_tolerance=0,
             f_relative_tolerance=1e7,
             initial_inverse_hessian_estimate=None,
             max_iterations=1000,
             parallel_iterations=1,
             optimizer="l-bfgs",
             trace=False,
             logger=print):
    """
    Takes a function whose arguments are subclasses of boa.core.AbstractVariable,
    and performs some form of BFGS on it.
    :param function:
    :param vs: Structure of NTFO variables
    :param explicit: If True, `function` must have the signature `function(*vs)`.
    If False, `function` must have the signature `function()`
    :param num_correction_pairs:
    :param tolerance:
    :param x_tolerance:
    :param f_relative_tolerance:
    :param initial_inverse_hessian_estimate:
    :param max_iterations:
    :param parallel_iterations:
    :param logger:
    :param trace:
    :param optimizer:
    :return:
    """

    if optimizer not in AVAILABLE_OPTIMIZERS:
        raise OptimizationError(f"Specified optimizer ({optimizer}) must be one of {AVAILABLE_OPTIMIZERS}!")

    if not isinstance(vs, Iterable):
        raise OptimizationError(f"Variables passed must be in an iterable structure!")

    optimizer = AVAILABLE_OPTIMIZERS[optimizer]

    float64_machine_eps = finfo(float64).eps

    # Check if the function and the passed arguments are compatible
    num_function_params = len(signature(function).parameters)
    num_args = len(vs)

    if explicit and num_function_params != num_args:
        raise OptimizationError(f"Optimization target takes {num_function_params} argument(s) " \
                                f"but {num_args} were given!")

    # These are chosen to match the parameters of
    # scipy.optimizer.fmin_l_bfgs_b
    optimizer_args = {"num_correction_pairs": num_correction_pairs,
                      "tolerance": tolerance,  # This is pgtol in scipy
                      "x_tolerance": x_tolerance,

                      # This is eps * factr in scipy
                      "f_relative_tolerance": float64_machine_eps * f_relative_tolerance,
                      "initial_inverse_hessian_estimate": initial_inverse_hessian_estimate,
                      "max_iterations": max_iterations,
                      "parallel_iterations": parallel_iterations}

    # Get the reparameterization of the
    reparameterizations = get_reparametrizations(vs)

    initial_position, bounds, shapes = recursive_flatten(reparameterizations)

    def unflatten(xs):
        return _recursive_unflatten(xs, bounds, shapes)

    # Pull-back of the function to the unconstrained domain:
    # Reparameterize the function such that instead of taking its original bounded
    # arguments, it takes the unconstrained ones, and they get forward transformed.
    def reparameterized_function(*args):
        new_args = recursive_forward_transform(args, vs)
        return function(*new_args)

    def fn_with_grads(x, first_run=False):

        if explicit:
            # Get back the original arguments
            args = unflatten(x)

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(args)
                value = reparameterized_function(*args)

            gradients = tape.gradient(value, args)
        else:
            # If we're performing implicit optimization, we assign the parameter
            # values to the watched variables at the start
            values = unflatten(x)
            recursive_assign(reparameterizations, values)

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(reparameterizations)
                value = function()

            gradients = tape.gradient(value, reparameterizations)

            if first_run:
                for grad in gradients:
                    if grad is None:
                        raise OptimizationError("Given function does not depend on some of the given variables!")

        # We must concatenate the gradients because lbfgs_minimize expects a single vector
        gradients, _, _ = recursive_flatten(gradients)

        return value, gradients

    # Dry run to check if the optimization can be performed
    fn_with_grads(initial_position, first_run=True)

    optimizer_results = optimizer(fn_with_grads,
                                  initial_position=initial_position,
                                  **optimizer_args)

    if trace:
        logger(f"Optimizer evaluated the objective {optimizer_results.num_objective_evaluations.numpy()} times!")
        logger(f"Optimizer terminated after "
               f"{optimizer_results.num_iterations.numpy()}/{max_iterations} iterations!")
        logger(f"Optimizer converged: {optimizer_results.converged.numpy()}")
        logger(f"Optimizer diverged: {optimizer_results.failed.numpy()}")

    optimum = unflatten(optimizer_results.position)

    # Assign the results to the variables
    recursive_assign(reparameterizations, optimum)

    # Return the loss
    return optimizer_results.objective_value, optimizer_results.converged, optimizer_results.failed


def recursive_assign(vs, vals):
    for v, val in zip(vs, vals):
        if isinstance(v, tf.Variable):
            v.assign(val)

        elif isinstance(v, Iterable):
            recursive_assign(v, val)

        else:
            raise OptimizationError(f"v was of type {type(v)} in recursive_assign!")


def recursive_flatten(xs):
    res, _, bounds, shapes = _recursive_flatten(xs, 0)

    return tf.concat(res, axis=0), bounds, shapes


def _recursive_flatten(xs, index):
    res = []
    bounds = []
    shapes = []

    for x in xs:
        if isinstance(x, (tf.Variable, tf.Tensor)):

            flat = tf.reshape(x, [-1])
            res.append(flat)

            shapes.append(x.shape)

            size = tf.size(x).numpy()

            bounds.append((index, index + size))

            index += size

        elif isinstance(x, Iterable):
            sub_res, sub_ind, sub_bounds, sub_shapes = _recursive_flatten(x, index)

            res += sub_res
            index = sub_ind
            bounds.append(sub_bounds)
            shapes.append(sub_shapes)

        else:
            raise OptimizationError(f"Invalid type of argument was supplied to recursive_flatten: {type(x)}")

    return res, index, bounds, shapes


def _recursive_unflatten(x, bounds, shapes):
    res = []

    for bound, shape in zip(bounds, shapes):

        if isinstance(bound, tuple):
            low, high = bound

            res.append(tf.reshape(x[low:high], shape))

        else:
            sub_res = _recursive_unflatten(x, bound, shape)

            res.append(sub_res)

    return res


def recursive_forward_transform(args, vs):
    new_args = []

    for arg, v in zip(args, vs):
        if issubclass(v.__class__, AbstractVariable):
            new_args.append(v.forward_transform(arg))
        else:
            new_args.append(recursive_forward_transform(arg, v))

    return new_args


def get_reparametrizations(vars, flatten=False):
    """
    Returns the list of reparameterizations for a list of AbstractVariables. Useful to pass to
    tf.GradientTape.watch
    :param bounded_vars:
    :return:
    """

    res = []
    for v in vars:
        if issubclass(v.__class__, AbstractVariable):
            res.append(v.var)

        elif isinstance(v, Iterable):
            reparams = get_reparametrizations(v)

            if flatten:
                res += reparams
            else:
                res.append(reparams)

        else:
            raise OptimizationError(f"Item had invalid type: {type(v)} in {vars}")

    return res
