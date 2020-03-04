import abc
from collections.abc import Iterable
import tensorflow as tf


class VariableError(Exception):
    """
    Base error thrown by modules in the core
    """


def sigmoid_inverse(x):
    if tf.reduce_any(x < 0.) or tf.reduce_any(x > 1.):
        raise ValueError(f"x = {x} was not in the sigmoid function's range ([0, 1])!")
    x = tf.clip_by_value(x, 1e-10, 1 - 1e-10)

    return -tf.math.log(1. / x - 1.)


class AbstractVariable(tf.Module, abc.ABC):
    """
    Wrapper class for a Tensorflow variable. It enables solving constrained optimization
    problems using gradient-based methods by smoothly reparameterizing them.
    """

    def __init__(self, init,
                 dtype=tf.float64,
                 name="abstract_variable",
                 **kwargs):

        super().__init__(name=name,
                         **kwargs)

        self.dtype = dtype

        prepared = self._prepare_value(init)
        self.var = tf.Variable(self.backward_transform(prepared),
                               dtype=self.dtype)

    @tf.Module.with_name_scope
    @abc.abstractmethod
    def forward_transform(self, x):
        pass

    @tf.Module.with_name_scope
    @abc.abstractmethod
    def backward_transform(self, x):
        pass

    @tf.Module.with_name_scope
    def assign(self, x):
        self.var.assign(self.backward_transform(x))

    @tf.Module.with_name_scope
    def __call__(self):
        return self.forward_transform(self.var)

    @tf.Module.with_name_scope
    @abc.abstractmethod
    def valid_range(self):
        """
        Returns a tuple of the upper and lower bounds for the
        allowed values of the variable

        :return: (lower, upper) - tuple of lower and upper bounds
        """
        pass

    @tf.Module.with_name_scope
    def _prepare_value(self, x):
        """
        Attempts basic conversion and casting to the appropriate TF 2.0
        tensor.
        :param x: Value we are preparing
        :returns: Prepared value
        """

        x = tf.convert_to_tensor(x)
        x = tf.cast(x, self.dtype)

        return x


class UnconstrainedVariable(AbstractVariable):
    """
    Wrapper class for a Tensorflow 2.0 variable. Does no reparameterization.
    """

    def __init__(self, init,
                 dtype=tf.float64,
                 name="unconstrained_variable",
                 **kwargs):

        super().__init__(init=init,
                         dtype=dtype,
                         name=name,
                         **kwargs)

    @tf.Module.with_name_scope
    def forward_transform(self, x):
        return x

    @tf.Module.with_name_scope
    def backward_transform(self, x):
        return x

    @tf.Module.with_name_scope
    def valid_range(self):
        inf = tf.cast(float("inf"), self.dtype)

        return -inf, inf


class PositiveVariable(AbstractVariable):

    def __init__(self, init,
                 dtype=tf.float64,
                 name="positive_variable",
                 **kwargs):

        super().__init__(init=init,
                         dtype=dtype,
                         name=name,
                         **kwargs)

    @tf.Module.with_name_scope
    def forward_transform(self, x):
        pass

    @tf.Module.with_name_scope
    def backward_transform(self, x):
        pass

    @tf.Module.with_name_scope
    def valid_range(self):
        inf = tf.cast(float("inf"), self.dtype)
        zero = tf.cast(0, self.dtype)

        return zero, inf


class BoundedVariable(AbstractVariable):
    """
    Wrapper class for a Tensorflow variable. It enables solving constrained optimization
    problems using gradient-based methods by smoothly reparameterizing an unconstrained variable through a
    sigmoid transformation.
    """
    def __init__(self,
                 init,
                 lower,
                 upper,
                 dtype=tf.float64,
                 name="bounded_variable",
                 **kwargs):

        super(BoundedVariable, self).__init__(init=init,
                                              dtype=dtype,
                                              name=name,
                                              **kwargs)

        self.lower = tf.convert_to_tensor(lower, dtype=self.dtype)
        self.upper = tf.convert_to_tensor(upper, dtype=self.dtype)

    @tf.Module.with_name_scope
    def forward_transform(self, x):
        """
        Go from unconstrained domain to constrained domain
        :param x: tensor to be transformed in the unconstrained domain
        :return: tensor in the constrained domain
        """
        x = tf.convert_to_tensor(x, dtype=self.dtype)
        return (self.upper - self.lower) * tf.nn.sigmoid(x) + self.lower

    @tf.Module.with_name_scope
    def backward_transform(self, x, eps=1e-12):
        """
        Go from constrained domain to unconstrained domain
        :param x: tensor to be transformed in the constrained domain
        :return: tensor in the unconstrained domain
        """
        x = tf.convert_to_tensor(x, dtype=self.dtype)
        return sigmoid_inverse((x - self.lower) / (self.upper - self.lower + eps))

    @tf.Module.with_name_scope
    def valid_range(self):
        return self.lower, self.upper

    # TODO: Maybe there is a nicer way of doing this
    @staticmethod
    def get_all(bounded_vars):
        """
        Get the forward transforms of all given bounded variables
        :param bounded_vars:
        :return:
        """

        res = []
        for bv in bounded_vars:
            if isinstance(bv, BoundedVariable):
                res.append(bv())

            elif isinstance(bv, Iterable):
                res.append(BoundedVariable.get_all(bv))

        return res

    @staticmethod
    def get_reparametrizations(bounded_vars, flatten=False):
        """
        Returns the list of reparameterizations for a list of BoundedVariables. Useful to pass to
        tf.GradientTape.watch
        :param bounded_vars:
        :return:
        """

        res = []
        for bv in bounded_vars:
            if isinstance(bv, BoundedVariable):
                res.append(bv.reparameterization)

            elif isinstance(bv, Iterable):

                reparams = BoundedVariable.get_reparametrizations(bv)

                if flatten:
                    res += reparams
                else:
                    res.append(reparams)

        return res
