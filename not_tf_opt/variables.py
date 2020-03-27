import abc
import tensorflow as tf
import tensorflow_probability as tfp

from numpy import float32, float64, finfo

from .utils import map_to_bounded_interval, map_from_bounded_interval, VariableError

__all__ = [
    "AbstractVariable",
    "UnconstrainedVariable",
    "PositiveVariable",
    "BoundedVariable",
]

eps_f32 = finfo(float32).eps
eps_f64 = finfo(float64).eps


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
                               dtype=self.dtype,
                               name=f"{name}/reparametrization")

    @tf.Module.with_name_scope
    @abc.abstractmethod
    def forward_transform(self, x):
        """
        Maps the argument from the unconstrained domain to the constrained domain.
        :param x:
        :return:
        """

    @tf.Module.with_name_scope
    @abc.abstractmethod
    def backward_transform(self, x):
        """
        Maps the argument from the constrained domain to the unconstrained domain.

        :param x:
        :return:
        """

    @tf.Module.with_name_scope
    def assign(self, x):
        """
        Assigns a value from the constrained domain to the unconstrained reparameterization.
        :param x:
        :return:
        """
        self.var.assign(self.backward_transform(x))

    def assign_var(self, x):

        if not isinstance(x, self.__class__):
            raise VariableError(f"Variable must have type {self.__class__} but had type {type(x)}!")

        self.assign(x())

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
        """
        Forward transform using softplus:

        c(x) = log(1 + e^x)

        :param x:
        :return:
        """

        x = self._prepare_value(x)
        return tf.nn.softplus(x)

    @tf.Module.with_name_scope
    def backward_transform(self, x):
        """
        Backward transform using inverse softplus:

        x(c) = log(e^c - 1)

        :param x:
        :return:
        """

        if tf.reduce_any(x <= 0):
            raise VariableError(f"All provided values must be positive! (Got x = {x})")

        x = self._prepare_value(x)
        return tfp.math.softplus_inverse(x)

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

        init = tf.convert_to_tensor(init)

        lower = tf.convert_to_tensor(lower)
        lower = tf.cast(lower, dtype)

        upper = tf.convert_to_tensor(upper)
        upper = tf.cast(upper, dtype)

        if lower.shape == ():
            lower = tf.ones(init.shape, dtype=dtype) * lower

        if upper.shape == ():
            upper = tf.ones(init.shape, dtype=dtype) * upper

        self._lower = tf.Variable(lower, dtype=dtype, name="lower_bound")
        self._upper = tf.Variable(upper, dtype=dtype, name="upper_bound")

        super(BoundedVariable, self).__init__(init=init,
                                              dtype=dtype,
                                              name=name,
                                              **kwargs)

        if tf.reduce_any(self.lower >= self.upper):
            raise VariableError(f"Lower bound {self.lower} must be less than upper bound {self.upper}!")

        if tf.reduce_any(self.lower >= self()) or tf.reduce_any(self.upper <= self()):
            raise VariableError("Initialization value must be between given lower and upper bounds!")

    @property
    def lower(self):
        return self._lower.value()

    @lower.setter
    def lower(self, x):
        x = self._prepare_value(x)
        if tf.reduce_any(x >= self.upper):
            raise VariableError(f"New lower bound {x} must be less than upper bound {self.upper}!")

        if x.shape == ():
            x = tf.ones(self.var.shape, dtype=self.dtype) * x

        self._lower.assign(x)

    @property
    def upper(self):
        return self._upper.value()

    @upper.setter
    def upper(self, x):
        x = self._prepare_value(x)

        if tf.reduce_any(self.lower >= x):
            raise VariableError(f"Lower bound {self.lower} must be less than new upper bound {x}!")

        if x.shape == ():
            x = tf.ones(self.var.shape, dtype=self.dtype) * x

        self._upper.assign(x)

    @tf.Module.with_name_scope
    def forward_transform(self, x):
        """
        Go from unconstrained domain to constrained domain
        :param x: tensor to be transformed in the unconstrained domain
        :return: tensor in the constrained domain
        """
        x = self._prepare_value(x)
        return map_to_bounded_interval(x, self.lower, self.upper)

    @tf.Module.with_name_scope
    def backward_transform(self, x, eps=1e-12):
        """
        Go from constrained domain to unconstrained domain
        :param x: tensor to be transformed in the constrained domain
        :return: tensor in the unconstrained domain
        """
        x = self._prepare_value(x)

        return map_from_bounded_interval(x, self.lower, self.upper)

    @tf.Module.with_name_scope
    def valid_range(self):
        return self.lower.numpy(), self.upper.numpy()

    def assign_var(self, x):

        if not isinstance(x, self.__class__):
            raise VariableError(f"Variable must have type {self.__class__} but had type {type(x)}!")

        self.lower = x.lower
        self.upper = x.upper

        self.assign(x())
