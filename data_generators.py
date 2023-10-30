import numpy as np
import tensorflow as tf
from options import (
    EuropeanOption,
    EuropeanBasketOption,
    EuropeanSwap,
    TimeEuropean,
    TimeEuropeanBasketOption,
    InterestRateSwap,
    ZeroCouponBond,
    InterestRateSwaption,
)
import math

european_list = [
    EuropeanOption,
    EuropeanBasketOption,
    EuropeanSwap,
    TimeEuropean,
    TimeEuropeanBasketOption,
    InterestRateSwap,
    ZeroCouponBond,
    InterestRateSwaption,
]


class BaseGenerator(tf.keras.utils.Sequence):
    """Create batches of random points for the network training."""

    def __init__(self, sde, config, option, N: int = 100):
        """Initialise the generator by saving the batch size."""
        self.config = config
        self.batch_size = config.batch_size
        self.option = option
        self.sde = sde
        self.params_model = self.sde.sample_parameters(N)
        self.params_option = self.option.sample_parameters(N)
        self.time_steps = self.config.time_steps + 1
        time_stamp = tf.range(0, self.config.T + self.config.dt, self.config.dt)
        # time_stamp = tf.linspace(0, self.config.T, self.time_steps)
        time_stamp = tf.reshape(time_stamp, [1, 1, self.time_steps, 1])
        self.time_stamp = tf.tile(
            time_stamp, [self.config.batch_size, self.config.sample_size, 1, 1]
        )

    def __len__(self):
        """Describes the number of points to create"""
        return math.ceil(len(self.params_model) / self.batch_size)

    def __getitem__(self, idx: int):
        raise NotImplementedError


class DiffusionModelGenerator(BaseGenerator):
    """Create batches of random points for the network training."""

    def __init__(self, sde, config, option, N: int):
        super(DiffusionModelGenerator, self).__init__(sde, config, option, N)

    def __getitem__(self, idx: int):
        """
        Get one batch of random points in the interior of the domain to
        train the PDE residual and with initial time to train the initial value.
        for the parameter, we always let the risk neutral drift r be at the first entry.
        """
        params_batch_model = self.params_model[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        params_batch_option = self.params_option[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        x, dw = self.sde.sde_simulation(params_batch_model, self.config.sample_size)
        if type(self.option) not in european_list:
            markov_var = self.option.markovian_var(x)
            x = tf.concat([x, markov_var], axis=-1)  # x contains markov variable
        model_param = self.sde.expand_batch_inputs_dim(params_batch_model)
        option_param = self.option.expand_batch_inputs_dim(params_batch_option)
        u_hat = tf.concat([model_param, option_param], -1)
        t = self.time_stamp
        data = t, x, dw, u_hat
        return (data,)
