import tensorflow as tf
from options import (
    EuropeanOption,
    GeometricAsian,
    LookbackOption,
    EuropeanSwap,
    InterestRateSwap,
)
from sde import GeometricBrownianMotion, HullWhiteModel
import json
import munch


sde_list = ["GBM", "TGBM", "SV", "HW", "SVJ"]
option_list = [
    "European",
    "EuropeanPut",
    "Lookback",
    "Asian",
    "Basket",
    "BasketnoPI",
    "Swap",
    "Swaption",
    "TimeEuropean",
    "BermudanPut",
]
dim_list = [1, 3, 5, 10, 20]


def load_config(sde_name: str, option_name: str, dim: int = 1):
    """
    This function is for introducing config json files into test functions
    """
    if (
        (sde_name not in sde_list)
        or (option_name not in option_list)
        or (dim not in dim_list)
    ):
        raise ValueError(
            f"please input right sde_name in {sde_list},\
                          option_name in {option_list} and dim in {dim_list}"
        )
    else:
        json_path = f"./configs/{sde_name}_{option_name}_{dim}.json"
    with open(json_path) as json_data_file:
        config = json.load(json_data_file)

    config = munch.munchify(config)
    return config


class TestEuropeanOption(tf.test.TestCase):
    """
    This test is for Vanilla option
    """
    def testPayoff(self):
        config = load_config("GBM", "European", 1)
        opt = EuropeanOption(config)
        t = tf.random.uniform([3, 5, 6, 2]) 
        x = tf.random.uniform([3, 5, 6, 2]) + 0.1
        u_hat = tf.random.uniform([3, 5, 6, 4])
        payoff = opt.payoff(t, x, u_hat)
        self.assertEqual(payoff.shape, [3, 5, 1])

    def testExactprice(self):
        config = load_config("GBM", "European", 1)
        opt = EuropeanOption(config)
        sde = GeometricBrownianMotion(config)
        u_sde = sde.sample_parameters(1)
        u_opt = opt.sample_parameters(1)
        u_hat = tf.concat([u_sde, u_opt], axis=-1)
        x, _ = sde.sde_simulation(u_hat, config.eqn_config.sample_size)
        time_stamp = tf.range(
            0, config.eqn_config.T + config.eqn_config.dt, config.eqn_config.dt
        )
        time_steps = int(config.eqn_config.T / config.eqn_config.dt) + 1
        time_stamp = tf.reshape(time_stamp, [1, 1, time_steps, 1])
        t = tf.tile(time_stamp, [u_hat.shape[0], config.eqn_config.sample_size, 1, 1])
        u_hat = sde.expand_batch_inputs_dim(u_hat)
        price = opt.exact_price(t, x, u_hat)
        strike = tf.expand_dims(u_hat[:, :, -1, -1], axis=-1)
        exact_terminal_payoff = tf.nn.relu(x[:, :, -1, :] - strike)
        self.assertAllLessEqual(
            tf.reduce_mean(tf.abs(price[:, :, -1, :] - exact_terminal_payoff)), 0.01
        )

    def testSampleparameters(self):
        config = load_config("GBM", "European", 1)
        opt = EuropeanOption(config)
        strike_params = opt.strike_range
        parameters = opt.sample_parameters()
        empirical_mean = tf.reduce_mean(parameters)
        self.assertAllLessEqual(
            tf.abs(empirical_mean - strike_params[0]), 0.01
        )


class TestGeoAsianOption(tf.test.TestCase):
    """
    This test is for Asian option
    """
    def testMarkovVar(self):
        config = load_config("GBM", "European", 1)
        opt = GeometricAsian(config)

        x = tf.constant(
            [
                [[0.2, 1, 3, 0.8], [0.7, 4, 0.8, 3]],
                [[0.2, 1, 3, 0.8], [0.7, 4, 0.8, 3]],
                [[0.2, 1, 3, 0.8], [0.7, 4, 0.8, 3]],
            ]
        )

        g = tf.constant(
            [
                [[0.2, 0.4472, 0.84343, 0.83236], [0.7, 1.6733, 1.3084, 1.61006]],
                [[0.2, 0.4472, 0.84343, 0.83236], [0.7, 1.6733, 1.3084, 1.61006]],
                [[0.2, 0.4472, 0.84343, 0.83236], [0.7, 1.6733, 1.3084, 1.61006]],
            ]
        )

        x = tf.expand_dims(x, axis=-1)
        g = tf.expand_dims(g, axis=-1)
        g_hat = opt.markovian_var(x)
        x_test = tf.random.uniform([3, 5, 6, 1]) + 0.1
        g_test = opt.markovian_var(x_test)
        dt = config.eqn_config.dt
        for i in range(5):
            geo_avg = tf.exp(
                tf.reduce_sum(tf.math.log(x_test[:, :, : i + 1, :]), axis=2)
                * dt
                / ((i + 1) * dt)
            )
            self.assertAllLessEqual(
                tf.reduce_mean(tf.math.abs(g_test[:, :, i, :] - geo_avg)), 1e-6
            )
        self.assertAllEqual(g_test.shape, x_test.shape, msg="shape is not same")
        self.assertAllLessEqual(tf.reduce_mean(tf.math.abs(g_hat - g)), 1e-4)

    def testPayoff(self):
        config = load_config("GBM", "European", 1)
        opt = GeometricAsian(config)
        t = tf.random.uniform([3, 5, 6, 1]) + 0.1
        x = tf.random.uniform([3, 5, 6, 1]) + 0.1
        u_hat = tf.random.uniform([3, 5, 6, 4])
        payoff = opt.payoff(t, x, u_hat)
        self.assertEqual(payoff.shape, [3, 5, 1])


class TestlookbackOption(tf.test.TestCase):
    """
    This test is for lookback option
    """
    def testMarkovVar(self):
        config = load_config("GBM", "European", 1)
        opt = LookbackOption(config)
        x = tf.constant(
            [
                [[2, 1, 3, 0, -1], [7, 4, 8, 3, 2]],
                [[4, 5, 3, 4, 2], [6, 5, 23, 8, 1]],
                [[2, 4, 1, -7, -78], [6, 5, 7, 8, 9]],
            ]
        )

        m = tf.constant(
            [
                [[2, 1, 1, 0, -1], [7, 4, 4, 3, 2]],
                [[4, 4, 3, 3, 2], [6, 5, 5, 5, 1]],
                [[2, 2, 1, -7, -78], [6, 5, 5, 5, 5]],
            ]
        )

        m_hat = opt.markovian_var(tf.expand_dims(x, -1))

        x_test = tf.random.uniform([3, 5, 6, 2])
        m_test = opt.markovian_var(x_test)
        for i in range(5):
            self.assertAllEqual(
                m_test[:, :, i, :],
                tf.reduce_min(x_test[:, :, : i + 1, :], axis=2),
                msg="wrong 1",
            )
        self.assertAllEqual(
            m_test[:, :, -1, :], tf.reduce_min(x_test, axis=2), msg="wrong 2"
        )
        self.assertAllEqual(m_hat, tf.expand_dims(m, -1), msg="wrong 3")
        self.assertAllEqual(m_test.shape, x_test.shape, msg="wrong 4")

    def testPayoff(self):
        config = load_config("GBM", "European", 1)
        opt = LookbackOption(config)
        t = tf.random.uniform([3, 5, 6, 1])
        x = tf.random.uniform([3, 5, 6, 1]) + 0.1
        u_hat = tf.random.uniform([3, 5, 6, 4])
        payoff = opt.payoff(t, x, u_hat)
        self.assertEqual(payoff.shape, [3, 5, 1])
        x_min = tf.reduce_min(x, axis=2)
        self.assertAllEqual(payoff, x[:, :, -1, :] - x_min)

    def testExactprice(self):
        config = load_config("GBM", "European", 1)
        opt = LookbackOption(config)
        sde = GeometricBrownianMotion(config)
        u_hat = sde.sample_parameters(1)
        x, _ = sde.sde_simulation(u_hat, config.eqn_config.sample_size)
        x_m = opt.markovian_var(x)
        arg_x = tf.concat([x, x_m], axis=-1)
        time_stamp = tf.range(
            0, config.eqn_config.T + config.eqn_config.dt, config.eqn_config.dt
        )
        time_steps = int(config.eqn_config.T / config.eqn_config.dt) + 1
        time_stamp = tf.reshape(time_stamp, [1, 1, time_steps, 1])
        t = tf.tile(time_stamp, [u_hat.shape[0], config.eqn_config.sample_size, 1, 1])
        u_hat = sde.expand_batch_inputs_dim(u_hat)
        price = opt.exact_price(t, arg_x, u_hat)
        exact_terminal_payoff = x[:, :, -1, :] - tf.reduce_min(x, 2)
        self.assertAllLessEqual(tf.abs(price[:, :, -1, :] - exact_terminal_payoff), 0.1)


class TestSwap(tf.test.TestCase):
    """
    This test is for swap and forward
    """
    def testPayoff(self):
        config = load_config("GBM", "European", 1)
        opt = EuropeanSwap(config)
        t = tf.random.uniform([3, 5, 6, 1])
        x = tf.random.uniform([3, 5, 6, 1])
        u_hat = tf.zeros_like(x)
        payoff = opt.payoff(t, x, u_hat)
        self.assertEqual(payoff.shape, [3, 5, 1])
        self.assertAllLessEqual(payoff - x[:, :, -1, :], 1e-4)

    def testExactprice(self):
        config = load_config("GBM", "European", 1)
        opt = EuropeanSwap(config)
        sde = GeometricBrownianMotion(config)
        u_sde = sde.sample_parameters(1)
        u_opt = opt.sample_parameters(1)
        u_hat = tf.concat([u_sde, u_opt], axis=-1)
        x, _ = sde.sde_simulation(u_hat, config.eqn_config.sample_size)
        time_stamp = tf.range(
            0, config.eqn_config.T + config.eqn_config.dt, config.eqn_config.dt
        )
        time_steps = int(config.eqn_config.T / config.eqn_config.dt) + 1
        time_stamp = tf.reshape(time_stamp, [1, 1, time_steps, 1])
        t = tf.tile(time_stamp, [u_hat.shape[0], config.eqn_config.sample_size, 1, 1])
        u_hat = sde.expand_batch_inputs_dim(u_hat)
        # print(u_hat)
        price = opt.exact_price(t, x, u_hat)
        strike = tf.expand_dims(u_hat[:, :, -1, -1], axis=-1)
        exact_terminal_payoff = x[:, :, -1, :] - strike
        self.assertAllLessEqual(
            tf.reduce_mean(tf.abs(price[:, :, -1, :] - exact_terminal_payoff)), 0.01
        )


class TestInterestRateSwap(tf.test.TestCase):
    """
    This test is for swap and swaption
    """
    def setUp(self):
        self.config = load_config("HW", "Swaption", 1)
        self.sde = HullWhiteModel(self.config)
        self.option = InterestRateSwap(self.config)
        self.u_hat = tf.constant([[0.045, 0.4, 0.03, 0.011]])
        return super(TestInterestRateSwap, self).setUp()

    def testonefloatleg(self):
        kappa = 0.045
        theta = 0.4
        sigma = 0.03

        def zcp(t, r, T):
            B = (1 - tf.exp(-kappa * (T - t))) / kappa
            A = tf.exp(
                (B - T + t) * (kappa**2 * theta - sigma**2 / 2) / kappa**2
                + (sigma * B) ** 2 / (4 * kappa)
            )
            p = A * tf.exp(-B * tf.reduce_sum(r, axis=-1, keepdims=True))
            return p

        r = tf.ones((1, 1, 1)) * 0.02  # suppose r_t = 0.02
        T_1 = 1.0
        T_2 = 2.0
        for t in [0, 0.5, 1]:
            p1 = self.option.one_float_leg(
                t, r, tf.expand_dims(self.u_hat, axis=1), 2.0
            )
            p2 = zcp(t, r, T_1) - zcp(t, r, T_2)
            self.assertAllLessEqual(tf.reduce_mean(tf.abs(p1 - p2)), 1e-7)

    def testswapvalue(self):
        config1 = load_config("HW", "Swaption", 1)
        option1 = InterestRateSwap(config1)
        option1.reset_dates([1.0, 2.0])
        kappa = 0.045
        theta = 0.4
        sigma = 0.03
        strike = option1.fix_rate

        def zcp(t, r, T):
            B = (1 - tf.exp(-kappa * (T - t))) / kappa
            A = tf.exp(
                (B - T + t) * (kappa**2 * theta - sigma**2 / 2) / kappa**2
                + (sigma * B) ** 2 / (4 * kappa)
            )
            p = A * tf.exp(-B * tf.reduce_sum(r, axis=-1, keepdims=True))
            return p

        r = tf.ones((1, 1, 1)) * 0.02
        for t in [0, 0.5, 1]:
            p1 = option1.swap_value(t, r, tf.expand_dims(self.u_hat, axis=1))
            p2 = (
                strike * zcp(t, r, 2) - (zcp(t, r, 1) - zcp(t, r, 2))
            ) * option1.notional
            self.assertAllLessEqual(tf.reduce_mean(tf.abs(p1 - p2)), 1e-7)


if __name__ == "__main__":
    tf.test.main()
