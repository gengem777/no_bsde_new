import tensorflow as tf
import numpy as np
from sde_jump import GBMwithSimpleJump
from utils import load_config


class TestGBMSimpleJump(tf.test.TestCase):
    """
    This test is for GBM and time dependent GBM
    """

    def test_initial_sampler(self):
        config = load_config("GBMJ", "Swap", 1)
        sde = GBMwithSimpleJump(config)
        u_hat = tf.constant([[0.05, 0.45], [0.04, 0.58]])
        dim = config.eqn_config.dim
        initial_state = sde.initial_sampler(u_hat, 10000)
        self.assertEqual(initial_state.shape, [2, 10000, dim])

    def test_drift(self):
        config = load_config("GBMJ", "Swap", 1)
        sde = GBMwithSimpleJump(config)
        dim = config.eqn_config.dim
        u_hat = tf.constant([[0.05, 0.45, 1.05], [0.05, 0.58, 1.06]])
        state = tf.random.uniform([2, 10, dim])
        drift = sde.drift(0, state, u_hat)
        self.assertAllLessEqual(drift - 0.05 * state, 1e-7)

    def test_diffusion(self):
        config = load_config("GBMJ", "Swap", 1)
        sde = GBMwithSimpleJump(config)
        dim = config.eqn_config.dim
        u_hat = tf.constant([[0.05, 0.45, 1.05], [0.04, 0.45, 1.06]])
        state = tf.random.uniform([2, 10, dim])
        diffusion = sde.diffusion(0, state, u_hat)
        self.assertAllLessEqual(diffusion - 0.45 * state, 1e-7)

    def test_corr_matrix(self):
        config = load_config("GBMJ", "Swap", 1)
        config.eqn_config.dim = 3
        sde = GBMwithSimpleJump(config)
        u_hat = tf.constant([[0.05, 0.45, 0.01, 1.05], [0.04, 0.45, 0.02, 1.06]])
        dim = config.eqn_config.dim
        state = tf.random.uniform([2, 10, dim])
        corr = sde.corr_matrix(state, u_hat)
        det = tf.linalg.det(corr)
        self.assertAllGreaterEqual(det, 0.0)

    def test_brownian_motion(self):
        config = load_config("GBMJ", "Swap", 1)
        config.eqn_config.dim = 3
        sde = GBMwithSimpleJump(config)
        u_hat = tf.constant([[0.05, 0.45, 0.01, 1.05], [0.04, 0.45, 0.02, 1.06]])
        dim = config.eqn_config.dim
        state = tf.random.uniform([2, 1000, dim])
        dw = sde.brownian_motion(state, u_hat)
        mean = tf.reduce_mean(dw)
        self.assertAllLessEqual(mean, 1e-3)
        std = tf.math.sqrt(tf.math.reduce_variance(dw))
        self.assertAllLessEqual(std - 0.1, 1e-2)

    def test_jump_happen(self):
        """
        test the expection match or not
        """
        config = load_config("GBMJ", "Swap", 1)
        sde = GBMwithSimpleJump(config)
        u_hat = tf.constant(
            [
                [0.05, 0.1, 2.0],
                [0.05, 0.1, 2.5],
                [0.05, 0.1, 3.0],
                [0.05, 0.1, 3.5],
                [0.05, 0.1, 4.0],
            ]
        )
        dt = config.eqn_config.dt
        dim = config.eqn_config.dim
        state = tf.ones(
            [config.eqn_config.batch_size, config.eqn_config.sample_size, dim]
        )
        intensity = u_hat[:, 2]
        happens = sde.jump_happen(state, intensity, 100000)
        mean_exact = [[2.0 * dt], [2.5 * dt], [3.0 * dt], [3.5 * dt], [4.0 * dt]]
        mean_empirical = tf.reduce_mean(happens, axis=1)
        self.assertAllLessEqual(
            tf.reduce_mean(tf.abs((mean_exact - mean_empirical) / mean_exact)), 0.1
        )

    def test_compensate_expectation(self):
        """
        test the compensation drift
        """
        config = load_config("GBMJ", "Swap", 1)
        sde = GBMwithSimpleJump(config)
        u_hat = tf.constant([[0.05, 0.1, 2.0]])
        state = tf.ones([1, 10000, 1])
        com_jump = sde.compensate_jump(state, u_hat)
        self.assertAllLessEqual(tf.reduce_mean(tf.abs(com_jump - 0.0)), 1e-2)

    def test_martingale_property(self):
        config = load_config("GBMJ", "Swap", 1)
        sde = GBMwithSimpleJump(config)
        u_hat = tf.constant([[0.05, 0.1, 2.0], [0.05, 0.2, 3.0]])
        dim = config.eqn_config.dim
        s, dw, h, z = sde.sde_simulation(u_hat, 10000)
        s_exp = tf.reduce_mean(s[:, :, -1, :dim], axis=1) * tf.exp(-0.05)
        s_init = tf.reduce_mean(s[:, :, 0, :dim], axis=1)
        self.assertAllLessEqual(tf.reduce_mean(tf.abs((s_init - s_exp) / s_init)), 1e-2)


if __name__ == "__main__":
    tf.test.main()
