import tensorflow as tf
import numpy as np
from sde import CEVModel, HestonModel, GeometricBrownianMotion, HullWhiteModel
# from config import Config

import json
import munch
import sde as eqn
import options as opts
import solvers as sls

sde_list = ["GBM", "TGBM", "SV", "HW", "SVJ"]
option_list = ["European", "EuropeanPut", "Lookback", "Asian", "Basket", "BasketnoPI", "Swap", "TimeEuropean", "BermudanPut"]
dim_list = [1, 3, 5, 10, 20]

def load_config(sde_name: str, option_name: str, dim: int=1):
    """
    This function is for introducing config json files into test functions
    """
    if (sde_name not in sde_list) or (option_name not in option_list) or (dim not in dim_list):
        raise ValueError(f"please input right sde_name in {sde_list},\
                          option_name in {option_list} and dim in {dim_list}")
    else:
        json_path = f'./config/{sde_name}_{option_name}_{dim}.json'
    with open(json_path) as json_data_file:
        config = json.load(json_data_file)
    
    config = munch.munchify(config)
    return config


class TestGeometricBrowianMotion(tf.test.TestCase):
    def test_initial_sampler(self):
        config = load_config("GBM", "European", 1)
        sde = GeometricBrownianMotion(config)
        u_hat = tf.constant([[0.05, 0.45, 1.05],[0.04, 0.58, 1.06]])
        dim = config.eqn_config.dim
        x_init = config.eqn_config.x_init
        initial_state = sde.initial_sampler(u_hat, 10000)
        # print(tf.reduce_mean(initial_state))
        self.assertEqual(initial_state.shape, [2, 10000, dim])
        self.assertAllLessEqual(tf.reduce_mean(initial_state - x_init), 5e-2)

    def test_drift(self):
        config = load_config("GBM", "European", 1)
        sde = GeometricBrownianMotion(config)
        dim = config.eqn_config.dim
        u_hat = tf.constant([[0.05, 0.45, 1.05],[0.05, 0.58, 1.06]])
        state = tf.random.uniform([2, 10, dim])
        drift = sde.drift(0, state, u_hat)
        self.assertAllLessEqual(drift - 0.05 * state, 1e-7)

    def test_diffusion(self):
        config = load_config("GBM", "European", 1)
        sde = GeometricBrownianMotion(config)
        dim = config.eqn_config.dim
        u_hat = tf.constant([[0.05, 0.45, 1.05],[0.04, 0.45, 1.06]])
        state = tf.random.uniform([2, 10, dim])
        diffusion = sde.diffusion(0, state, u_hat)
        self.assertAllLessEqual(diffusion - 0.45 * state, 1e-7)

    def test_corr_matrix(self):
        config = load_config("GBM", "European", 1)
        config.eqn_config.dim = 3
        sde = GeometricBrownianMotion(config)
        u_hat = tf.constant([[0.05, 0.45, 0.01, 1.05],[0.04, 0.45, 0.02, 1.06]])
        dim = config.eqn_config.dim
        state = tf.random.uniform([2, 10, dim])
        corr = sde.corr_matrix(state, u_hat)
        det = tf.linalg.det(corr)
        self.assertAllGreaterEqual(det, 0.0)

    def test_brownian_motion(self):
        config = load_config("GBM", "European", 1)
        config.eqn_config.dim = 3
        sde = GeometricBrownianMotion(config)
        u_hat = tf.constant([[0.05, 0.45, 0.01, 1.05],[0.04, 0.45, 0.02, 1.06]])
        dim = config.eqn_config.dim
        state = tf.random.uniform([2, 1000, dim])
        dw = sde.brownian_motion(state, u_hat)
        mean = tf.reduce_mean(dw)
        self.assertAllLessEqual(mean, 1e-3)
        std = tf.math.sqrt(tf.math.reduce_variance(dw))
        self.assertAllLessEqual(std - 0.1, 1e-2)
        
    def test_martingale_property(self):
        config = load_config("GBM", "European", 1)
        sde = GeometricBrownianMotion(config)
        u_hat = tf.constant([[0.05, 0.1, 0.2],[0.05, 0.2, 0.5]])
        dim = config.eqn_config.dim
        s,_ = sde.sde_simulation(u_hat, 10000)
        s_exp = tf.reduce_mean(s[:, :, -1, :dim], axis=1) * tf.exp(-0.05)
        s_init = tf.reduce_mean(s[:, :, 0, :dim], axis=1)
        self.assertAllLessEqual(tf.reduce_mean(tf.abs(s_init - s_exp)), 1e-2)

class TestHestonModel(tf.test.TestCase):
    def test_initial_sampler(self):
        config = load_config("SV", "Swap", 1)
        sde = HestonModel(config)
        u_hat = tf.constant([[0.05, 0.45, 1.05, 1.04],[0.04, 0.58, 1.06, 1.06]])
        dim = config.eqn_config.dim
        samples = config.eqn_config.sample_size
        initial_state = sde.initial_sampler(u_hat, samples)
        self.assertEqual(initial_state.shape, [2, samples, 2*dim])

    def test_drift(self):
        config = load_config("SV", "Swap", 1)
        sde = HestonModel(config)
        u_hat = tf.constant([[0.05, 0.45, 1.05, 1.04],[0.04, 0.58, 1.06, 1.06]])
        state = tf.constant([[1.2, 0.12],[1.13, 0.11]])
        a_1 = 0.05 * 1.2
        b_1 = 0.04 * 1.13
        a_2 = 0.45 * (1.05 - 0.12)
        b_2 = 0.58 * (1.06 - 0.11)
        drift = tf.constant([[a_1, a_2], [b_1, b_2]])
        dim = config.eqn_config.dim
        M = config.eqn_config.sample_size

        def expand_dim(tensor2d):
            B = tf.shape(tensor2d)[0]
            tensor = tf.reshape(tensor2d, [B, 1, dim*2])
            tensor3d = tf.tile(tensor, [1, M, 1])
            return tensor3d
        
        state = expand_dim(state)
        drift = expand_dim(drift)
        drift_to_test = sde.drift(0, state, u_hat)
        self.assertAllLessEqual(tf.reduce_mean(tf.abs(drift - drift_to_test)), 1e-6)

    def test_diffusion(self):
        config = load_config("SV", "Swap", 1)
        sde = HestonModel(config)
        u_hat = tf.constant([[0.05, 0.45, 1.05, 1.04],[0.04, 0.58, 1.06, 1.06]])
        state = tf.constant([[1.2, 0.12],[1.13, 0.11]])
        v_1 = np.sqrt(0.12) * 1.04
        v_2 = np.sqrt(0.11) * 1.06
        s_1 = np.sqrt(0.12) * 1.2
        s_2 = np.sqrt(0.11) * 1.13
        diff = tf.constant([[s_1, v_1], [s_2, v_2]])

        dim = config.eqn_config.dim
        M = config.eqn_config.sample_size

        def expand_dim(tensor2d):
            B = tf.shape(tensor2d)[0]
            tensor = tf.reshape(tensor2d, [B, 1, dim*2])
            tensor3d = tf.tile(tensor, [1, M, 1])
            return tensor3d
        
        state = expand_dim(state)
        diff = tf.cast(expand_dim(diff), tf.float32)
        diff_to_test = sde.diffusion(0, state, u_hat)
        self.assertAllLessEqual(tf.reduce_mean(tf.abs(diff - diff_to_test)), 1e-6)

    def test_drift_onestep(self):
        config = load_config("SV", "Swap", 1)
        sde = HestonModel(config)
        u_hat = tf.constant([[0.05, 0.45, 1.05, 1.04], [0.05, 0.45, 1.05, 1.04]])
        state = tf.constant([[1.2, 0.12],[1.13, 0.11]])
        dim = config.eqn_config.dim
        M = config.eqn_config.sample_size
        N = config.eqn_config.time_steps
        def expand_dim(tensor2d):
            B = tf.shape(tensor2d)[0]
            tensor = tf.reshape(tensor2d, [B, 1, 1, dim*2])
            tensor4d = tf.tile(tensor, [1, M, N, 1])
            return tensor4d 
        state_tensor = expand_dim(state)
        u_hat = tf.reshape(u_hat, [u_hat.shape[0], 1, 1, u_hat.shape[-1]])
        u_hat = tf.tile(u_hat, [1, M, N, 1])
        drift_state = sde.drift_onestep(0, state_tensor, u_hat)
        s = state_tensor[..., :dim]
        v = state_tensor[..., dim:]
        s_plus = s * 0.05
        v_plus = 0.45 * (1.05 - v)
        drift_target = tf.concat([s_plus, v_plus], axis=-1)
        self.assertAllLessEqual(tf.reduce_mean(tf.abs(drift_state - drift_target)), 1e-6)

    def test_diffusion_onestep(self):
        config = load_config("SV", "Swap", 1)
        sde = HestonModel(config)
        u_hat = tf.constant([[0.05, 0.45, 1.05, 1.04], [0.05, 0.45, 1.05, 1.04]])
        state = tf.constant([[1.2, 0.12],[1.13, 0.11]])
        dim = config.eqn_config.dim
        M = config.eqn_config.sample_size
        N = config.eqn_config.time_steps
        def expand_dim(tensor2d):
            B = tf.shape(tensor2d)[0]
            tensor = tf.reshape(tensor2d, [B, 1, 1, dim*2])
            tensor4d = tf.tile(tensor, [1, M, N, 1])
            return tensor4d
        state_tensor = expand_dim(state)
        u_hat = tf.reshape(u_hat, [u_hat.shape[0], 1, 1, u_hat.shape[-1]])
        u_hat = tf.tile(u_hat, [1, M, N, 1])
        diff_state = sde.diffusion_onestep(0, state_tensor, u_hat)
        s = state_tensor[..., :dim]
        v = state_tensor[..., dim:]
        s_plus = tf.sqrt(v) * s
        v_plus = tf.sqrt(v) * 1.04
        diff_target = tf.concat([s_plus, v_plus], axis=-1)
        self.assertAllLessEqual(tf.reduce_mean(tf.abs(diff_state - diff_target)), 1e-6)

    def test_martingale_property(self):
        """
        this is integrate test to test whether under Q measure the discounted process is a martingale
        """
        config = load_config("SV", "Swap", 1)
        sde = HestonModel(config)
        u_hat = tf.constant([[0.05, 0.1, 0.3, 0.02, -0.1, 1.04],[0.05, 0.11, 0.3, 0.02, -0.1, 1.06]])
        dim = config.eqn_config.dim
        s,_ = sde.sde_simulation(u_hat, 10000)
        s_exp = tf.reduce_mean(s[:, :, -1, :dim], axis=1) * tf.exp(-0.05)
        s_init = tf.reduce_mean(s[:, :, 0, :dim], axis=1)
        self.assertAllLessEqual(tf.reduce_mean(tf.abs(s_init - s_exp)), 1e-2)

class TestHullWhiteModel(tf.test.TestCase):
    def test_positivity(self):
        config = load_config("HW", "Swap", 1)
        sde = HullWhiteModel(config)
        u_hat = tf.constant([[0.045, 0.4, 0.03, 0.011],
                             [0.005, 0.2, 0.03, 0.011],
                             [0.005, 0.6, 0.03, 0.011],
                             [0.08, 0.2, 0.03, 0.011],
                             [0.08, 0.6, 0.03, 0.011],
                             [0.045, 0.4, 0.01, 0.011],
                             [0.005, 0.2, 0.01, 0.011],
                             [0.005, 0.6, 0.01, 0.011],
                             [0.08, 0.2, 0.01, 0.011],
                             [0.08, 0.6, 0.01, 0.011]])
        r,_ = sde.sde_simulation(u_hat, 1000)
        # self.assertAllGreaterEqual(r, 0.0)
    
    def test_drift(self):
        config = load_config("HW", "Swap", 1)
        sde = HullWhiteModel(config)
        u_hat = tf.constant([[0.045, 0.4, 0.03, 0.011]])
        state = tf.ones((1, 100, 1)) * 0.02
        drift_exact = 0.045 * (0.4 - 0.02)
        drift = sde.drift(0, state, u_hat)
        self.assertAllLessEqual(drift - drift_exact, 1e-9)

    def test_euler_one_step(self):
        config = load_config("HW", "Swap", 1)
        sde = HullWhiteModel(config)
        dt = config.eqn_config.dt
        u_hat = tf.constant([[0.045, 0.4, 0.03, 0.011]])
        state = tf.ones((1, 10000, 1)) * 0.02
        mean_exact = 0.045 * (0.4 - 0.02) * dt
        std_exact = 0.03 * tf.math.sqrt(dt)
        r, _ = sde.euler_maryama_step(0, state, u_hat)
        gap = r - state
        mean = tf.reduce_mean(gap)
        std = tf.math.sqrt(tf.math.reduce_variance(gap)) 
        self.assertAllLessEqual(tf.abs(mean - mean_exact), 1e-3)
        self.assertAllLessEqual(tf.abs(std - std_exact), 1e-3)
    
    def test_zero_coupon_bond(self):
        config = load_config("HW", "Swap", 1)
        sde = HullWhiteModel(config)
        T = config.eqn_config.T
        u_hat = tf.constant([[0.045, 0.4, 0.03, 0.011]])
        state = tf.ones((1, 1, 1)) * 0.01
        kappa = 0.045
        theta = 0.4
        sigma = 0.03
        B = (1 - tf.exp(-kappa * (T - 0)))/ kappa
        A = tf.exp((B - T + 0)*(kappa**2 * theta 
                                - sigma**2/2)/kappa**2 
                                + (sigma*B)**2/(4*kappa))
        p_exact = A * tf.exp(-B * tf.reduce_sum(state, axis=-1, keepdims=True))
        p = sde.zcp_value(0, state, u_hat, T)
        self.assertAllLessEqual(tf.abs(p - p_exact), 1e-9)
        p_2 = sde.zcp_value(T, state, u_hat, T)
        self.assertAllLessEqual(tf.abs(p_2 - 1.0), 1e-9)
    
    def test_sde_bond(self):
        config = load_config("HW", "Swap", 1)
        sde = HullWhiteModel(config)
        T = config.eqn_config.T
        u_hat = tf.constant([[0.4, 0.05, 0.03, 0.011]])
        state = tf.ones((1, 1, 1)) * 0.05
        p_exact = sde.zcp_value(0, state, u_hat, T)
        dt = config.eqn_config.dt
        r ,_ = sde.sde_simulation(u_hat, 1000)
        cum_rate = tf.reduce_sum((r[:, :, :-1,:] + r[:, :, 1:,:])/2, axis=2) * dt
        p_simulate = tf.reduce_mean(tf.exp(-cum_rate), axis=1)
        # self.assertAllLessEqual(tf.abs(p_simulate - p_exact), 1e-2)
        print(p_simulate, p_exact)

if __name__ == "__main__":
    tf.test.main()
