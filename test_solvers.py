import tensorflow as tf
import numpy as np
from data_generators import DiffusionModelGenerator
from function_space import BlackScholesFormula
# from markov_solver import MarkovianSolver, BaseBSDESolver, BSDEMarkovianModel
from solvers import MarkovianSolver
from sde import GeometricBrownianMotion, CEVModel, HestonModel
from options import EuropeanOption, GeometricAsian, LookbackOption
from config import Config

import json
import munch
import sde as eqn
import options as opts
import solvers as sls


sde_list = ["GBM", "TGBM", "SV", "CEV", "SVJ"]
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

class TestSolver(tf.test.TestCase):
    def test_loss_function(self):
        """
        In this problem, we test the loss function by plugging the analytical solution
        into the loss function and check the value is close d to zero or not. In this testing,
        we choose sde as geometricBrownianmotion and option as European option and choose the no_net
        as the BlackScholes formula.
        """
        config = load_config("GBM", "European", 1)
        sde = GeometricBrownianMotion(config)
        opt = EuropeanOption(config)
        solver = MarkovianSolver(sde, opt, config)
        solver.no_net = BlackScholesFormula(1.0)
        samples = config.eqn_config.sample_size
        time_steps = config.eqn_config.time_steps
        dims = config.eqn_config.dim
        dataset_path = './dataset/GBM_European_1'
        dataset = tf.data.experimental.load(dataset_path, element_spec=(
            tf.TensorSpec(shape=(samples, time_steps, 1)),
            tf.TensorSpec(shape=(samples, time_steps, dims)),
            tf.TensorSpec(shape=(samples, time_steps-1, dims)),
            tf.TensorSpec(shape=(samples, time_steps, 3))))
        dataset = dataset.batch(100)
        for e in dataset.take(1):
            _, l1 , l2 = solver.loss(e)
        # print(l1/time_steps, l2)
        self.assertAllLessEqual(l1/time_steps, 1e-3)
        self.assertAllLessEqual(l2, 1e-3)

if __name__ == "__main__":
    tf.test.main()