import tensorflow as tf
from function_space import BlackScholesFormula, ZeroCouponBondFormula

# from markov_solver import MarkovianSolver, BaseBSDESolver, BSDEMarkovianModel
from pricers import MarkovianPricer
from sde import GeometricBrownianMotion, HullWhiteModel
from options import EuropeanOption, ZeroCouponBond
import json
import munch

sde_list = ["GBM", "TGBM", "SV", "CEV", "SVJ", "HW"]
option_list = [
    "European",
    "EuropeanPut",
    "Lookback",
    "Asian",
    "Basket",
    "BasketnoPI",
    "Swap",
    "TimeEuropean",
    "BermudanPut",
    "Bond"
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


class TestPricer(tf.test.TestCase):
    def test_loss_function_equity(self):
        """
        In this problem, we test the loss function by plugging the analytical solution
        into the loss function and check the value is close d to zero or not. In this testing,
        we choose sde as geometricBrownianmotion and option as European option and choose the no_net
        as the BlackScholes formula.
        """
        config = load_config("GBM", "European", 1)
        sde = GeometricBrownianMotion(config)
        opt = EuropeanOption(config)
        solver = MarkovianPricer(sde, opt, config)
        solver.no_net = BlackScholesFormula(1.0)
        samples = config.eqn_config.sample_size
        time_steps = config.eqn_config.time_steps
        dims = config.eqn_config.dim
        dataset_path = "./dataset/GBM_European_1_100"
        dataset = tf.data.experimental.load(
            dataset_path,
            element_spec=(
                tf.TensorSpec(shape=(samples, time_steps + 1, 1)),
                tf.TensorSpec(shape=(samples, time_steps + 1, dims)),
                tf.TensorSpec(shape=(samples, time_steps, dims)),
                tf.TensorSpec(shape=(samples, time_steps + 1, 3)),
            ),
        )
        dataset = dataset.batch(100)
        for e in dataset.take(1):
            _, l1, l2 = solver.loss_fn(e)
        # print(l1/time_steps, l2)
        self.assertAllLessEqual(l1 / time_steps, 1e-3)
        self.assertAllLessEqual(l2, 1e-3)
    
    def test_loss_function_bond(self):
        """
        In this problem, we test the loss function by plugging the analytical solution
        into the loss function and check the value is close d to zero or not. In this testing,
        we choose sde as geometricBrownianmotion and option as European option and choose the no_net
        as the zero coupon bond formula.
        """
        config = load_config("HW", "Bond", 1)
        sde = HullWhiteModel(config)
        opt = ZeroCouponBond(config)
        solver = MarkovianPricer(sde, opt, config)
        solver.no_net = ZeroCouponBondFormula(1.0)
        samples = config.eqn_config.sample_size
        time_steps = config.eqn_config.time_steps
        dims = config.eqn_config.dim
        dataset_path = "./dataset/HW_Bond_1_10"
        dataset = tf.data.experimental.load(
            dataset_path,
            element_spec=(
                tf.TensorSpec(shape=(samples, time_steps + 1, 1)),
                tf.TensorSpec(shape=(samples, time_steps + 1, dims)),
                tf.TensorSpec(shape=(samples, time_steps, dims)),
                tf.TensorSpec(shape=(samples, time_steps + 1, 4)),
            ),
        )
        dataset = dataset.batch(100)
        for e in dataset.take(1):
            _, l1, l2 = solver.loss_fn(e)
        # print(l1/time_steps, l2)
        self.assertAllLessEqual(l1 / time_steps, 1e-3)
        self.assertAllLessEqual(l2, 1e-3)


if __name__ == "__main__":
    tf.test.main()
