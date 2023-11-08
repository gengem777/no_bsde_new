import tensorflow as tf
from function_space import BlackScholesFormula, ZeroCouponBondFormula, EquitySwapFormula

# from markov_solver import MarkovianSolver, BaseBSDESolver, BSDEMarkovianModel
from pricers import MarkovianPricer, PureJumpPricer
from sde import GeometricBrownianMotion, HullWhiteModel
from sde_jump import GBMwithSimpleJump
from options import EuropeanOption, ZeroCouponBond, EuropeanSwap
from utils import load_config
from generate_data import create_dataset
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
    
    def test_loss_jump_swap(self):
        """
        In this problem, we test the loss function by plugging the analytical solution
        into the loss function and check the value is close d to zero or not. In this testing,
        we choose sde as geometricBrownianmotion and option as European option and choose the no_net
        as the BlackScholes formula.
        """
        config = load_config("GBMJ", "Swap", 1)
        sde = GBMwithSimpleJump(config)
        opt = EuropeanSwap(config)
        solver = PureJumpPricer(sde, opt, config)
        solver.no_net = EquitySwapFormula(1.0)
        samples = config.eqn_config.sample_size
        time_steps = config.eqn_config.time_steps
        dims = config.eqn_config.dim
        # dataset_path = "./dataset/GBMJ_Swap_1_100"
        create_dataset("GBMJ", "Swap", 1, 10)
        dataset_path = "./dataset/GBMJ_Swap_1_100"
        dataset = tf.data.experimental.load(
            dataset_path,
            element_spec=(
                tf.TensorSpec(shape=(samples, time_steps + 1, 1)),
                tf.TensorSpec(shape=(samples, time_steps + 1, dims)), # x
                tf.TensorSpec(shape=(samples, time_steps, dims)), # dw
                tf.TensorSpec(shape=(samples, time_steps, dims)), # h
                tf.TensorSpec(shape=(samples, time_steps, dims)), # z
                tf.TensorSpec(shape=(samples, time_steps + 1, 4)), # degree = 4
            ),
        )
        dataset = dataset.batch(5)
        for e in dataset.take(1):
            l, l1, l2 = solver.loss_fn(e)
        print(l1/time_steps, l2, l)
        self.assertAllLessEqual(l1 / time_steps, 1e-3)
        self.assertAllLessEqual(l2, 1e-3)
    


if __name__ == "__main__":
    tf.test.main()
