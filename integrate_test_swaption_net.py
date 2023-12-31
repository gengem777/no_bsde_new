import tensorflow as tf
import numpy as np
import json
import munch
import sde as eqn
import options as opts
from pricers import EarlyExercisePricer, MarkovianPricer, FixIncomeEuropeanPricer
from generate_data import create_dataset
import global_constants as constants
from utils import load_config

# load config
sde_list = ["GBM", "TGBM", "SV", "SVJ", "HW"]
option_list = [
    "European",
    "EuropeanPut",
    "Lookback",
    "Asian",
    "Basket",
    "BasketnoPI",
    "Swap",
    "Bond",
    "Swaption",
    "SwaptionFirst",
    "SwaptionLast",
    "TimeEuropean",
    "BermudanPut",
]
dim_list = [1, 3, 5, 10, 20]
tf.random.set_seed(0)
np.random.seed(0)
# import data and classes for phase 1 training
sde_name = "HW"
option_name = "SwaptionLast"
dim = 1
epsilon = 1.0 
config = load_config(sde_name, option_name, dim)
initial_mode = config.eqn_config.initial_mode
kernel_type = config.net_config.kernel_type
sde = getattr(eqn, config.eqn_config.sde_name)(config)
option = getattr(opts, config.eqn_config.option_name)(config)

samples = config.eqn_config.sample_size
time_steps = config.eqn_config.time_steps
dims = config.eqn_config.dim
dataset_path = f"./dataset/{sde_name}_{option_name}_{dim}_{time_steps}"
create_dataset(sde_name, option_name, dim, 50)

# load dataset
dataset = tf.data.experimental.load(
    dataset_path,
    element_spec=(
        tf.TensorSpec(shape=(samples, time_steps + 1, 1)),
        tf.TensorSpec(shape=(samples, time_steps + 1, dims)),
        tf.TensorSpec(shape=(samples, time_steps, dims)),
        tf.TensorSpec(shape=(samples, time_steps + 1, 4)), # degree = 4
    ),
)
dataset = dataset.batch(config.eqn_config.batch_size)

# phase1 training and testing
def slice_fn(t, x, dw, u):
    t_slice = t[:, :, 10:, :]
    x_slice = x[:, :, 10:, :]
    dw_slice = dw[:, :, 10:, :]
    u_slice = u[:, :, 10:, :]
    return t_slice, x_slice, dw_slice, u_slice
sub_dataset = dataset.map(slice_fn)
test_dataset = sub_dataset.take(10)
train_dataset = sub_dataset.skip(10)
checkpoint_path_last = f"./checkpoint2/HW_bermudan/{sde_name}_{option_name}_{dim}_last"
# initialize the solver and train
pricer = MarkovianPricer(sde, option, config)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=config.net_config.lr, decay_steps=2000, decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-6)
pricer.compile(optimizer=optimizer)
# tf.config.run_functions_eagerly(True)
pricer.fit(x=dataset, epochs=20)
pricer.no_net.save_weights(checkpoint_path_last)

# import data and classes for phase 2 training
# In this section, we use the EuropeanPricer train the Bermudan swaption with continuation value from the last network.
sde_name = "HW"
option_name = "SwaptionFirst"
dim = 1
config = load_config(sde_name, option_name, dim)
initial_mode = config.eqn_config.initial_mode
kernel_type = config.net_config.kernel_type
sde = getattr(eqn, config.eqn_config.sde_name)(config)
option = getattr(opts, config.eqn_config.option_name)(config)

samples = config.eqn_config.sample_size
time_steps = config.eqn_config.time_steps
dims = config.eqn_config.dim
dataset_path = f"./dataset/{sde_name}_{option_name}_{dim}_{time_steps}"
create_dataset(sde_name, option_name, dim, 50)

# load dataset
dataset = tf.data.experimental.load(
    dataset_path,
    element_spec=(
        tf.TensorSpec(shape=(samples, time_steps + 1, 1)),
        tf.TensorSpec(shape=(samples, time_steps + 1, dims)),
        tf.TensorSpec(shape=(samples, time_steps, dims)),
        tf.TensorSpec(shape=(samples, time_steps + 1, 4)),
    ),
)
dataset = dataset.batch(config.eqn_config.batch_size)
# phase2 training and testing
def slice_fn(t, x, dw, u):
    t_slice = t[:, :, :11, :]
    x_slice = x[:, :, :11, :]
    dw_slice = dw[:, :, :11, :]
    u_slice = u[:, :, :11, :]
    return t_slice, x_slice, dw_slice, u_slice
sub_dataset = dataset.map(slice_fn)
test_dataset = sub_dataset.take(10)
train_dataset = sub_dataset.skip(10)
checkpoint_path_first = f"./checkpoint2/HW_bermudan/{sde_name}_{option_name}_{dim}_1"
# initialize the solver and train
pricer = FixIncomeEuropeanPricer(sde, option, config)
pricer.step_to_next_round() # make sure we can use the max(cont, early_payoff) as the terminal condition
pricer.no_net_target.load_weights(checkpoint_path_last)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=config.net_config.lr, decay_steps=2000, decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-6)
pricer.compile(optimizer=optimizer)
tf.config.run_functions_eagerly(True)
pricer.fit(x=dataset, epochs=20)
pricer.no_net.save_weights(checkpoint_path_first)

# Do integrate test
t = tf.reshape(
    tf.constant([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), [10, 1, 1, 1]
)
x = tf.reshape(
    tf.constant([0.052, 0.053, 0.055, 0.057, 0.058, 0.052, 0.053, 0.055, 0.057, 0.058]),
    [10, 1, 1, 1],
)
u = tf.reshape(
    tf.constant(
        [
            [0.4, 0.052, 0.03, 0.0],
            [0.4, 0.053, 0.03, 0.0],
            [0.4, 0.055, 0.03, 0.0],
            [0.4, 0.057, 0.03, 0.0],
            [0.4, 0.058, 0.03, 0.0],
            [0.4, 0.052, 0.05, 0.0],
            [0.4, 0.053, 0.05, 0.0],
            [0.4, 0.055, 0.05, 0.0],
            [0.4, 0.057, 0.05, 0.0],
            [0.4, 0.058, 0.05, 0.0],
        ]
    ),
    [10, 1, 1, 4],
)
y_pred = pricer((t, x, u))
dates = config.eqn_config.leg_dates
y_exact = option.exact_price(t, x, u) 
error = tf.reduce_mean(tf.abs((y_pred - y_exact)/(epsilon + y_exact)))
assert error <= 1.0
print(y_pred, y_exact)
print("test passed")