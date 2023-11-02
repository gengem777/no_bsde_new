import tensorflow as tf
import numpy as np
import json
import munch
import sde as eqn
import options as opts
from pricers import EarlyExercisePricer, MarkovianPricer
from generate_data import create_dataset
import global_constants as constants
from utils import load_config

# load config
print("load dataset and config and classes for phase one training")
sde_name = "HW"
option_name = "SwaptionLast"
dim = 1
epsilon = 1.0 
config = load_config(sde_name, option_name, dim)
config = munch.munchify(config)
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
checkpoint_path = f"./checkpoint2/HW_bermudan/{sde_name}_{option_name}_{dim}_2"
# initialize the solver and train
pricer = MarkovianPricer(sde, option, config)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=config.net_config.lr, decay_steps=2000, decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-6)
pricer.compile(optimizer=optimizer)
# tf.config.run_functions_eagerly(True)
print("phase1 training")
pricer.fit(x=dataset, epochs=20)
pricer.no_net.save_weights(checkpoint_path)


sde_name = "HW"
option_name = "SwaptionFirst"
dim = 1
config = load_config(sde_name, option_name, dim)
config = munch.munchify(config)
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
# In this section, we directly use the continuation value from the last time period which is also the last time interval for the time
def slice_fn(t, x, dw, u):
    t_slice = t[:, :, :11, :]
    x_slice = x[:, :, :11, :]
    dw_slice = dw[:, :, :11, :]
    u_slice = u[:, :, :11, :]
    return t_slice, x_slice, dw_slice, u_slice
sub_dataset = dataset.map(slice_fn)
test_dataset = sub_dataset.take(10)
train_dataset = sub_dataset.skip(10)
checkpoint_path = f"./checkpoint2/HW_bermudan/{sde_name}_{option_name}_{dim}_2"
# initialize the solver and train
pricer = MarkovianPricer(sde, option, config)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=config.net_config.lr, decay_steps=2000, decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-6)
pricer.compile(optimizer=optimizer)
# tf.config.run_functions_eagerly(True)
print("phase2 training")
pricer.fit(x=dataset, epochs=20)
pricer.no_net.save_weights(checkpoint_path)

# Do Integrate test
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
print("test passed")