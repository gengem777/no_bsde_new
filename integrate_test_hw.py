import tensorflow as tf
import numpy as np
from pricers import MarkovianPricer
from generate_data import create_dataset
import global_constants as constants
import json
import munch
import sde as eqn
import options as opts
from utils import load_config


# load config
sde_name = "HW"
option_name = "Bond"
dim = 1
epsilon = 1.0 # corresponded to the paper in Berner 2020.
config = load_config(sde_name, option_name, dim)
initial_mode = config.eqn_config.initial_mode
kernel_type = config.net_config.kernel_type
sde = getattr(eqn, config.eqn_config.sde_name)(config)
option = getattr(opts, config.eqn_config.option_name)(config)

# load dataset
samples = config.eqn_config.sample_size
time_steps = config.eqn_config.time_steps
dims = config.eqn_config.dim
dataset_path = f"./dataset/{sde_name}_{option_name}_{dim}_{time_steps}"
create_dataset("HW", "Bond", 1, 50)
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
test_dataset = dataset.take(10) 
train_dataset = dataset.skip(10)
checkpoint_path = f"./checkpoint/{sde_name}_{option_name}_{dim}_{time_steps}"
# initialize the solver and train
pricer = MarkovianPricer(sde, option, config)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=config.net_config.lr, decay_steps=2000, decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-6)
pricer.compile(optimizer=optimizer)
# tf.config.run_functions_eagerly(True)
pricer.fit(x=dataset, epochs=20)
pricer.no_net.save_weights(checkpoint_path)
# split dataset
for element in test_dataset.take(5):
    t, x, _, u_hat = element
y_pred = pricer((t, x, u_hat))
y_exact = option.exact_price(t, x, u_hat)
y_pred, y_exact = y_pred.numpy(), y_exact.numpy()

def evaluate(y1, y2):
    idx=-1
    t = (np.abs(y1[:,:,:idx] - y2[:,:,:idx]))/(epsilon + y2[:,:,:idx])
    return np.mean(t), np.std(t)

mean, std = evaluate(y_pred, y_exact)
print(mean)
assert mean <= 1e-1 
assert std <= 1e-1 
print("test passed")