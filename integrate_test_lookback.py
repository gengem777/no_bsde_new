import tensorflow as tf
import numpy as np
import json
import munch
import sde as eqn
import options as opts
from pricers import MarkovianPricer
from generate_data import create_dataset

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
    "TimeEuropean",
    "BermudanPut",
]
dim_list = [1, 3, 10]

sde_name = "GBM"
option_name = "Lookback"
dim = 1
epsilon = 1.0 # corresponded to the paper in Berner 2020.
tf.random.set_seed(0)
np.random.seed(0)
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
        tf.TensorSpec(shape=(samples, time_steps + 1, 2 * dims)),
        tf.TensorSpec(shape=(samples, time_steps, dims)),
        tf.TensorSpec(shape=(samples, time_steps + 1, 3)), # degree = 3
    ),
)
dataset = dataset.batch(config.eqn_config.batch_size)
test_dataset = dataset.take(10) 
train_dataset = dataset.skip(10)
checkpoint_path = f"./checkpoint/{sde_name}_{option_name}_{dim}"
# initialize the solver and train
pricer = MarkovianPricer(sde, option, config)
learning_rate =config.net_config.lr
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=learning_rate,
    decay_steps=2000,
    decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-6)
pricer.compile(optimizer=optimizer)
# tf.config.run_functions_eagerly(True)
pricer.fit(x=dataset, epochs=10)
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
assert mean <= 0.1 
assert std <= 0.1
print("test passed")

