import tensorflow as tf
import json
import munch
import sde as eqn
import options as opts
from pricers import EarlyExercisePricer
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
    "Bond",
    "Swaption",
    "TimeEuropean",
    "BermudanPut",
]
dim_list = [1, 3, 5, 10, 20]

sde_name = "HW"
option_name = "Swaption"
dim = 1

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
create_dataset(sde_name, option_name, dim, 2)

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
test_dataset = dataset.take(10)
train_dataset = dataset.skip(10)
checkpoint_path = f"./checkpoint2/HW_bermudan/{sde_name}_{option_name}_{dim}"
# initialize the solver and train
pricer = EarlyExercisePricer(sde, option, config)
tf.config.run_functions_eagerly(True)
pricer.fit(dataset, 1, checkpoint_path)
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
y_pred = pricer.no_nets[0]((t, x, u))
dates = config.eqn_config.leg_dates
y_exact = option.zcp(0, x, u, dates[0]) - option.zcp(0, x, u, dates[-1])
print(y_pred, y_exact)
error = tf.reduce_mean(tf.abs((y_pred - y_exact)/y_exact))
print(error)
