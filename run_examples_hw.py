import tensorflow as tf
from pricers import MarkovianPricer
from generate_data import create_dataset
import json
import munch
import sde as eqn
import options as opts


# load config
sde_list = ["GBM", "TGBM", "SV", "HW", "SVJ"]
option_list = [
    "European",
    "EuropeanPut",
    "Lookback",
    "Asian",
    "Basket",
    "BasketnoPI",
    "Swap",
    "Bond",
    "TimeEuropean",
    "BermudanPut",
]
dim_list = [1, 3, 5, 10, 20]

sde_name = "HW"
option_name = "Bond"
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
        tf.TensorSpec(shape=(samples, time_steps + 1, 4)),
    ),
)
dataset = dataset.batch(config.eqn_config.batch_size)
checkpoint_path = f"./checkpoint2/HW_Bond/{sde_name}_{option_name}_{dim}_{time_steps}"
# initialize the solver and train
pricer = MarkovianPricer(sde, option, config)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=config.net_config.lr, decay_steps=2000, decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-6)
pricer.compile(optimizer=optimizer)
# tf.config.run_functions_eagerly(True)
pricer.fit(x=dataset, epochs=10)