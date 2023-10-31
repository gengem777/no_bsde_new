import tensorflow as tf
import numpy as np
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

sde_name = "SV"
option_name = "Swap"
dim = 1
epsilon = 1.0 # corresponded to the paper in Berner 2020.

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
create_dataset(sde_name, option_name, dim, 50)
dataset = tf.data.experimental.load(
    dataset_path,
    element_spec=(
        tf.TensorSpec(shape=(samples, time_steps + 1, 1)),
        tf.TensorSpec(shape=(samples, time_steps + 1, 2 * dims)),
        tf.TensorSpec(shape=(samples, time_steps, 2 * dims)),
        tf.TensorSpec(shape=(samples, time_steps + 1, 6)),
    ),
)
dataset = dataset.batch(config.eqn_config.batch_size)
test_dataset = dataset.take(10) 
train_dataset = dataset.skip(10)
checkpoint_path = f"./checkpoint/{sde_name}_{option_name}_{dim}"
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
y1, y_exact1, t1, x1, u1 = y_pred.numpy(), y_exact.numpy(), t.numpy(), x.numpy(), u_hat.numpy()

def evaluate(y1, y_exact1):
    idx=-1
    t = (np.abs(y1[:,:,:idx] - y_exact1[:,:,:idx]))/(epsilon + y_exact1[:,:,:idx])
    return np.mean(t), np.std(t)

mean, std = evaluate(y1, y_exact1)
print(mean)
assert mean <= 1e-1 
assert std <= 1e-1 
print("test passed")
