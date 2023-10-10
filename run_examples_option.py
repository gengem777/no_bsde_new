import tensorflow as tf
import json
import munch
import sde as eqn
import options as opts
from trainer import EarlyExerciseTrainer
from generate_data import create_dataset

# load config
sde_list = ["GBM", "TGBM", "SV", "CEV", "SVJ"]
option_list = ["European", "EuropeanPut", "Lookback", "Asian", "Basket", "BasketnoPI", "Swap", "TimeEuropean", "BermudanPut"]
dim_list = [1, 3, 5, 10, 20]

sde_name = "GBM"
option_name = "European"
dim = 1

if (sde_name not in sde_list) or (option_name not in option_list) or (dim not in dim_list):
        raise ValueError(f"please input right sde_name in {sde_list},\
                          option_name in {option_list} and dim in {dim_list}")
else:
    json_path = f'./configs/{sde_name}_{option_name}_{dim}.json'
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
dataset_path = f'./dataset/{sde_name}_{option_name}_{dim}_{time_steps}'
create_dataset(sde_name, option_name, dim, 50)

#load dataset
dataset = tf.data.experimental.load(dataset_path, element_spec=(
    tf.TensorSpec(shape=(samples, time_steps + 1, 1)),
    tf.TensorSpec(shape=(samples, time_steps + 1, dims)),
    tf.TensorSpec(shape=(samples, time_steps, dims)),
    tf.TensorSpec(shape=(samples, time_steps + 1, 3))))
dataset = dataset.batch(config.eqn_config.batch_size)
checkpoint_path = f'./checkpoint2/GBM_bermudan/{sde_name}_{option_name}_{dim}'
#initialize the solver and train
solver = EarlyExerciseTrainer(sde, option, config)
# tf.config.run_functions_eagerly(True)
solver.train(dataset, 10, checkpoint_path)




