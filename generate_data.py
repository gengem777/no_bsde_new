import json
import munch
import time
import os

import numpy as np
import tensorflow as tf
import sde as eqn
import options as opts
import solvers as sls
from trainer import BSDETrainer, BermudanTrainer
from data_generators import DiffusionModelGenerator


sde_list = ["GBM", "TGBM", "SV", "HW", "SVJ"]
option_list = ["European", "EuropeanPut", "Lookback", "Asian", "Basket", "BasketnoPI", "Swap", "Bond", "TimeEuropean", "BermudanPut"]
dim_list = [1, 3, 5, 10, 20]

def create_dataset(sde_name: str, option_name: str, dim: int=10, num_of_batches: int=100):
    if (sde_name not in sde_list) or (option_name not in option_list) or (dim not in dim_list):
        raise ValueError(f"please input right sde_name in {sde_list},\
                          option_name in {option_list} and dim in {dim_list}")
    else:
        json_path = f'./config/{sde_name}_{option_name}_{dim}.json'
    with open(json_path) as json_data_file:
        config = json.load(json_data_file)
    
    config = munch.munchify(config)
    sde = getattr(eqn, config.eqn_config.sde_name)(config)
    option = getattr(opts, config.eqn_config.option_name)(config)
    
    data_generator = DiffusionModelGenerator(sde, config.eqn_config, option, num_of_batches)
    
    def gen_data_generator():
        for i in range(data_generator.__len__()):
            yield data_generator.__getitem__(i)[0]
    M = config.eqn_config.sample_size
    N = config.eqn_config.time_steps 
    d = config.eqn_config.dim

    gen_dataset =  tf.data.Dataset.from_generator(gen_data_generator, output_signature=(
        tf.TensorSpec(shape=(None, M, N + 1, 1)),
        tf.TensorSpec(shape=(None, M, N + 1, d)),
        tf.TensorSpec(shape=(None, M, N, d)),
        tf.TensorSpec(shape=(None, M, N + 1, None))))
    
    for element in gen_dataset.take(1):
        t, x, dw, u = element
    
    for element in gen_dataset.skip(1):
        t = tf.concat([t, element[0]], axis=0)
        x = tf.concat([x, element[1]], axis=0)
        dw = tf.concat([dw, element[2]], axis=0)
        u = tf.concat([u, element[3]], axis=0)
    
    dataset = tf.data.Dataset.from_tensor_slices((t, x, dw, u))
    save_dir = f'./dataset/{sde_name}_{option_name}_{dim}_{N}'
    tf.data.experimental.save(dataset, save_dir)
    print(dataset.element_spec)
    # return dataset

def save_dataset(dataset, path):
    tf.data.experimental.save(dataset, path)


     
if __name__ == '__main__':
    create_dataset("HW", "Swap", 1, 10)
    # # import tensorflow_datasets as tfds
    # dataset = tf.data.experimental.load(f'./dataset/SV_Swap_10', element_spec=(
    #     tf.TensorSpec(shape=(50, 100, 1)),
    #     tf.TensorSpec(shape=(50, 100, 20)),
    #     tf.TensorSpec(shape=(50, 99, 20)),
    #     tf.TensorSpec(shape=(50, 100, None))))
    # print("begin!")
    # # print(dataset.take(1))
    # # for a in dataset.take(1):
    # #     print(a)
    # i=0
    # for element in dataset.batch(1):
    #   print(i)
    #   print(element[1].shape)
    #   i += 1

    # create_dataset("GBM", "European", 1)
    # create_dataset("GBM", "Basket", 10)
    # create_dataset("SV", "Swap", 10)
    #create_dataset("GBM", "Lookback", 1)