import json
import munch
import tensorflow as tf
import sde as eqn
import sde_jump as eqn_j
import options as opts
from options import GeometricAsian, LookbackOption
from data_generators import DiffusionModelGenerator, JumpDiffusionModelGenerator
import global_constants as constants

path_dependent_list = [
    GeometricAsian, 
    LookbackOption,
]
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
    "Swaption",
    "SwaptionFirst",
    "SwaptionLast",
    "TimeEuropean",
    "BermudanPut",
]
dim_list = [1, 3, 5, 10, 20]


def create_dataset(
    sde_name: str, option_name: str, dim: int = 10, num_of_batches: int = 100
):
    if (
        (sde_name not in constants.SDE_LIST)
        or (option_name not in constants.OPTION_LIST)
        or (dim not in constants.DIM_LIST)
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
    if sde_name in constants.JUMP_LIST:
        sde = getattr(eqn_j, config.eqn_config.sde_name)(config)
    else:
        sde = getattr(eqn, config.eqn_config.sde_name)(config)
    option = getattr(opts, config.eqn_config.option_name)(config)
    M = config.eqn_config.sample_size
    N = config.eqn_config.time_steps
    d = config.eqn_config.dim
    # generate data without jump
    if sde_name not in constants.JUMP_LIST:

        data_generator = DiffusionModelGenerator(
            sde, config.eqn_config, option, num_of_batches
        )
        def gen_data_generator():
            for i in range(data_generator.__len__()):
                yield data_generator.__getitem__(i)[0]

        if sde_name != "SV" and type(option) not in path_dependent_list:
            gen_dataset = tf.data.Dataset.from_generator(
                gen_data_generator,
                output_signature=(
                    tf.TensorSpec(shape=(None, M, N + 1, 1)),
                    tf.TensorSpec(shape=(None, M, N + 1, d)),
                    tf.TensorSpec(shape=(None, M, N, d)),
                    tf.TensorSpec(shape=(None, M, N + 1, None)),
                ),
            )
        else:
            if sde_name == "SV":
                gen_dataset = tf.data.Dataset.from_generator(
                    gen_data_generator,
                    output_signature=(
                        tf.TensorSpec(shape=(None, M, N + 1, 1)),
                        tf.TensorSpec(shape=(None, M, N + 1, 2 * d)),
                        tf.TensorSpec(shape=(None, M, N, 2 * d)),
                        tf.TensorSpec(shape=(None, M, N + 1, None)),
                    ),
                )
            else:
                gen_dataset = tf.data.Dataset.from_generator(
                    gen_data_generator,
                    output_signature=(
                        tf.TensorSpec(shape=(None, M, N + 1, 1)),
                        tf.TensorSpec(shape=(None, M, N + 1, 2 * d)),
                        tf.TensorSpec(shape=(None, M, N, d)),
                        tf.TensorSpec(shape=(None, M, N + 1, None)),
                    ),
                )
        for element in gen_dataset.take(1):
            t, x, dw, u = element

        for element in gen_dataset.skip(1):
            t = tf.concat([t, element[0]], axis=0)
            x = tf.concat([x, element[1]], axis=0)
            dw = tf.concat([dw, element[2]], axis=0)
            u = tf.concat([u, element[3]], axis=0)
        dataset = tf.data.Dataset.from_tensor_slices((t, x, dw, u))
    # generate data with jump
    else:
        data_generator = JumpDiffusionModelGenerator(
            sde, config.eqn_config, option, num_of_batches
        )
        def gen_data_generator():
            for i in range(data_generator.__len__()):
                yield data_generator.__getitem__(i)[0]

        if sde_name != "SV" and type(option) not in path_dependent_list:
            gen_dataset = tf.data.Dataset.from_generator(
                gen_data_generator,
                output_signature=(
                    tf.TensorSpec(shape=(None, M, N + 1, 1)),
                    tf.TensorSpec(shape=(None, M, N + 1, d)),
                    tf.TensorSpec(shape=(None, M, N, d)), # for dw
                    tf.TensorSpec(shape=(None, M, N, d)), # for happen
                    tf.TensorSpec(shape=(None, M, N, d)), # for z
                    tf.TensorSpec(shape=(None, M, N + 1, None)),
                ),
            )
        else:
            if sde_name == "SV":
                gen_dataset = tf.data.Dataset.from_generator(
                    gen_data_generator,
                    output_signature=(
                        tf.TensorSpec(shape=(None, M, N + 1, 1)),
                        tf.TensorSpec(shape=(None, M, N + 1, 2 * d)),
                        tf.TensorSpec(shape=(None, M, N, 2 * d)), # for dw
                        tf.TensorSpec(shape=(None, M, N, d)), # for happen
                        tf.TensorSpec(shape=(None, M, N, d)), # for z
                        tf.TensorSpec(shape=(None, M, N + 1, None)),
                    ),
                )
            else:
                gen_dataset = tf.data.Dataset.from_generator(
                    gen_data_generator,
                    output_signature=(
                        tf.TensorSpec(shape=(None, M, N + 1, 1)),
                        tf.TensorSpec(shape=(None, M, N + 1, 2 * d)),
                        tf.TensorSpec(shape=(None, M, N, d)), # for dw
                        tf.TensorSpec(shape=(None, M, N, d)), # for happen
                        tf.TensorSpec(shape=(None, M, N, d)), # for z
                        tf.TensorSpec(shape=(None, M, N + 1, None)),
                    ),
                )
        for element in gen_dataset.take(1):
            t, x, dw, h, z, u = element

        for element in gen_dataset.skip(1):
            t = tf.concat([t, element[0]], axis=0)
            x = tf.concat([x, element[1]], axis=0)
            dw = tf.concat([dw, element[2]], axis=0)
            h = tf.concat([h, element[3]], axis=0)
            z = tf.concat([z, element[4]], axis=0)
            u = tf.concat([u, element[5]], axis=0)
        dataset = tf.data.Dataset.from_tensor_slices((t, x, dw, h, z, u))
        
    save_dir = f"./dataset/{sde_name}_{option_name}_{dim}_{N}"
    tf.data.experimental.save(dataset, save_dir)
    print(dataset.element_spec)

def save_dataset(dataset, path):
    tf.data.experimental.save(dataset, path)

if __name__ == "__main__":
    create_dataset("HW", "Swaption", 1, 50)
