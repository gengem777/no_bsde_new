from global_constants import SDE_LIST, OPTION_LIST, DIM_LIST
import munch
import json


def load_config(sde_name: str, option_name: str, dim: int = 1):
    """
    This function is for introducing config json files into test functions
    """
    if (
        (sde_name not in SDE_LIST)
        or (option_name not in OPTION_LIST)
        or (dim not in DIM_LIST)
    ):
        raise ValueError(
            f"please input right sde_name in {SDE_LIST},\
                          option_name in {OPTION_LIST} and dim in {DIM_LIST}"
        )
    else:
        json_path = f"./configs/{sde_name}_{option_name}_{dim}.json"
    with open(json_path) as json_data_file:
        config = json.load(json_data_file)

    config = munch.munchify(config)
    return config