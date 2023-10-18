import tensorflow as tf
from pricers import EuropeanPricer, FixIncomeEuropeanPricer
from function_space import DeepONet, DeepKernelONetwithPI, DeepKernelONetwithoutPI
from options import BaseOption
from sde import ItoProcessDriver, HullWhiteModel

class EarlyExercisePricer:
    """
    In this class, we initialize a list of no_net models and train each one
    recursively with the class EuropeanSolver. Then this class gives a pipeline of training process
    """
    def __init__(self, sde: ItoProcessDriver, option: BaseOption, config):
        self.sde = sde
        if isinstance(self.sde, HullWhiteModel):
            self.european_solver = FixIncomeEuropeanPricer(sde, option, config)
        else:
            self.european_solver = EuropeanPricer(sde, option, config)
        self.option = option
        self.eqn_config = config.eqn_config # config of the model
        self.net_config = config.net_config # config of the network
        self.dim = self.eqn_config.dim # num of assets
        self.exercise_date = self.eqn_config.exercise_date # the list of early exercise dates
        self.exercise_index = self.option.exer_index # [40, 60] <=> [0, 1] len=2 (for index) is the time index for early exercise
        if self.net_config.pi == "true":
            if self.net_config.kernel_type == "dense":
                self.no_nets = [DeepKernelONetwithPI(branch_layer=self.net_config.branch_layer_sizes, 
                                                    trunk_layer=self.net_config.trunk_layer_sizes, 
                                                    pi_layer=self.pi_layer_sizes,  
                                                    num_assets=self.dim, 
                                                    dense=True, 
                                                    num_outputs=6)
                                                    for _ in range (len(self.exercise_index) - 1)] # initialize a list of DeepKernelONetwithPI models with dense operator
            else:                                 
                self.no_nets = [DeepKernelONetwithPI(branch_layer=self.net_config.branch_layer_sizes, 
                                                    trunk_layer=self.net_config.trunk_layer_sizes, 
                                                    pi_layer=self.pi_layer_sizes, 
                                                    num_assets=self.dim, 
                                                    dense=False, 
                                                    num_outputs=6,
                                                    filters=self.net_config.num_filters, 
                                                    strides=self.net_config.num_strides)
                                                    for _ in range (len(self.exercise_index) - 1)] # initialize a list of DeepKernelONetwithPI models with CNN operator
        else:
            if  self.net_config.kernel_type == "no":
                self.no_nets = [DeepONet(branch_layer=self.net_config.branch_layer_sizes, 
                                        trunk_layer=self.net_config.trunk_layer_sizes,     
                                        activation = "tanh",
                                        )
                                        for _ in range (len(self.exercise_index) - 1)] # initialize a list of DeepONet models
            else:
                self.no_nets = [DeepKernelONetwithoutPI(branch_layer=self.net_config.branch_layer_sizes, 
                                                        trunk_layer=self.net_config.trunk_layer_sizes, 
                                                        dense=True, 
                                                        num_outputs=6,
                                                        filters=self.net_config.num_filters, 
                                                        strides=self.net_config.num_strides)
                                                        for _ in range (len(self.exercise_index) - 1)] # initialize a list of DeepKernelONetwithoutPI models
        assert len(self.no_nets) == len(self.exercise_index) - 1 # check whether the length of no_nets list satisfy the number of sub time intervals
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.net_config.lr,
            decay_steps=200,
            decay_rate=0.9
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-6) # set the optimizer of the European solver
        self.european_solver.compile(optimizer=self.optimizer)
    
    def slice_dataset(self, dataset: tf.data.Dataset, idx: int):
        r"""
        In this method, we split the dataset based on the index of early exercise dates:

        For example, if the global dataset has the following shapes:
        t: (None, M, 100, 1)
        x: (None, M, 100, 1)
        dw: (None, M, 99, 1)
        u: (None, M, 100, 3+1)

        Then we get the index of time step with idx_now = self.exercise_index[idx-1] and suppose we know that there are 20 time steps between two consecutive dates.
        then we just attain the dataset with shape:
        t_slice: (None, M, 20, 1)
        x_slice: (None, M, 20, 1)
        dw_slice: (None, M, 19, 1)
        u_slice: (None, M, 20, 3+1)
        
        return: the four tuple (t_slice, x_slice, dw_slice, u_slice)
        """
        if len(self.option.exer_dates) == 2:
            sub_dataset = dataset
        else:
            idx_now = self.exercise_index[idx-1] # the index of the time at the beginning of the interval
            if idx == len(self.exercise_index)-1:
                def slice_fn(t, x, dw, u):
                    t_slice = t[:, :, idx_now:, :]
                    x_slice = x[:, :, idx_now:, :]
                    dw_slice = dw[:, :, idx_now:, :]
                    u_slice = u[:, :, idx_now:, :]
                    return t_slice, x_slice, dw_slice, u_slice
            else:
                idx_fut = self.exercise_index[idx] # the index of the time at the end of the interval
                def slice_fn(t, x, dw, u):
                    t_slice = t[:, :, idx_now:idx_fut+1, :]
                    x_slice = x[:, :, idx_now:idx_fut+1, :]
                    dw_slice = dw[:, :, idx_now:idx_fut, :]
                    u_slice = u[:, :, idx_now:idx_fut+1, :]
                    return t_slice, x_slice, dw_slice, u_slice
            sub_dataset = dataset.map(slice_fn)
        return sub_dataset


    def fit(self, data: tf.data.Dataset, epochs: int, checkpoint_path: str):
        """
        The total training pipeline and we finally attain the no_nets in each sub-time interval.
        """
        # learning_rate = self.net_config.lr
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.net_config.lr,
            decay_steps=200,
            decay_rate=0.9
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-6) # set the optimizer of the European solver
        self.european_solver.compile(optimizer=optimizer)
        for idx in reversed(range(1, len(self.exercise_index))):
            # construct data set from the original dataset
            path = checkpoint_path + f"{idx}"
            print(f"=============begin {idx} th interval============")
            dataset = self.slice_dataset(data, idx) # slice the dataset from the total dataset based on the time interval between two consecutive exercise dates               
            self.european_solver.fit(x=dataset, epochs=epochs) # training the operator in the corresponded idx-th sub-interval
            self.european_solver.no_net.save_weights(path) # save the weights in the  idx-th path
            self.european_solver.no_net_target.load_weights(path)  # load the weights in the  idx-th path that means we initialize the weight for next task with the previous weight
            self.european_solver.step_to_next_round() # move ahead the index of task
            print(f"===============end {idx} th interval============")
        self.european_solver.reset_round() # reset the index of task to zero
        print("---end---")

    def load_weights_nets(self, checkpoint_path: str):
        for idx in range(len(self.no_nets)):
            self.no_nets[idx].load_weights(checkpoint_path + f"{idx + 1}")