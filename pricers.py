import tensorflow as tf
from function_space import DeepONet, DeepKernelONetwithPI, DeepKernelONetwithoutPI, BlackScholesFormula, ZeroCouponBondFormula
from sde import HestonModel, ItoProcessDriver, HullWhiteModel
from typing import Tuple
from options import BaseOption


class BaseBSDEPricer(tf.keras.Model):
    r"""
    1. The configs of the pricer is given by a json file: The config of the model is illustrated as follows:
       eqn_config: the config corresponded to model and training, which includes:
        -sde_name: the name of class name of sde
        -option_name: string: the name of class name of option
        -solver_name: string: the name of class name of pricer
        -exercise_date: string: the list of early exercise dates
        -style: boolean: call or put
        -T: tf.float.Tensor: the time horizon of the problem
        -dt: tf.float.Tensor: the step size of Monte Carlo simulation
        -dim: int: the number of assets 
        -time_steps: int: number of time steps of the whole time horizon
        -time_steps_one_year: number of time steps per year
        -strike_range: List[2], the distribution of moneyness of the product, List[0] gives the mean, \
            List[1] gives the standard deviation
        -r_range: List[2], the uniform distribution of the risk free rate
        -s_range: List[2], the uniform distribution of the volatility rate
        -rho_range: List[2], the uniform distribution of the correlation between assets
        -x_init: tf.float.Tensor, the initial state value
        -vol_init: tf.float.Tensor, the initial stochastic volatility value
        -batch_size: int, batch size of input functions and parameters
        -sample_size: int, number of paths of a single set of functions and parameters
        -sensors: int, input size of the deepOnet
        -initial_mode: three choices: "fixed"-simulate sde from a fixed initial value; 
            "partial fixed"-simulate sde from a fixed initial value just for each set of functions; 
            "random"-simulate sde from a randomed initial value; 

       net_config: the config corresponded to the keras.model of the neural network, which includes:
        -pi: a list of layer sizes of the permutation invariant network;
        -branch_layer_sizes: a list of layer sizes of the branch network;
        -trunk_layer_sizes: a list of layer sizes of the trunk network;
        -pi_layer_sizes:  a list of layer sizes of permutation invariant layer.
        -kernel_type: three choices: "dense"-dense operator as in paper of deepOnet; 
                                     "conv"-CNN kernel
                                     "no"-directly send parameters of function into the branch network.
        -num_filters: the num of filters if the kernel operator is CNN, this is valid only when kernel_type is "conv"
        -num_strides: the num of strides if the kernel operator is CNN, this is valid only when kernel_type is "conv"
        -lr: the learning rate
        -epochs: the epoch
        -alpha: the penalty of the interior loss

       val_config: the config for validation, which includes:
        -r_range: List[2], the uniform distribution of the risk free rate
        -s_range: List[2], the uniform distribution of the volatility rate
        -rho_range: List[2], the uniform distribution of the correlation between assets
        -x_init: tf.float.Tensor, the initial state value
        -vol_init: tf.float.Tensor, the initial stochastic volatility value
        -batch_size: int, batch size of input functions and parameters
        -sample_size: int, number of paths of a single set of functions and parameters

    2. This is the class to construct the tain step of the BSDE loss function and inference the value of price with certain time, state and coefficients, the data is generated from three
    external classes: the sde class which yield the data of input parameters. the option class which yields the
    input function data and give the payoff function and exact option price. In this docstring, we give global notations:
    B: batch size of functions
    M: number of paths of a single set of functions and parameters
    N: steps of the asset path
    d: number of assets
    The notation will be used in all methods is illustrated as:
    args:
       t: tf.float.Tensor time tensor with shape [B, M, N, 1]
       x: state tensor with shape [B, M, N, d], [B, M, N, 2d] under SV model, [B, M, N, 3d] under SV model for path dependent product
       u_hat: parameter related to input functions with shape [B, M, N, k], k is the number of parameters
    """

    def __init__(self, sde, option, config):
        super(BaseBSDEPricer, self).__init__()
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.option = option
        self.sde = sde
        self.dim = self.eqn_config.dim  # dimension of assets
        self.activation = None  # activation function of network
        if self.net_config.pi == "true":
            if self.net_config.kernel_type == "dense":
                self.no_net = DeepKernelONetwithPI(
                    branch_layer=self.net_config.branch_layer_sizes,
                    trunk_layer=self.net_config.trunk_layer_sizes,
                    pi_layer=self.net_config.pi_layer_sizes,
                    num_assets=self.dim,
                    dense=True,
                    num_outputs=self.net_config.num_outputs,
                    activation=self.activation,
                )  # DeepONet with kernel operator be dense operator
            else:
                self.no_net = DeepKernelONetwithPI(
                    branch_layer=self.net_config.branch_layer_sizes,
                    trunk_layer=self.net_config.trunk_layer_sizes,
                    pi_layer=self.net_config.pi_layer_sizes,
                    num_assets=self.dim,
                    dense=False,
                    num_outputs=self.net_config.num_outputs,
                    activation=self.activation,
                    filters=self.net_config.num_filters,
                    strides=self.net_config.num_strides,
                )  # DeepONet with kernel operator be CNN
        else:
            if self.net_config.kernel_type == "no":
                self.no_net = DeepONet(
                    branch_layer=self.net_config.branch_layer_sizes,
                    trunk_layer=self.net_config.trunk_layer_sizes,
                    activation=self.activation,
                )  # DeepONet without kernel operator
            else:
                self.no_net = DeepKernelONetwithoutPI(
                    branch_layer=self.net_config.branch_layer_sizes,
                    trunk_layer=self.net_config.trunk_layer_sizes,
                    dense=True,
                    num_outputs=self.net_config.num_outputs,
                    activation=self.activation,
                    filters=self.net_config.num_filters,
                    strides=self.net_config.num_strides,
                )  # DeepONet without pi layers
        self.time_horizon = (
            self.eqn_config.T
        )  # the time horizon $T$ of the problem which is fixed
        self.batch_size = (
            self.eqn_config.batch_size
        )  # the batch size of input functions
        self.samples = (
            self.eqn_config.sample_size
        )  # the number of paths sampled on each input functions set
        self.dt = self.eqn_config.dt  # time step size $\Delta t$
        self.time_steps = (
            self.eqn_config.time_steps
        )  # number of time steps $N = T/\Delta t$
        time_stamp = tf.range(0, self.time_horizon, self.dt)
        time_stamp = tf.reshape(time_stamp, [1, 1, self.time_steps, 1])
        self.time_stamp = tf.tile(
            time_stamp, [self.batch_size, self.samples, 1, 1]
        )  # time tensor of the problem
        self.alpha = self.net_config.alpha  # penalty of the interior loss

    def call(self, inputs: Tuple[tf.Tensor]) -> tf.Tensor:
        r"""
        The forward process is discussed under two different circumstance：
        when the no_net is the instance of DeepONet, then we just need to input the three tuple (t, x, u):
            t: time tensor with shape [B, M, N, 1]
            x: state tensor with shape [B, M, N, d], [B, M, N, 2d] under SV model, [B, M, N, 3d] under SV model for path dependent product
            u: parameter related to input functions with shape [B, M, N, k], k is the number of parameters
            return: the out put function value with shape [B, M, N, 1]
        when the no_net is not the instance of DeepONet, then the function values and kernel operaton will be made with the method split_uhat()
        and the output is a two tuple: u_c, u_p, then:
            t: time tensor with shape [B, M, N, 1]
            x: state tensor with shape [B, M, N, d], [B, M, N, 2d] under SV model, [B, M, N, 3d] under SV model for path dependent product
            u_c: function embedding tensor [B, M, N, k_1]
            u_p: other parameters tensor [B, M, N, k_2]
            return: the out put function value with shape [B, M, N, 1]
        """
        t, x, u_hat = inputs  # [B, M, N, 1], [B, M, N, d], [B, M, N, k]
        if type(self.no_net) == DeepONet or type(self.no_net) == BlackScholesFormula or type(self.no_net) == ZeroCouponBondFormula:
            y = self.no_net((t, x, u_hat))  # [B, M, N, 1]
        else:
            u_c, u_p = self.sde.split_uhat(u_hat)  # [B, M, N, k_1], [B, M, N, k_2]
            y = self.no_net((t, x, u_c, u_p))  # [B, M, N, 1]
        return y

    def get_gradient(self, inputs: Tuple[tf.Tensor]) -> tf.Tensor:
        """
        This calculate the gradient of the network output with respect to the state x
        """
        t, x, u_hat = inputs  # [N, M, N, 1], [B, M, N, d], [B, M, N, k]
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x)
            f = self((t, x, u_hat))  # [B, M, N, 1]
            grad = tape.gradient(f, x)  # [B, M, N, d]
        return grad

    def loss_interior(self, data: Tuple[tf.Tensor], training=None) -> tf.Tensor:
        raise NotImplementedError

    def loss_terminal(self, data: Tuple[tf.Tensor], training=None) -> tf.Tensor:
        raise NotImplementedError

    def loss_fn(self, data: Tuple[tf.Tensor], training=None) -> tf.Tensor:
        loss_interior, loss_tml = self.loss_interior(
            data, training
        ), self.loss_terminal(data, training)
        loss = self.alpha * self.loss_interior(data, training) + self.loss_terminal(
            data, training
        )
        return loss, loss_interior, loss_tml

    @tf.function
    def train_step(self, inputs: Tuple[tf.Tensor]) -> dict:
        raise NotImplementedError

    def drift_bsde(
        self, t: tf.Tensor, x: tf.Tensor, y: tf.Tensor, u_hat: tf.Tensor
    ) -> tf.Tensor:
        """
        the driver term of the BSDE, which is determined by the SDE
        """
        raise NotImplementedError

    def diffusion_bsde(
        self,
        t: tf.Tensor,
        x: tf.Tensor,
        grad: tf.Tensor,
        dw: tf.Tensor,
        u_hat: tf.Tensor,
    ) -> tf.Tensor:
        """
        the diffusion erm of the BSDE, which is determined by the SDE
        """
        raise NotImplementedError

    def payoff_func(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        """
        The payoff function of the time horizon and paths based on the given time tensor and state tensor
        """
        raise NotImplementedError


class MarkovianPricer(BaseBSDEPricer):
    """
    This is the class to construct the tain step of the BSDE loss function, the data is generated from three
    external classes: the sde class which yield the data of input parameters. the option class which yields the
    input function data and give the payoff function and exact option price.
    """

    def __init__(self, sde, option, config):
        super(MarkovianPricer, self).__init__(sde, option, config)

    def loss_interior(self, data: Tuple[tf.Tensor], training=None) -> tf.Tensor:
        r"""
        In this method, f is the no_net output value of t, x, u on the time horizon from 0 to T.
        f_now: the no_net output value of t, x, u on the time horizon from 0 to T-1, with shape [B, M, T-1, 1];
        f_pls: the no_net output value of t, x, u on the time horizon from 1 to T, with shape [B, M, T-1, 1];
        other variables is calculated based on these two variables

        This loss function give the BSDE loss through the time interval [T_0, T_1] and we let T_0 = 0, T-1 = T
        We first calculate the loss evaluated at each time point and the sum up.
        For a certain time point t_i, the loss is:
        l_i(y_i +\sum_{j=i}^N driver_bsde(t_j, X_j, y_j, z_j, u_hat) * dt + z_j * dW_j  - g(T, X_T, u_hat)) ** 2, where:
        y_i = no_net(t_j, X_j, u_hat) and z_i = sde.diffusion_onestep(t_i, X_i, u_hat) * \nabla y_i
        Then the tital loss is: \sum_{i=0}^Nl_i.

        :param t: time B x M x (T-1) x 1
        :param x: state B x M x (T-1) x D
        :param  u: B x K ->(repeat) B x M x (T-1) x K
        :return: value (B, M, (T-1), 1), gradient_x (B, M, (T-1), D)
        """
        (
            t,
            x,
            dw,
            u_hat,
        ) = data  # [B, M, N, 1], [B, M, N, d], [B, M, N-1, d], [B, M, N, k]
        steps = self.eqn_config.time_steps
        loss_interior = 0.0
        u_before = u_hat[:, :, :-1, :]  # [B, M, N-1, k]
        x_before = x[:, :, :-1, :]  # [B, M, N-1, d]
        t_before = t[:, :, :-1, :]  # [B, M, N-1, 1]
        f = self((t, x, u_hat))  # [B, M, N, 1]
        grad = self.get_gradient((t, x, u_hat))
        f_before = f[:, :, :-1, :]  # [B, M, N-1, 1]
        f_after = f[:, :, 1:, :]  # [B, M, N-1, 1]
        grad = grad[:, :, :-1, :]  # [B, M, N-1, d]
        for n in range(steps - 1):
            V_pls = f_after[:, :, n:, :]  # [B, M, N-n, 1]
            V_now = f_before[:, :, n:, :]  # [B, M, N-n, 1]
            V_hat = (
                V_now
                - self.drift_bsde(
                    t_before[:, :, n:, :],
                    x_before[:, :, n:, :],
                    V_now,
                    u_before[:, :, n:, :],
                )
                * self.dt
                + self.diffusion_bsde(
                    t_before[:, :, n:, :],
                    x_before[:, :, n:, :],
                    grad[:, :, n:, :],
                    dw[:, :, n:, :],
                    u_before[:, :, n:, :],
                )
            )
            tele_sum = tf.reduce_sum(V_pls - V_hat, axis=2)  # [B, M, 1]
            loss_interior += tf.reduce_mean(
                tf.square(
                    tele_sum + self.payoff_func(t, x, u_hat) - f_after[:, :, -1, :]
                )
            )  # sum([B, M, 1] - [B, M, 1]) -> []
        return loss_interior

    def loss_terminal(self, data: Tuple[tf.Tensor], training=None) -> tf.Tensor:
        t, x, _, u_hat = data
        f = self((t, x, u_hat))
        loss_tml = tf.reduce_mean(
            tf.square(
                f[:, :, -1, :] - self.payoff_func(t, x, u_hat)
            )  # sum([B, M, 1] - [B, M, 1]) -> []
        )
        return loss_tml

    def train_step(self, inputs: Tuple[tf.Tensor]):
        with tf.GradientTape() as tape:
            loss, loss_int, loss_tml = self.loss_fn(inputs)
            loss, loss_int, loss_tml = (
                tf.reduce_mean(loss),
                tf.reduce_mean(loss_int),
                tf.reduce_mean(loss_tml),
            )
            grad = tape.gradient(loss, self.no_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.no_net.trainable_variables))
        return {"loss": loss, "loss interior": loss_int, "loss terminal": loss_tml}

    def drift_bsde(
        self, t: tf.Tensor, x: tf.Tensor, y: tf.Tensor, u_hat: tf.Tensor
    ) -> tf.Tensor:  # get h function
        """
        the driver term is r*y for MTG property
        """
        return self.sde.driver_bsde(t, x, y, u_hat)  # [B, M, 1]

    def diffusion_bsde(
        self,
        t: tf.Tensor,
        x: tf.Tensor,
        grad: tf.Tensor,
        dw: tf.Tensor,
        u_hat: tf.Tensor,
    ) -> tf.Tensor:
        """
        grad: (B, M, N-1, d)
        dw: (B, M, N-1, d)
        give: \sigma(t, x) * grad
        for a batch of (t, x, par)
        """
        if not isinstance(
            self.sde, HestonModel
        ):  # for Heston model, the state is [S, V] so we should truncate the state to just asset
            x = x[..., : self.dim]  # [B, M, d]
            grad = grad[..., : self.dim]  # [B, M, d]
            v_tx = self.sde.diffusion_onestep(t, x[..., : self.dim], u_hat)  # [B, M, d]
        else:
            v_tx = self.sde.diffusion_onestep(
                t, x, u_hat
            )  # else, we do not this truncation
        z = tf.reduce_sum(v_tx * grad * dw, axis=-1, keepdims=True)  # [B, M, d]
        return z

    def payoff_func(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        """
        terminal payoff:
        in single model case, this is just the payoff terminal function;
        in multi model cases, this is max(payoff, no_net_target(t,x,u))
        t, x, u are the ones for the given time interval;
        In European case: the corresponded interval is the whole time horizon
        """
        payoff = self.option.payoff(t, x, u_hat)  # [B, M, 1]
        return payoff


class EuropeanPricer(MarkovianPricer):
    def __init__(self, sde, option, config):
        super(EuropeanPricer, self).__init__(sde, option, config)
        self.num_of_time_intervals_for_early_exercise = (
            0  # this gives the index of the round of training
        )
        if self.net_config.pi == "true":
            if self.net_config.kernel_type == "dense":
                self.no_net_target = DeepKernelONetwithPI(
                    branch_layer=self.net_config.branch_layer_sizes,
                    trunk_layer=self.net_config.trunk_layer_sizes,
                    pi_layer=self.pi_layer_sizes,
                    num_assets=self.dim,
                    dense=True,
                    num_outputs=self.net_config.num_outputs,
                    activation=self.activation,
                )
            else:
                self.no_net_target = DeepKernelONetwithPI(
                    branch_layer=self.net_config.branch_layer_sizes,
                    trunk_layer=self.net_config.trunk_layer_sizes,
                    pi_layer=self.pi_layer_sizes,
                    num_assets=self.dim,
                    dense=False,
                    num_outputs=self.net_config.num_outputs,
                    activation=self.activation,
                    filters=self.net_config.num_filters,
                    strides=self.net_config.num_strides,
                )
        else:
            if self.net_config.kernel_type == "no":
                self.no_net_target = DeepONet(
                    branch_layer=self.net_config.branch_layer_sizes,
                    trunk_layer=self.net_config.trunk_layer_sizes,
                    activation=self.activation,
                )
            else:
                self.no_net_target = DeepKernelONetwithoutPI(
                    branch_layer=self.net_config.branch_layer_sizes,
                    trunk_layer=self.net_config.trunk_layer_sizes,
                    dense=True,
                    num_outputs=self.net_config.num_outputs,
                    activation=self.activation,
                    filters=self.net_config.num_filters,
                    strides=self.net_config.num_strides,
                )
        self.no_net_target.trainable = False

    def step_to_next_round(self):
        self.num_of_time_intervals_for_early_exercise += 1

    def reset_round(self):
        self.num_of_time_intervals_for_early_exercise = 0

    def net_target_forward(self, inputs: Tuple[tf.Tensor]) -> tf.Tensor:
        """
        Same forward propagation as net_forward() but this is not trainable
        """
        t, x, u = inputs  # [B, M, N, 1], [B, M, N, d], [B, M, N, k]
        if type(self.no_net_target) == DeepONet or type(self.no_net) == BlackScholesFormula or type(self.no_net) == ZeroCouponBondFormula:
            y = self.no_net_target((t, x, u))  # [B, M, N, 1]
        else:
            u_c, u_p = self.sde.split_uhat(u)  # [B, M, N, k_1], [B, M, N, k_2]
            y = self.no_net_target((t, x, u_c, u_p))  # [B, M, N, 1]
        return y

    def payoff_func(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        """
        terminal payoff of each subproblem and when the exercise period is not zero,
        the terminal payoff is the maximum of early exercise and continuation value given by target network
        """
        t_last = tf.expand_dims(t[:, :, -1, :], axis=2)
        x_last = tf.expand_dims(x[:, :, -1, :], axis=2)
        u_last = tf.expand_dims(u_hat[:, :, -1, :], axis=2)
        if self.num_of_time_intervals_for_early_exercise != 0:
            cont_value = tf.squeeze(
                self.net_target_forward((t_last, x_last, u_last)), axis=2
            )
            early_payoff = self.option.payoff(x, u_hat)
            payoff = tf.maximum(early_payoff, cont_value)
        else:
            payoff = self.option.payoff(x, u_hat)
            cont = self.net_target_forward((t, x, u_hat))
        return payoff


class FixIncomeEuropeanPricer(EuropeanPricer):
    def __init__(self, sde, option, config):
        super(FixIncomeEuropeanPricer, self).__init__(sde, option, config)

    def payoff_func(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        """
        terminal payoff for each sub time-interval
        the terminal payoff is the maximum of early exercise and continuation value given by target network
        """
        t_last = tf.expand_dims(t[:, :, -1, :], axis=2)  # (B, M, 1, 1)
        x_last = tf.expand_dims(x[:, :, -1, :], axis=2)  # (B, M, 1, 1)
        u_last = tf.expand_dims(u_hat[:, :, -1, :], axis=2)  # (B, M, 1, k)
        if len(self.option.exer_dates) == 2:
            payoff = self.option.payoff_inter(
                t[:, :, -1, :], x[:, :, -1, :], u_hat[:, :, -1, :]
            )

        else:
            if self.num_of_time_intervals_for_early_exercise != 0:
                cont_value = tf.squeeze(
                    self.net_target_forward((t_last, x_last, u_last)), axis=2
                )  # (B, M, 1)
                early_payoff = self.option.payoff_inter(
                    t[:, :, -1, :], x[:, :, -1, :], u_hat[:, :, -1, :]
                )
                payoff = tf.maximum(early_payoff, cont_value)
            else:
                payoff = self.option.payoff_at_maturity(t, x, u_hat)
                cont = self.net_target_forward((t, x, u_hat))
        return payoff


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
        self.eqn_config = config.eqn_config  # config of the model
        self.net_config = config.net_config  # config of the network
        self.dim = self.eqn_config.dim  # num of assets
        self.exercise_date = (
            self.eqn_config.exercise_date
        )  # the list of early exercise dates
        self.exercise_index = (
            self.option.exer_index
        )  # [40, 60] <=> [0, 1] len=2 (for index) is the time index for early exercise
        if self.net_config.pi == "true":
            if self.net_config.kernel_type == "dense":
                self.no_nets = [
                    DeepKernelONetwithPI(
                        branch_layer=self.net_config.branch_layer_sizes,
                        trunk_layer=self.net_config.trunk_layer_sizes,
                        pi_layer=self.pi_layer_sizes,
                        num_assets=self.dim,
                        dense=True,
                        num_outputs=self.net_config.num_outputs,
                    )
                    for _ in range(len(self.exercise_index) - 1)
                ]  # initialize a list of DeepKernelONetwithPI models with dense operator
            else:
                self.no_nets = [
                    DeepKernelONetwithPI(
                        branch_layer=self.net_config.branch_layer_sizes,
                        trunk_layer=self.net_config.trunk_layer_sizes,
                        pi_layer=self.pi_layer_sizes,
                        num_assets=self.dim,
                        dense=False,
                        num_outputs=self.net_config.num_outputs,
                        filters=self.net_config.num_filters,
                        strides=self.net_config.num_strides,
                    )
                    for _ in range(len(self.exercise_index) - 1)
                ]  # initialize a list of DeepKernelONetwithPI models with CNN operator
        else:
            if self.net_config.kernel_type == "no":
                self.no_nets = [
                    DeepONet(
                        branch_layer=self.net_config.branch_layer_sizes,
                        trunk_layer=self.net_config.trunk_layer_sizes,
                        activation="tanh",
                    )
                    for _ in range(len(self.exercise_index) - 1)
                ]  # initialize a list of DeepONet models
            else:
                self.no_nets = [
                    DeepKernelONetwithoutPI(
                        branch_layer=self.net_config.branch_layer_sizes,
                        trunk_layer=self.net_config.trunk_layer_sizes,
                        dense=True,
                        num_outputs=self.net_config.num_outputs,
                        filters=self.net_config.num_filters,
                        strides=self.net_config.num_strides,
                    )
                    for _ in range(len(self.exercise_index) - 1)
                ]  # initialize a list of DeepKernelONetwithoutPI models
        assert (
            len(self.no_nets) == len(self.exercise_index) - 1
        )  # check whether the length of no_nets list satisfy the number of sub time intervals
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.net_config.lr, decay_steps=200, decay_rate=0.9
        )
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule, epsilon=1e-6
        )  # set the optimizer of the European solver
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
            idx_now = self.exercise_index[
                idx - 1
            ]  # the index of the time at the beginning of the interval
            if idx == len(self.exercise_index) - 1:

                def slice_fn(t, x, dw, u):
                    t_slice = t[:, :, idx_now:, :]
                    x_slice = x[:, :, idx_now:, :]
                    dw_slice = dw[:, :, idx_now:, :]
                    u_slice = u[:, :, idx_now:, :]
                    return t_slice, x_slice, dw_slice, u_slice

            else:
                idx_fut = self.exercise_index[
                    idx
                ]  # the index of the time at the end of the interval

                def slice_fn(t, x, dw, u):
                    t_slice = t[:, :, idx_now : idx_fut + 1, :]
                    x_slice = x[:, :, idx_now : idx_fut + 1, :]
                    dw_slice = dw[:, :, idx_now:idx_fut, :]
                    u_slice = u[:, :, idx_now : idx_fut + 1, :]
                    return t_slice, x_slice, dw_slice, u_slice

            sub_dataset = dataset.map(slice_fn)
        return sub_dataset

    def fit(self, data: tf.data.Dataset, epochs: int, checkpoint_path: str):
        """
        The total training pipeline and we finally attain the no_nets in each sub-time interval.
        """
        # learning_rate = self.net_config.lr
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.net_config.lr, decay_steps=200, decay_rate=0.9
        )
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule, epsilon=1e-6
        )  # set the optimizer of the European solver
        self.european_solver.compile(optimizer=optimizer)
        for idx in reversed(range(1, len(self.exercise_index))):
            # construct data set from the original dataset
            path = checkpoint_path + f"{idx}"
            print(f"=============begin {idx} th interval============")
            dataset = self.slice_dataset(
                data, idx
            )  # slice the dataset from the total dataset based on the time interval between two consecutive exercise dates
            self.european_solver.fit(
                x=dataset, epochs=epochs
            )  # training the operator in the corresponded idx-th sub-interval
            self.european_solver.no_net.save_weights(
                path
            )  # save the weights in the  idx-th path
            self.european_solver.no_net_target.load_weights(
                path
            )  # load the weights in the  idx-th path that means we initialize the weight for next task with the previous weight
            self.european_solver.step_to_next_round()  # move ahead the index of task
            print(f"===============end {idx} th interval============")
        self.european_solver.reset_round()  # reset the index of task to zero
        print("---end---")

    def load_weights_nets(self, checkpoint_path: str):
        for idx in range(len(self.no_nets)):
            self.no_nets[idx].load_weights(checkpoint_path + f"{idx + 1}")
