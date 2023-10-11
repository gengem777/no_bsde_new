import tensorflow as tf
from abc import ABC
from function_space import DeepONet, DeepKernelONetwithPI, DeepKernelONetwithoutPI
from sde import HestonModel
from typing import Tuple

class BaseBSDEPricer(tf.keras.Model):
    """
    This is the class to construct the tain step of the BSDE loss function, the data is generated from three
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
        self.dim = self.eqn_config.dim # dimension of assets
        self.branch_layers = self.net_config.branch_layers # a list: neurons of each layer in branch network
        self.trunk_layers = self.net_config.trunk_layers # a list: neurons of each layer in trunk network
        self.filters = self.net_config.filters # filters when the kernel operator is CNN
        self.strides = self.net_config.strides # strides when the kernel operator is CNN
        self.pi_layers = self.net_config.pi_layers # a list: neurons of each layer in permutation invariant network
        self.activation = None # activation function of network
        if self.net_config.pi == "true":
            if self.net_config.kernel_type == "dense":
                self.no_net = DeepKernelONetwithPI(branch_layer=self.branch_layers, 
                                                    trunk_layer=self.trunk_layers, 
                                                    pi_layer=self.pi_layers, 
                                                    num_assets=self.dim, 
                                                    dense=True, 
                                                    num_outputs=6,
                                                    activation=self.activation) # DeepONet with kernel operator be dense operator
            else:                                 
                self.no_net = DeepKernelONetwithPI(branch_layer=self.branch_layers, 
                                                    trunk_layer=self.trunk_layers, 
                                                    pi_layer=self.pi_layers, 
                                                    num_assets=self.dim, 
                                                    dense=False, 
                                                    num_outputs=6,
                                                    activation=self.activation,
                                                    filters=self.filters, 
                                                    strides=self.strides) # DeepONet with kernel operator be CNN
        else:
            if  self.net_config.kernel_type == "no":
                self.no_net = DeepONet(branch_layer=self.branch_layers, 
                                       trunk_layer=self.trunk_layers,
                                       activation = self.activation) # DeepONet without kernel operator
            else:
                self.no_net = DeepKernelONetwithoutPI(branch_layer=self.branch_layers, 
                                                        trunk_layer=self.trunk_layers, 
                                                        dense=True, 
                                                        num_outputs=6,
                                                        activation=self.activation,
                                                        filters=self.filters, 
                                                        strides=self.strides) # DeepONet without pi layers
        self.time_horizon = self.eqn_config.T # the time horizon $T$ of the problem which is fixed
        self.batch_size = self.eqn_config.batch_size # the batch size of input functions
        self.samples = self.eqn_config.sample_size # the number of paths sampled on each input functions set
        self.dt = self.eqn_config.dt # time step size $\Delta t$
        self.time_steps = self.eqn_config.time_steps # number of time steps $N = T/\Delta t$
        time_stamp = tf.range(0, self.time_horizon, self.dt)
        time_stamp = tf.reshape(time_stamp, [1, 1, self.time_steps, 1])
        self.time_stamp = tf.tile(time_stamp, [self.batch_size, self.samples, 1, 1]) # time tensor of the problem
        self.alpha = self.net_config.alpha # penalty of the interior loss

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
        t, x, u_hat = inputs
        if type(self.no_net) == DeepONet:
            y = self.no_net((t, x, u_hat))
        # print(t.shape, x.shape, u.shape)
        else:
            u_c, u_p = self.sde.split_uhat(u_hat)
            # print(u_c.shape, u_p.shape)
            y = self.no_net((t, x, u_c, u_p))
        return y
        
    def loss_fn(self, data: Tuple[tf.Tensor], training=None):
        raise NotImplementedError

    @tf.function
    def train_step(self, inputs: Tuple[tf.Tensor]) -> dict:
        raise NotImplementedError

    def drift_bsde(self, t: tf.Tensor, x: tf.Tensor, y: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor: 
        """
        the driver term of the BSDE, which is determined by the SDE
        """
        raise NotImplementedError

    def diffusion_bsde(self, t: tf.Tensor, x: tf.Tensor, grad: tf.Tensor, dw: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
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

    def loss_fn(self, data: Tuple[tf.Tensor], training=None) -> Tuple[tf.Tensor]:
        """
        :param t: time B x M x (T-1) x 1
        :param x: state B x M x (T-1) x D
        :param  u: B x K ->(repeat) B x M x (T-1) x K
        :return: value (B, M, (T-1), 1), gradient_x (B, M, (T-1), D)
        """
        t, x, dw, u_hat = data
        steps = self.eqn_config.time_steps
        loss_interior = 0.0
        u_now = u_hat[:, :, :-1, :]
        x_now = x[:, :, :-1, :]
        t_now = t[:, :, :-1, :]
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x)
            f = self.call((t, x, u_hat))
            grad = tape.gradient(f, x) 
        f_now = f[:, :, :-1, :]
        f_pls = f[:, :, 1:, :]
        grad = grad[:,:,:-1,:]
        for n in range(steps-1):
            V_pls = f_pls[:, :, n:, :]
            V_now = f_now[:, :, n:, :]
            V_hat = V_now - self.drift_bsde(
                t_now[:,:, n:,:], 
                x_now[:,:, n:,:], 
                V_now,
                u_now[:,:, n:,:],
            ) * self.dt + self.diffusion_bsde(
                t_now[:,:, n:,:], 
                x_now[:,:, n:,:], 
                grad[:,:, n:,:], 
                dw[:,:, n:,:], 
                u_now[:,:, n:,:],
            ) 
            tele_sum = tf.reduce_sum(V_pls - V_hat, axis=2)
            loss_interior += tf.reduce_mean(tf.square(tele_sum + self.payoff_func(t, x, u_hat) - f_pls[:, :, -1, :]))
        
        loss_tml = tf.reduce_mean(tf.square(f_pls[:, :, -1, :] - self.payoff_func(t, x, u_hat))) 
        loss = self.alpha * loss_tml + loss_interior
        return loss, loss_interior, loss_tml

    def train_step(self, inputs: Tuple[tf.Tensor]):
        with tf.GradientTape() as tape:
            loss, loss_int, loss_tml = self.loss_fn(inputs)
            loss, loss_int, loss_tml = tf.reduce_mean(loss), tf.reduce_mean(loss_int), tf.reduce_mean(loss_tml)
            grad = tape.gradient(loss, self.no_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.no_net.trainable_variables))
        return {"loss": loss,
                "loss interior": loss_int,
                "loss terminal": loss_tml}
    

    def drift_bsde(self, t: tf.Tensor, x: tf.Tensor, y: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:  # get h function
        """
        the driver term is r*y for MTG property
        """
        return self.sde.driver_bsde(t, x, y, u_hat) 

    def diffusion_bsde(self, t: tf.Tensor, x: tf.Tensor, grad: tf.Tensor, dw: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        """
        grad: (B, M, N-1, d)
        dw: (B, M, N-1, d)
        give: \sigma(t, x) * grad
        for a batch of (t, x, par)
        """
        if not isinstance(self.sde, HestonModel):
            x = x[...,:self.dim]
            grad = grad[...,:self.dim]
            v_tx = self.sde.diffusion_onestep(t, x[...,:self.dim], u_hat)
        else:
            v_tx = self.sde.diffusion_onestep(t, x, u_hat)
        z = tf.reduce_sum(v_tx * grad * dw, axis=-1, keepdims=True)
        return z

    def payoff_func(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        """
        terminal payoff:
        in single model case, this is just the payoff terminal function;
        in multi model cases, this is max(payoff, no_net_target(t,x,u))
        t, x, u are the ones for the given time interval; 
        In European case: the corresponded interval is the whole time horizon
        """
        payoff = self.option.payoff(t, x, u_hat)
        return payoff

class EuropeanSolver(MarkovianPricer):
    def __init__(self, sde, option, config):
        super(EuropeanSolver, self).__init__(sde, option, config)
        self.train_round = 0 # this gives the index of the round of training
        if self.net_config.pi == "true":
            if self.net_config.kernel_type == "dense":
                self.no_net_target = DeepKernelONetwithPI(branch_layer=self.branch_layers, 
                                                    trunk_layer=self.trunk_layers, 
                                                    pi_layer=self.pi_layers, 
                                                    num_assets=self.dim, 
                                                    dense=True, 
                                                    num_outputs=6,
                                                    activation=self.activation)
            else:                                 
                self.no_net_target = DeepKernelONetwithPI(branch_layer=self.branch_layers, 
                                                    trunk_layer=self.trunk_layers, 
                                                    pi_layer=self.pi_layers, 
                                                    num_assets=self.dim, 
                                                    dense=False, 
                                                    num_outputs=6,
                                                    activation=self.activation,
                                                    filters=self.filters, 
                                                    strides=self.strides)
        else:
            if  self.net_config.kernel_type == "no":
                self.no_net_target = DeepONet(branch_layer=self.branch_layers, 
                                       trunk_layer=self.trunk_layers,
                                       activation = self.activation)
            else:
                self.no_net_target = DeepKernelONetwithoutPI(branch_layer=self.branch_layers, 
                                                        trunk_layer=self.trunk_layers, 
                                                        dense=True, 
                                                        num_outputs=6,
                                                        activation=self.activation,
                                                        filters=self.filters, 
                                                        strides=self.strides)
        self.no_net_target.trainable = False
        
    def step_to_next_round(self):
        self.train_round += 1
    
    def reset_round(self):
        self.train_round = 0

    def net_target_forward(self, inputs: Tuple[tf.Tensor]) -> tf.Tensor:
        """
        Same forward propagation as net_forward() but this is not trainable
        """
        t, x, u = inputs
        if type(self.no_net_target) == DeepONet:
            y = self.no_net_target((t, x, u))
        else:
            u_c, u_p = self.sde.split_uhat(u)
            y = self.no_net_target((t, x, u_c, u_p))
        return y
    
    def payoff_func(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        """
        terminal payoff of each subproblem and when the traiing round is not zero,
        the terminal payoff is the maximum of early exercise and continuation value given by target network
        """
        t_last = tf.expand_dims(t[:,:,-1,:], axis=2)
        x_last = tf.expand_dims(x[:,:,-1,:], axis=2)
        u_last = tf.expand_dims(u_hat[:,:,-1,:], axis=2)
        if self.train_round != 0:
            cont_value = tf.squeeze(self.net_target_forward((t_last, x_last, u_last)), axis=2)
            early_payoff = self.option.payoff(x, u_hat)
            payoff = tf.maximum(early_payoff, cont_value)
        else:
            payoff = self.option.payoff(x, u_hat)
            cont = self.net_target_forward((t, x, u_hat))
        return payoff # (B, M, 1)
    
class FixIncomeEuropeanSolver(EuropeanSolver):
    def __init__(self, sde, option, config):
        super(FixIncomeEuropeanSolver, self).__init__(sde, option, config)  
    
    def payoff_func(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        """
        terminal payoff for each sub time-interval
        the terminal payoff is the maximum of early exercise and continuation value given by target network
        """
        t_last = tf.expand_dims(t[:,:,-1,:], axis=2) # (B, M, 1, 1)
        x_last = tf.expand_dims(x[:,:,-1,:], axis=2) # (B, M, 1, 1)
        u_last = tf.expand_dims(u_hat[:,:,-1,:], axis=2) # (B, M, 1, k)
        if len(self.option.exer_dates) == 2:
            payoff = self.option.payoff_inter(t[:,:,-1,:], x[:,:,-1,:], u_hat[:,:,-1,:])
        
        else:
            if self.train_round != 0:
                cont_value = tf.squeeze(self.net_target_forward((t_last, x_last, u_last)), axis=2) # (B, M, 1)
                early_payoff = self.option.payoff_inter(t[:,:,-1,:], x[:,:,-1,:], u_hat[:,:,-1,:])
                payoff = tf.maximum(early_payoff, cont_value)
            else:
                payoff = self.option.payoff_at_maturity(t, x, u_hat)
                cont = self.net_target_forward((t, x, u_hat))
        return payoff 
