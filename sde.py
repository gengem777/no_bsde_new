from abc import ABC, abstractmethod
import tensorflow as tf
import tensorflow_probability as tfp
from function_representors import QuadraticRepresentor, ExponentialDecayRepresentor, ConstantRepresentor


class ItoProcessDriver(ABC):
    """Abstract class for Ito processes, themselves driven by another Ito process B(t).
    This class represents the solution to an Ito stochastic differential equation with jump diffusion of the form
    $dX_t = r * X_t dt + \sigma(X(t), t) X_t dB(t)
    1. do initialization of X_0
    2. one step forward simulation
    3. simulate the path and output path tensor with Brownian motion increment tensor
    The class ItoProcessDriven implements common methods for sampling from the solution to this SDE,
    """

    def __init__(self, config):  # TODO
        """
        config is the set of the hyper-parameters
        we give the notations:
        dim: dimension of assets
        T; time horizon of simulation
        dt: length of time increments
        N: number of time steps
        x_init: initial state value
        vol_init: inital stochastic volatility value
        """
        self.config = config.eqn_config
        self.val_config = config.val_config
        self.dim = self.config.dim
        self.initial_mode = self.config.initial_mode
        self.x_init = self.config.x_init
        self.vol_init = self.config.vol_init
        self.range_list = []
        self.val_range_list = []
        self.representor = ConstantRepresentor(config)

    def get_batch_size(self, u_hat: tf.Tensor):
        return tf.shape(u_hat)[0]

    def initial_sampler(self, u_hat: tf.Tensor, samples: int) -> tf.Tensor:
        """
        Initial sampling of the asset price:
        In all SDE related classes, we denote:
          batch_size as B,
          sample_size for per input function as M,
          time step for the tensor as N,
          dimension of the state variable as d.
        return: float tensor with shape [B, M, d]
        """
        batch_size = self.get_batch_size(u_hat)
        dimension = self.config.dim
        dist = tfp.distributions.TruncatedNormal(
            loc=self.x_init,
            scale=self.x_init * 0.2,
            low=self.x_init * 0.05,
            high=self.x_init * 3.0,
        )
        if self.initial_mode == "random":
            state = tf.reshape(
                dist.sample(batch_size * samples * dimension),
                [batch_size, samples, dimension],
            )

        elif self.initial_mode == "fixed":
            state = self.x_init * tf.ones(shape=(batch_size, samples, dimension))

        elif self.initial_mode == "partial_fixed":
            state = tf.reshape(dist.sample(batch_size), [batch_size, 1, 1])
            state = tf.tile(state, [1, samples, dimension])
        return state

    def drift(self, time: tf.Tensor, state: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        r"""
        Computes the drift \mu(t, X_t) of this stochastic process.
        operator setting
        mu: (batch_size, 1)
        param state: (batch_size, path_size, dim)
        return: batch_size, path_size, dim)
        """
        batch = tf.shape(u_hat)[0]
        mu = tf.reshape(u_hat[:, 0], [batch, 1, 1])
        return mu * state

    @abstractmethod
    def diffusion(
        self, state: tf.Tensor, time: tf.Tensor, u_hat: tf.Tensor
    ) -> tf.Tensor:
        raise NotImplementedError

    @abstractmethod
    def diffusion_onestep(
        self, time_tensor: tf.Tensor, state_tensor: tf.Tensor, u_hat: tf.Tensor
    ):
        r"""
        get \sigma(t, X_t) with shape (B,M,N,d) sequence operation
        """
        raise NotImplementedError

    def drift_onestep(
        self, time_tensor: tf.Tensor, state_tensor: tf.Tensor, u_hat: tf.Tensor
    ):
        """
        all inputs are [batch, sample, T, dim] like
        what we do is calculate the drift for the whole tensor
        """
        raise NotImplementedError

    def driver_bsde(
        self, t: tf.Tensor, x: tf.Tensor, y: tf.Tensor, u_hat: tf.Tensor
    ) -> tf.Tensor:
        """
        this yields the driver term of the BSDE coupled with this SDE
        param t: [B, M, N, 1]
        param x: [B, M, N, d]
        param y: [B, M, N, 1]
        param u_hat: [B, M, N, 4]
        """
        raise NotImplementedError

    def corr_matrix(self, state: tf.Tensor, u_hat: tf.Tensor):
        r"""
        In base class the corr is just d x d identity matrix:
        	\begin{equation*}
		   \left(\begin{array}{cccc}
			1 & 0 & \ldots & 0 \\
			0 & 1 & \ldots & 0 \\
			\vdots & \vdots & \ddots & 0 \\
			0 & 0 & 0 & 1
		\end{array}\right)
	    \end{equation*}
        with rhe shape [B, M, d, d]
        """
        batch = tf.shape(u_hat)[0]
        samples = tf.shape(state)[1]
        corr = tf.eye(self.dim, batch_shape=[batch, samples])  # [B, M, d, d]
        return corr

    def brownian_motion(self, state: tf.Tensor, u_hat: tf.Tensor):
        """
        generate non iid Brownian Motion increments for a certain time step
        the time increment is dt, the way to calculate the dependent BM increment:
        1. calculate corr matrix with the function corr_matrix (at least in this class)
        2. do cholasky decomposition on corr matrix to a low diagnal matrix L
        3. multiply the matrix L with the iid BMs
        denote dim as the dimension of asset, the return shape is (batch, paths, dim), and
        (batch, paths, dim * 2) under stochastic vol cases.
        """
        dt = self.config.dt
        batch_size = tf.shape(u_hat)[0]
        samples = tf.shape(state)[1]
        actual_dim = tf.shape(state)[2]  # in SV model actual_dim = 2* dim =/= dim
        state_asset = state[..., : self.dim]  # [B, M, d]
        corr = self.corr_matrix(state_asset, u_hat)  # [B, M, d, d]
        cholesky_matrix = tf.linalg.cholesky(corr)  # [B, M, d, d]
        white_noise = tf.random.normal(
            [batch_size, samples, actual_dim], mean=0.0, stddev=tf.sqrt(dt)
        )  # [B, M, d]
        state_noise = tf.einsum(
            "...ij,...j->...i", cholesky_matrix, white_noise[..., : self.dim]
        )  # [B, M, d]
        vol_noise = white_noise[..., self.dim :]  # [B, M, d]
        return tf.concat([state_noise, vol_noise], axis=-1)  # [B, M, 2d]

    def euler_maryama_step(self, time: tf.Tensor, state: tf.Tensor, u_hat: tf.Tensor):
        """
        :param state: tf.Tensor
            The current state of the process.
            Shape is [batch, samples, dim].
        :param time: tf.Tensor
            The current time of the process. A scalar.
        :param noise: tf.Tensor
            The noise of the driving process.
            Shape is [batch, samples, dim].
        :param timestep : tf.Tensor
            A scalar; dt, the amount of time into the future in which we are stepping.
        :return (state, time) : (tf.Tensor, tf.Tensor)
            If this Ito process is X(t), the return value is (X(t+dt), dBt, St_*YdNt).
        """
        dt = self.config.dt
        noise = self.brownian_motion(state, u_hat)  # [B, M, d]
        drift = self.drift(time, state, u_hat)  # [B, M, d]
        diffusion = self.diffusion(time, state, u_hat)  # [B, M, d]
        increment = drift * dt + diffusion * noise  # [B, M, d]
        return state + increment, noise  # [B, M, d], [B, M, d]

    def sde_simulation(self, u_hat: tf.Tensor, samples: int):
        """
        the whole simulation process with each step iterated by method euler_maryama_step(),
        return is a tuple of tensors recording the paths of state and BM increments
        x: path of states, shape: (batch_size, path_size, num_timesteps, dim)
        dw: path of BM increments, shape: (batch_size, path_size, num_timesteps-1, dim)
        """
        stepsize = self.config.dt
        time_steps = self.config.time_steps
        state_process = tf.TensorArray(tf.float32, size=time_steps + 1)
        brownian_increments = tf.TensorArray(tf.float32, size=time_steps)
        state = self.initial_sampler(u_hat, samples)

        time = tf.constant(0.0)
        state_process = state_process.write(0, state)
        current_index = 1
        while current_index <= time_steps:
            state, dw = self.euler_maryama_step(
                time, state, u_hat
            )  # [B, M, d], [B, M, d]
            state_process = state_process.write(current_index, state)
            brownian_increments = brownian_increments.write(current_index - 1, dw)
            current_index += 1
            time += stepsize
        x = state_process.stack()
        dw = brownian_increments.stack()
        x = tf.transpose(x, perm=[1, 2, 0, 3])
        dw = tf.transpose(dw, perm=[1, 2, 0, 3])
        return x, dw  # [B, M, N, d], [B, M, N-1, d]

    def sample_parameters(self, N=100, training=True):  # N is the number of batch size
        """
        sample paremeters for simulating SDE given the range of each parameter from the config file,
        given the number of parameters need to be samples as k:
        the return tensorshape is [batch_size, k]
        """
        num_params = int(N * self.config.batch_size)
        if training:
            num_params = int(N * self.config.batch_size)
            return tf.concat(
                [
                    tf.random.uniform([num_params, 1], minval=p[0], maxval=p[1])
                    for p in self.range_list
                ],
                axis=1,
            )  # k = len(self.range_list), [B, k]
        else:
            num_params = int(N * self.val_config.batch_size)
            return tf.concat(
                [
                    tf.random.uniform([num_params, 1], minval=p[0], maxval=p[1])
                    for p in self.val_range_list
                ],
                axis=1,
            )  # k = len(self.range_list), [B, k]

    def expand_batch_inputs_dim(self, u_hat: tf.Tensor):
        """
        In order for consistence between input shape of parameter tensor and input state tensor into the network
        we need to unify the dimension. This method is to expand the dimension of 2-d tensor to 4-d using tf.tile method

        input: the parmeter tensor with shape [batch_size, K], K is the dim of parameters
        output: the expanded parameter [batch_size, sample_size, time_steps, K]
        we denote sample_size as M and time_steps as N.
        """
        u_hat = tf.reshape(u_hat, [u_hat.shape[0], 1, 1, u_hat.shape[-1]])
        u_hat = tf.tile(
            u_hat, [1, self.config.sample_size, self.config.time_steps + 1, 1]
        )
        return u_hat  # k = len(self.range_list), [B, M, N, k]

    def split_uhat(self, u_hat: tf.Tensor):
        """
        This method is to calculate the rate and volatility curve given a batch of parameters
        For example, GBM case, the parameters sampled has dimension batch_size + (3). where K=3
        and batch_size = (B, M, N)
        Then this function calculate the curve function based on parameter $\mu$ and $\sigma$
        return $\mu(t)$ and $\sigma(t)$ on the given support grid and return the batch of moneyness K
        Then Then return is a tuple of two tensors: (u_curve, u_param)
        u_curve: batch_size + (time_steps, num_curves), u_param = batch_size + (1)
        """
        return u_hat, _


class GeometricBrownianMotion(ItoProcessDriver):
    """
    A subclass of ItoProcess, MertonJumpDiffusionProcess under Q measure, mainly for testing.
    This class implements a multivariate geometric Brownian motion process:
    d S_t = r S_t dt + \sigma S_t dW_t

    The config provide the distribution of:
        -r_range: List[2], the uniform distribution of the risk free rate
        -s_range: List[2], the uniform distribution of the volatility rate
        -rho_range: List[2], the uniform distribution of the correlation between assets, None when dim=1.
    the parameter is a batch of parameter sampled from a distribution
    whose shape is (num_batch, dim_param = 2) \mu and \sigma
    one parameter correspond to num_sample paths
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config.eqn_config
        self.initial_mode = self.config.initial_mode
        self.x_init = self.config.x_init
        self.r_range = self.config.r_range
        self.s_range = self.config.s_range
        self.range_list = [self.r_range, self.s_range]
        self.val_range_list = [self.val_config.r_range, self.val_config.s_range]
        if self.config.dim != 1:
            self.rho_range = self.config.rho_range
            self.range_list.append(self.rho_range)
            self.val_range_list.append(self.val_config.rho_range)

    def diffusion(
        self, time: tf.Tensor, state: tf.Tensor, u_hat: tf.Tensor
    ) -> tf.Tensor:
        """
        Computes the instantaneous diffusion of this stochastic process.

        :param state : tf.Tensor
            Contains samples from the stochastic process at a specific time.
            Shape is [samples, self.dimension].
        :param time : tf.Tensor
            The current time; a scalar.
        :return diffusion: tf.Tensor
            The return is essentially a list of instantaneous diffusion matrices
            for each sampled state input.
            It is a tensor of shape [samples, self.dimension, self.dimension].

        param_input (B, 1)
        state (B, M, d)
        return (B, M, d)
        """
        batch = u_hat.shape[0]
        samples = state.shape[1]
        sigma = tf.reshape(u_hat[:, 1], [batch, 1, 1])  # [B, 1, 1]
        sigma = tf.linalg.diag(tf.tile(sigma, [1, samples, self.dim]))  # [B, M, d, d]
        return tf.einsum(
            "...ij,...j->...i", sigma, state
        )  # [B, M, d, d] x [B, M, d] -> [B, M, d]

    def corr_matrix(self, state: tf.Tensor, u_hat: tf.Tensor):
        r"""
        In this class the corr is j:
        	\begin{equation*}
		   \left(\begin{array}{cccc}
			1 & \rho & \ldots & \rho \\
			\rho & 1 & \ldots & \rho \\
			\vdots & \vdots & \ddots & \rho \\
			\rho & \rho & \rho & 1
		\end{array}\right)
	    \end{equation*}
        with rhe shape [B, M, d, d]
        """
        batch = state.shape[0]
        samples = state.shape[1]
        if not self.dim == 1:
            rho = tf.reshape(u_hat[:, 2], [batch, 1, 1, 1])  # [B, 1] -> [B, 1, 1, 1]
            rho_mat = tf.tile(rho, [1, samples, self.dim, self.dim])  # [B, M, d, d]
            i_mat = tf.eye(self.dim, batch_shape=[batch, samples])  # [B, M, d, d]
            rho_diag = tf.linalg.diag(rho_mat[..., 0])  # [B, M, d, d]
            corr = i_mat - rho_diag + rho_mat  # [B, M, d, d]
        else:
            corr = super(GeometricBrownianMotion, self).corr_matrix(
                state, u_hat
            )  # [B, M, d, d]
        return corr  # [B, M, d, d]

    def diffusion_onestep(
        self, time_tensor: tf.Tensor, state_tensor: tf.Tensor, u_hat: tf.Tensor
    ):
        """
        get \sigma(t,X) with shape (B,M,N,d)
        in GBM \sigma(t,X) = \sigma * X
        """
        vol_tensor = tf.expand_dims(u_hat[..., 1], -1)  # [B, M, N, 1]
        return vol_tensor * state_tensor  # [B, M, N, d]

    def driver_bsde(
        self, t: tf.Tensor, x: tf.Tensor, y: tf.Tensor, u_hat: tf.Tensor
    ) -> tf.Tensor:
        """
        this yields the driver term of the BSDE coupled with this SDE
        param t: [B, M, N, 1]
        param x: [B, M, N, d]
        param y: [B, M, N, 1]
        param u_hat: [B, M, N, 4]
        """
        r = u_hat[:, :, :, 0:1]  # [B, M, 1]
        return -r * y

    def euler_onestep(
        self,
        time_tensor: tf.Tensor,
        state_tensor: tf.Tensor,
        dw: tf.Tensor,
        u_hat: tf.Tensor,
    ):
        # assert self.diffusion(time, x_path, param).shape == dw.shape
        r = tf.expand_dims(u_hat[..., 0], -1)  # [B, M, N, 1]
        v = tf.expand_dims(u_hat[..., 1], -1)  # [B, M, N, 1]
        state_tensor_after_step = (
            state_tensor + r * state_tensor * self.config.dt + v * state_tensor * dw
        )
        return state_tensor_after_step  # [B, M, N, d]

    def split_uhat(self, u_hat: tf.Tensor):
        """
        GBM case, the parameters sampled has dimension batch_size + (3). where K=3
        and batch_size = (B, M, N)
        Then this function calculate the curve function based on parameter $\mu$ and $\sigma$
        return $\mu(t)$ and $\sigma(t)$ on the given support grid and return the batch of moneyness K
        Then Then return is a tuple of two tensors: (u_curve, u_param)
        u_curve: batch_size + (time_steps, num_curves), u_param = batch_size + (1)
        """
        u_curve = tf.expand_dims(u_hat[..., :2], axis=-2)
        u_curve = self.representor.get_sensor_value(u_curve)
        u_param = u_hat[..., 2:]
        return u_curve, u_param


class TimeDependentGBM(GeometricBrownianMotion):
    r"""
    This class implenent the time dependent GBM:
    d S_t = r(t) S_t dt + \sigma(t) S_t dW_t
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config.eqn_config
        self.initial_mode = self.config.initial_mode
        self.x_init = self.config.x_init
        self.r0_range = self.config.r0_range
        self.r1_range = self.config.r1_range
        self.r2_range = self.config.r2_range
        self.s0_range = self.config.s0_range
        self.beta_range = self.config.beta_range
        self.range_list = [
            self.r0_range,
            self.r1_range,
            self.r2_range,
            self.s0_range,
            self.beta_range,
        ]
        self.val_range_list = [
            self.val_config.r0_range,
            self.val_config.r1_range,
            self.val_config.r2_range,
            self.val_config.s0_range,
            self.val_config.beta_range,
        ]
        if self.config.dim != 1:
            self.rho_range = self.config.rho_range
            # if not self.config.iid:
            self.range_list.append(self.rho_range)
            self.val_range_list.append(self.val_config.rho_range)

        self.r_representor = QuadraticRepresentor(config)
        self.s_representor = ExponentialDecayRepresentor(config)

        self.t_grid = tf.linspace(0.0, self.config.T, self.config.sensors)

    def drift(self, time: tf.Tensor, state: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        """
        Computes the drift of this stochastic process.
        operator setting
        mu: (batch_size, 1)
        state: (batch_size, path_size, dim)
        mu * state on a batch
        return: batch_size, path_size, dim)
        """
        batch = tf.shape(u_hat)[0]
        r0 = tf.reshape(u_hat[:, 0], [batch, 1, 1])  # [B, 1, 1]
        r1 = tf.reshape(u_hat[:, 1], [batch, 1, 1])  # [B, 1, 1]
        r2 = tf.reshape(u_hat[:, 2], [batch, 1, 1])  # [B, 1, 1]
        r = r0 + r1 * time + r2 * time**2  # [B, M, d]
        return r * state

    def diffusion(
        self, time: tf.Tensor, state: tf.Tensor, u_hat: tf.Tensor
    ) -> tf.Tensor:
        """
        Computes the instantaneous diffusion of this stochastic process.

        :param state : tf.Tensor
            Contains samples from the stochastic process at a specific time.
            Shape is [samples, self.dimension].
        :param time : tf.Tensor
            The current time; a scalar.

        :return diffusion: tf.Tensor
            The return is essentially a list of instantaneous diffusion matrices
            for each sampled state input.
            It is a tensor of shape [samples, self.dimension, self.dimension].

        param_input (B, 1)
        state (B, M, D)
        return (B, M, D)
        """
        T = self.config.T
        batch = u_hat.shape[0]
        samples = state.shape[1]
        s_0 = tf.reshape(u_hat[:, 3], [batch, 1, 1])
        beta = tf.reshape(u_hat[:, 4], [batch, 1, 1])
        sigma = s_0 * tf.exp(beta * (time - T))
        sigma = tf.linalg.diag(tf.tile(sigma, [1, samples, self.dim]))
        return tf.einsum("...ij,...j->...i", sigma, state)  # [B, M, d]

    def drift_onestep(
        self, time_tensor: tf.Tensor, state_tensor: tf.Tensor, u_hat: tf.Tensor
    ):
        """
        get r(t,X) with shape (B,M,N,d)
        in TGBM r(t,X) = r_0 + r_1 * t + r_2 * t^2
        """
        r_0 = tf.expand_dims(u_hat[..., 0], -1)  # [B, M, N, 1]
        r_1 = tf.expand_dims(u_hat[..., 1], -1)
        r_2 = tf.expand_dims(u_hat[..., 2], -1)
        u_hat_r = tf.concat([r_0, r_1, r_2], axis=-1)
        r_t = self.r_representor.get_func_value(
            time_tensor, state_tensor, u_hat_r
        )  # [B, M, N, 1]
        return r_t  # [B, M, N, d]

    def diffusion_onestep(
        self, time_tensor: tf.Tensor, state_tensor: tf.Tensor, u_hat: tf.Tensor
    ):
        r"""
        get \sigma(t,X) with shape (B,M,N,d)
        in GBM \sigma(t,X) = \sigma_0 * \exp{-\beta * (T - t)}
        """
        T = self.config.T
        s_0 = tf.expand_dims(u_hat[..., 3], -1)  # [B, M, N, 1]
        beta = tf.expand_dims(u_hat[..., 4], -1)  # [B, M, N, 1]
        u_hat_s = tf.concat([s_0, beta], axis=-1)
        vol_tensor = self.s_representor.get_func_value(
            time_tensor, state_tensor, u_hat_s
        )  # [B, M, N, 1]
        return vol_tensor * state_tensor  # [B, M, N, d]

    def driver_bsde(
        self, t: tf.Tensor, x: tf.Tensor, y: tf.Tensor, u_hat: tf.Tensor
    ) -> tf.Tensor:
        """
        t: [B, M, N, 1]
        x: [B, M, N, d]
        y: [B, M, N, 1]
        u_hat: [B, M, N, 4]
        """
        r = self.drift_onestep(t, x, u_hat)  # (B, M, 1)
        return -r * y

    def euler_onestep(
        self,
        time_tensor: tf.Tensor,
        state_tensor: tf.Tensor,
        dw: tf.Tensor,
        u_hat: tf.Tensor,
    ):
        """
        time_tensor: [B, M, N, 1]
        state_tensor: [B, M, N, d]
        dw: [B, M, N, d]
        u_hat: [B, M, N, d]
        """
        # assert self.diffusion(time, x_path, param).shape == dw.shape
        r = self.drift_onestep(time_tensor, state_tensor, u_hat)  # [B, M, N, d]
        vs = self.diffusion_onestep(time_tensor, state_tensor, u_hat)  # [B, M, N, d]
        state_tensor_after_step = (
            state_tensor + r * state_tensor * self.config.dt + vs * dw
        )  # [B, M, N, d]
        return state_tensor_after_step

    def split_uhat(self, u_hat: tf.Tensor):
        """
        GBM case, the parameters sampled has dimension batch_size + (3). where K=3
        and batch_size = (B, M, N)
        Then this function calculate the curve function based on parameter $\mu$ and $\sigma$
        return $\mu(t)$ and $\sigma(t)$ on the given support grid and return the batch of moneyness K
        Then Then return is a tuple of two tensors: (u_curve, u_param)
        u_curve: batch_size + (time_steps, num_curves), u_param = batch_size + (1)
        """
        u_hat_r = u_hat[..., :2]
        u_hat_s = u_hat[..., 2:4]
        r_curve = self.r_representor.get_sensor_value(u_hat_r)
        s_curve = self.s_representor.get_sensor_value(u_hat_s)
        u_curve = tf.concat([r_curve, s_curve], axis=-1)
        u_param = u_hat[..., 4:]
        return u_curve, u_param


class HestonModel(ItoProcessDriver):
    r"""
    The Heston model is as follow:
     d S_t &= r S_t dt + \sqrt{v_t} S_t dW^S_t \\
     d v_t &= k(b - v_t) dt + vol * dW^v_t
    where dW^S_t * dW^v_t = \rho dt

    The config provide the distribution of:
        -kappa_range: List[2], the uniform distribution of the the mean revert rate;
        -theta_range: List[2], the uniform distribution of the the mean revert level;
        -s_range: List[2], the uniform distribution of the volatility of volatility rate;
        -rho_range: List[2], the uniform distribution of the correlation between assets and stochastic volatilities.
    """

    def __init__(self, config):
        super().__init__(config)
        self.r_range = self.config.r_range
        self.kappa_range = self.config.kappa_range
        self.theta_range = self.config.theta_range
        self.sigma_range = self.config.sigma_range
        self.rho_range = self.config.rho_range
        self.range_list = [
            self.r_range,
            self.theta_range,
            self.kappa_range,
            self.sigma_range,
            self.rho_range,
        ]
        self.val_range_list = [
            self.val_config.r_range,
            self.val_config.theta_range,
            self.val_config.kappa_range,
            self.val_config.sigma_range,
            self.val_config.rho_range,
        ]
        if self.dim > 1:
            self.rhos_range = self.config.rhos_range
            self.range_list.append(self.rhos_range)
            self.val_range_list.append(self.val_config.rhos_range)

    def initial_sampler(self, u_hat: tf.Tensor, samples: int) -> tf.Tensor:
        initial_state = super().initial_sampler(u_hat, samples)
        new_state = initial_state * 0.1
        initial_value = tf.concat([initial_state, new_state], axis=-1)
        return initial_value  # [B, M, N, 2d]

    def drift(self, time: tf.Tensor, state: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        """
        In Heston model state = (S_t, V_t) with dim 2 * d
        S = state[:dim]
        V = state[dim:]
        drift_asset = r * S_t
        drift_vol = k(b - v_t)
        """
        batch = u_hat.shape[0]
        mu = tf.reshape(u_hat[:, 0], [batch, 1, 1])
        asset_state = state[:, :, : self.dim]
        vol_state = tf.math.abs(state[:, :, self.dim :])
        asset_drift = mu * asset_state
        kappa = tf.reshape(u_hat[:, 1], [batch, 1, 1])
        theta = tf.reshape(u_hat[:, 2], [batch, 1, 1])
        vol_drift = kappa * (theta - vol_state)
        return tf.concat([asset_drift, vol_drift], axis=-1)  # [B, M, N, 2d]

    def diffusion(
        self, time: tf.Tensor, state: tf.Tensor, u_hat: tf.Tensor
    ) -> tf.Tensor:
        """
        Computes the instantaneous diffusion of this stochastic process.
        param_input (B, 1)
        state (B, M, D + D)
        return (B, M, D + D)
        diff_asset = sqrt(v_t) * S_t
        diff_vol = vol_of_vol * sqrt(v_t)
        """
        batch = u_hat.shape[0]
        asset_state = state[:, :, : self.dim]  # [B, M, d]
        vol_state = tf.math.abs(state[:, :, self.dim :])  # [B, M, d]
        sqrt_vol = tf.math.sqrt(vol_state)  # [B, M, d]
        asset_diffusion = sqrt_vol * asset_state  # [B, M, d]
        vol_of_vol = tf.reshape(u_hat[:, 3], [batch, 1, 1])  # [B, 1, 1]
        vol_diffusion = vol_of_vol * sqrt_vol  # [B, 1, 1] * [B, M, d] -> [B, M, d]
        return tf.concat([asset_diffusion, vol_diffusion], axis=-1)  # [B, M, 2d]

    def driver_bsde(
        self, t: tf.Tensor, x: tf.Tensor, y: tf.Tensor, u_hat: tf.Tensor
    ) -> tf.Tensor:
        """
        this yields the driver term of the BSDE coupled with this SDE
        param t: [B, M, N, 1]
        param x: [B, M, N, 2d]
        param y: [B, M, N, 1]
        param u_hat: [B, M, N, 6]
        """
        r = u_hat[:, :, :, 0:1]  # [B, M, N, 1]
        return -r * y

    def corr_matrix_1d(self, state: tf.Tensor, u_hat: tf.Tensor):
        r"""
        When it is 1d case, the matrix is simply [[1, \rho], [\rho, 1]]
        \rho is the correlation coefficient between asset and volatility.
        return: float tensor with shape [B, M, 2, 2]
        """
        batch = state.shape[0]
        samples = state.shape[1]
        actual_dim = state.shape[2]
        rho = tf.reshape(u_hat[:, 4], [batch, 1, 1, 1])  # [B, 1, 1, 1]
        rho_mat = tf.tile(rho, [1, samples, actual_dim, actual_dim])  # [B, M, 2, 2]
        i_mat = tf.eye(actual_dim, batch_shape=[batch, samples])  # [B, M, 2, 2]
        rho_diag = tf.linalg.diag(rho_mat[..., 0])
        corr = i_mat - rho_diag + rho_mat  # [B, M, 2, 2]
        return corr

    def cholesky_matrix_nd(self, state: tf.Tensor, u_hat: tf.Tensor):
        r"""
        For nd Heston model, we just calculate the matrix which is the cholesky decomposition of
        \begin{equation*}
		\Sigma=\left(\begin{array}{cc}
			\Sigma_S & \Sigma_{S V} \\
			\Sigma_{S V}^{\top} & I_d
		\end{array}\right)
	    \end{equation*}
        Then the matrix has three parts:
        1.The left top part is the Cholesky of the submatrix which is corresponded to the asset prices;
        2.The right top part is the Cholesky of the submatrix which is corresponded to each asset and its volatility;
        3.The right bottom part is the diagnal matrix whic is the vol of vol of each stochastic volatility.
        return: the float tensor with shape [B, M, d, d]
        """
        batch = state.shape[0]
        samples = state.shape[1]
        rho_s = tf.reshape(u_hat[:, 5], [batch, 1, 1, 1])  # [B, 1, 1, 1]
        rho_s_mat = tf.tile(rho_s, [1, samples, self.dim, self.dim])  # [B, M, d, d]
        zeros_mat = tf.zeros([batch, samples, self.dim, self.dim])  # [B, M, d, d]
        i_mat = tf.eye(self.dim, batch_shape=[batch, samples])  # [B, M, d, d]
        rho_s_diag = tf.linalg.diag(rho_s_mat[..., 0])  # [B, M, d, d]
        corr_s = i_mat - rho_s_diag + rho_s_mat  # [B, M, d, d]
        cholesky_s = tf.linalg.cholesky(corr_s)  # [B, M, d, d]
        rho_sv = tf.reshape(u_hat[:, 4], [batch, 1, 1, 1])  # [B, 1, 1, 1]
        rho_sv_mat = tf.tile(rho_sv, [1, samples, self.dim, self.dim])  # [B, M, d, d]
        rho_sv_diag = tf.linalg.diag(rho_sv_mat[..., 0])
        a = tf.concat([cholesky_s, rho_sv_diag], axis=3)
        b = tf.concat([zeros_mat, i_mat], axis=3)
        return tf.concat([a, b], axis=2)  # [B, M, 2d, 2d]

    def brownian_motion(self, state: tf.Tensor, u_hat: tf.Tensor):
        dt = self.config.dt
        batch_size = tf.shape(u_hat)[0]
        samples = tf.shape(state)[1]
        actual_dim = tf.shape(state)[2]  # in SV model actual_dim = 2* dim =/= dim
        if self.dim == 1:
            corr = self.corr_matrix_1d(state, u_hat)
            cholesky_matrix = tf.linalg.cholesky(corr)
        else:
            cholesky_matrix = self.cholesky_matrix_nd(state, u_hat)
        white_noise = tf.random.normal(
            [batch_size, samples, actual_dim], mean=0.0, stddev=tf.sqrt(dt)
        )
        state_noise = tf.einsum("...ij,...j->...i", cholesky_matrix, white_noise)
        return state_noise  # [B, M, 2d]

    def drift_onestep(
        self, time_tensor: tf.Tensor, state_tensor: tf.Tensor, u_hat: tf.Tensor
    ):
        """
        all inputs are [batch, sample, T, dim * 2] like
        what we do is calculate the drift for the whole tensor
        output: (r * S_t , k(b - v_t)) for all t with shape [batch, sample, T, dim * 2]
        """
        assert state_tensor.shape[0] == u_hat.shape[0]
        rate_tensor = tf.expand_dims(u_hat[..., 0], -1)
        kappa_tensor = tf.expand_dims(u_hat[..., 1], -1)
        theta_tensor = tf.expand_dims(u_hat[..., 2], -1)
        s_tensor = state_tensor[..., : self.dim]
        v_tensor = state_tensor[..., self.dim :]
        drift_s = rate_tensor * s_tensor  # [B, M, d]
        drift_v = kappa_tensor * (theta_tensor - v_tensor)  # [B, M, d]
        return tf.concat([drift_s, drift_v], axis=-1)  # [B, M, 2d]

    def diffusion_onestep(
        self, time_tensor: tf.Tensor, state_tensor: tf.Tensor, u_hat: tf.Tensor
    ):
        """
        get \sigma(t,X) with shape (B,M,N,d)
        in Heston \sigma(t,X) = (\sqrt(V_t) * X_t, vol_of_vol * \sqrt(V_t))
        """
        assert state_tensor.shape[0] == u_hat.shape[0]
        vol_of_vol = tf.expand_dims(u_hat[..., 3], -1)
        s_tensor = state_tensor[..., : self.dim]
        v_tensor = state_tensor[..., self.dim :]
        sqrt_v = tf.math.sqrt(tf.math.abs(v_tensor))
        diff_s = sqrt_v * s_tensor  # [B, M, d]
        diff_v = vol_of_vol * sqrt_v  # [B, M, d]
        return tf.concat([diff_s, diff_v], axis=-1)  # [B, M, 2d]

    def split_uhat(self, u_hat: tf.Tensor):
        r"""
        Heston case, the parameters sampled has dimension batch_size + (5). where K=5
        and batch_size = (B, M, N)
        Then this function calculate the curve function based on parameter $r$ and $\theta$
        return $r(t)$ and $\theta(t)$ on the given support grid and return the batch of moneyness (\kappa, \sigma, K)
        Then Then return is a tuple of two tensors: (u_curve, u_param)
        u_curve: batch_size + (time_steps, num_curves), u_param = batch_size + (3)
        """
        # B_0 = tf.shape(u_hat)[0]
        # B_1 = tf.shape(u_hat)[1]
        # B_2 = tf.shape(u_hat)[2]
        # u_curve = tf.reshape(u_hat[..., :2], [B_0, B_1, B_2, 1, 2])
        # u_curve = tf.tile(u_curve, [1, 1, 1, self.config.sensors, 1])

        u_curve = tf.expand_dims(u_hat[..., :2], axis=-2)
        u_curve = self.representor.get_sensor_value(u_curve)
        u_param = u_hat[..., 2:]
        return u_curve, u_param


class HullWhiteModel(ItoProcessDriver):
    r"""
    This class implement the one factor Hull-White model. The model is:
    d r_t = k(\theta(t) - r_t) dt + \sigma dW_t
    where \theta_t is consistent with the initial zero coupon curve.
    In this class, we simplify the model to the flatten initial forward rate curve and then we have:
    f(0, t) = r_0 = \theta(t) which is a constant.
        -kappa_range: List[2], the uniform distribution of the the mean revert rate;
        -theta_range: List[2], the uniform distribution of the the long term interest rate level (flat curve);
        -s_range: List[2], the uniform distribution of the volatility rate;
    """

    def __init__(self, config):
        super().__init__(config)
        self.kappa_range = self.config.kappa_range
        self.theta_range = self.config.theta_range
        self.sigma_range = self.config.sigma_range
        self.range_list = [self.kappa_range, self.theta_range,self.sigma_range]
        self.val_range_list = [
            self.val_config.kappa_range,
            self.val_config.theta_range,
            self.val_config.sigma_range,
        ]
        self.epsilon = 0.01

    def drift(self, time: tf.Tensor, state: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        r"""
        In Hull White state = r_t with dim d (d=1 usually)
        drift: \kappa (\theta - r_t)
        """
        batch = u_hat.shape[0]
        kappa = tf.reshape(u_hat[:, 0], [batch, 1, 1])  # [B, 1, 1]
        theta = tf.reshape(u_hat[:, 1], [batch, 1, 1])  # [B, 1, 1]
        drift = kappa * (theta - state)  # [B, M, 1]
        return drift

    def diffusion(
        self, time: tf.Tensor, state: tf.Tensor, u_hat: tf.Tensor
    ) -> tf.Tensor:
        """
        Computes the instantaneous diffusion of this stochastic process.
        param_input (B, 1)
        return sigma
        """
        batch = u_hat.shape[0]
        vol_of_vol = tf.reshape(u_hat[:, 2], [batch, 1, 1])
        return vol_of_vol

    def euler_maryama_step(self, time: tf.Tensor, state: tf.Tensor, u_hat: tf.Tensor):
        return super().euler_maryama_step(time, state, u_hat)

    def drift_onestep(
        self, time_tensor: tf.Tensor, state_tensor: tf.Tensor, u_hat: tf.Tensor
    ) -> tf.Tensor:
        """
        all inputs are [batch, sample, T, dim * 2] like
        what we do is calculate the drift for the whole tensor
        output: k(b - r_t) for all t with shape [batch, sample, T, dim] with dim=1 usually
        """
        assert state_tensor.shape[0] == u_hat.shape[0]
        kappa_tensor = tf.expand_dims(u_hat[..., 0], -1)  # [B, M, N, 1]
        theta_tensor = tf.expand_dims(u_hat[..., 1], -1)  # [B, M, N, 1]
        return kappa_tensor * (theta_tensor - state_tensor)  # [B, M, N, 1]

    def diffusion_onestep(
        self, time_tensor: tf.Tensor, state_tensor: tf.Tensor, u_hat: tf.Tensor
    ) -> tf.Tensor:
        """
        get \sigma(t,X) with shape (B,M,N,d)
        in Heston \sigma(t, X) = (\sqrt(V_t) * X_t, vol_of_vol * \sqrt(V_t))
        """
        assert state_tensor.shape[0] == u_hat.shape[0]
        vol_of_vol = tf.expand_dims(u_hat[..., 2], -1)  # [B, M, N, 1]
        return vol_of_vol * tf.ones_like(state_tensor)  # [B, M, N, 1]

    def driver_bsde(
        self, t: tf.Tensor, x: tf.Tensor, y: tf.Tensor, u_hat: tf.Tensor
    ) -> tf.Tensor:
        """
        t: [B, M, N, 1]
        x: [B, M, N, d] d=1 in HW case
        y: [B, M, N, 1]
        u_hat: [B, M, N, 4]
        """
        r = tf.reduce_mean(x, axis=-1, keepdims=True)  # [B, M, N, 1]
        return -r * y  # [B, M, N, 1]

    def split_uhat(self, u_hat: tf.Tensor):
        """
        GBM case, the parameters sampled has dimension batch_size + (3). where K=3
        and batch_size = (B, M, N)
        Then this function calculate the curve function based on parameter $\mu$ and $\sigma$
        return $\mu(t)$ and $\sigma(t)$ on the given support grid and return the batch of moneyness K
        Then Then return is a tuple of two tensors: (u_curve, u_param)
        u_curve: batch_size + (time_steps, num_curves), u_param = batch_size + (1)
        """
        u_curve = tf.expand_dims(u_hat[..., :3], axis=-2)
        u_curve = self.representor.get_sensor_value(u_curve)
        u_param = u_hat[..., 3:]
        return u_curve, u_param

    def zcp_value(
        self,
        time: tf.Tensor,
        state: tf.Tensor,
        u_hat: tf.Tensor,
        terminal_date: tf.Tensor,
    ) -> tf.Tensor:
        r"""
        evaluate the zero coupon bond value at time t with maturity T and state r_t.
        It is written as the following form:
         P(t, T) = A(t, T) * \exp{-B(t, T) * r_{t}} where function A and B are functions of T-t
         with \kappa, \theta, \sigma as parameters.
        time: tensor of t: [B, M, 1]
        state: tensor of r_t [B, M, d] d=1 usually
        u_hat: tensor of kappa, theta, sigma [B, M, 4]
        terminal_date: scalar tensor of T
        return [B, M ,1]
        """
        kappa = tf.expand_dims(u_hat[..., 0], axis=-1)
        theta = tf.expand_dims(u_hat[..., 1], axis=-1)
        sigma = tf.expand_dims(u_hat[..., 2], axis=-1)
        B = (1 - tf.exp(-kappa * (terminal_date - time))) / kappa
        A = tf.exp(
            (B - terminal_date + time)
            * (kappa**2 * theta - sigma**2 / 2)
            / kappa**2
            + (sigma * B) ** 2 / (4 * kappa)
        )
        p = A * tf.exp(-B * tf.reduce_sum(state, axis=-1, keepdims=True))
        return tf.where(time > terminal_date + self.epsilon, 0.0, p)  # [B, M, 1]
