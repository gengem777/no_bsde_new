from sde import ItoProcess
import tensorflow as tf
import tensorflow_probability as tfp
from function_representors import ConstantRepresentor


class ItoJumpProcess(ItoProcess):
    r"""
    Abstract class for Ito processes, themselves driven by another Ito process B(t).
    This class represents the solution to an Ito stochastic differential equation with jump diffusion of the form
    dS_t = r * S_t dt + \sigma(S_t, t) S_t dW(t) + \sum_t^{t + dt} (e^z - 1) S_t
    We inherite following methods in the ItoProcess
        1. do initialization of X_0
        2. one step forward simulation
    But we change the base class methods as follows:
        1. Euler step: a. we add jump diffusion
                       b. sampling jump size (with different distributions in different classes)
                       c. sampling intensities under different parameterization
        2. simulate the path and output path tensor with Brownian motion increment tensor and jump increment
    The class ItoProcessDriven implements common methods for sampling from the solution to this SDE,
    """

    def __init__(self, config):
        super(ItoJumpProcess, self).__init__(config)
        self.range_list = []
        self.jump_sizes = self.config.jump_size_list
        self.jump_probs = self.config.probs_list

    def jump_happen(
        self, state: tf.Tensor, intensity: tf.Tensor, samples: int
    ) -> tf.Tensor:
        """
        This give a {0, 1} variable to tell us whether the certain time point has the jump.
        state: [B, M, d]
        intensity: [B, ] which means a batch of intensities
        return: [B, M, d] -> {0, 1} variable to indicate the happening of jump event
        """
        actual_dim = tf.shape(state)[-1]
        dist = tfp.distributions.Bernoulli(probs=intensity * self.config.dt)
        happen = tf.transpose(dist.sample([samples, actual_dim]), perm=[2, 0, 1])
        return tf.cast(happen, tf.float32)

    def get_intensity(self, u_hat: tf.Tensor) -> tf.Tensor:
        """
        from [B, k] shape parameters to get the intensity [B, ]
        return [B, ]
        """
        raise NotImplementedError

    def jump_size(self, state: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        """
        This give the relative jump size z where z is the Bernoulli distribution in the case in base class
        This distribution is a discretized distribution:
        values: list
        probs: list
        state: [B, M, d]
        return: [B, M, d]
        """
        dist = tfp.distributions.FiniteDiscrete(self.jump_sizes, probs=self.jump_probs)
        z = dist.sample(tf.shape(state))
        return z

    def compensate_expectation(self, state: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        r"""
        Since In jump diffusion, we need compensate jump so that the process can be a martingale
        Then we calculate it here, which is \int \Gamma(S_t, z) v(dz), where v(dz) = \lambda * f(dz),
        where f(dz) is the law of the jump size.
        """
        expectation = sum(
            (tf.exp(z) - 1.0) * p for (z, p) in zip(self.jump_sizes, self.jump_probs)
        )
        intensity = tf.reshape(self.get_intensity(u_hat), [-1, 1, 1])  # [B, 1]
        samples = tf.shape(state)[1]  # M
        intensity = tf.tile(intensity, [1, samples, 1])  # [B, M, 1]
        compensate_exp = expectation * intensity
        return compensate_exp * state  # l * E[e^z - 1] * S

    def pure_jump(self, state: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        """
        happen * (e^z - 1)
        """
        intensity = self.get_intensity(u_hat)  # [B,]
        samples = tf.shape(state)[1]  # M
        happen = self.jump_happen(state, intensity, samples)  # [B, M, 1]
        z = self.jump_size(state, u_hat)  # [B, M, 1]
        pure_jump_noise = happen * (tf.exp(z) - 1.0)  # [B, M, 1]
        return pure_jump_noise * state  # Y * (e^z - 1) * S

    def compensate_jump(self, state: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        jump = self.pure_jump(state, u_hat)
        expect = self.compensate_expectation(state, u_hat) * self.config.dt
        return jump - expect

    def euler_maryama_step(self, time: tf.Tensor, state: tf.Tensor, u_hat: tf.Tensor):
        """
        We extraly calculate the jump term happen * (e^z - 1) * S_t
        return the tuple: S_{t + dt}: [B, M, d], dW_{t+dt} - dW_t: [B, M, d], happen * (e^z - 1) = jump_noise: [B, M, d]
        """
        newstate_without_jump, noise = super().euler_maryama_step(time, state, u_hat)
        intensity = self.get_intensity(u_hat)
        samples = tf.shape(state)[1]
        happen = self.jump_happen(state, intensity, samples)
        z = self.jump_size(state, u_hat)
        jump_increment = self.compensate_jump(state, u_hat)
        new_state = newstate_without_jump + jump_increment
        return new_state, noise, happen, z

    def sde_simulation(self, u_hat: tf.Tensor, samples: int):
        """
        the whole simulation process with each step iterated by method euler_maryama_step(),
        return is a tuple of tensors recording the paths of state and BM increments
        x: path of states, shape: (batch_size, path_size, num_timesteps, dim)
        dw: path of BM increments, shape: (batch_size, path_size, num_timesteps-1, dim)
        d_jump: path pf jump increments, shape: (batch_size, path_size, num_timesteps-1, dim)
        """
        tf.random.set_seed(0)
        stepsize = self.config.dt
        time_steps = self.config.time_steps
        state_process = tf.TensorArray(tf.float32, size=time_steps + 1)
        brownian_increments = tf.TensorArray(tf.float32, size=time_steps)
        jump_sizes = tf.TensorArray(tf.float32, size=time_steps)
        happen = tf.TensorArray(tf.float32, size=time_steps)
        state = self.initial_sampler(u_hat, samples)

        time = tf.constant(0.0)
        state_process = state_process.write(0, state)
        current_index = 1
        while current_index <= time_steps:
            state, dw, h, z = self.euler_maryama_step(
                time, state, u_hat
            )  # [B, M, d], [B, M, d], [B, M, d], [B, M, d]
            state_process = state_process.write(current_index, state)
            brownian_increments = brownian_increments.write(current_index - 1, dw)
            happen = happen.write(current_index - 1, h)
            jump_sizes = jump_sizes.write(current_index - 1, z)
            current_index += 1
            time += stepsize
        x = state_process.stack()
        dw = brownian_increments.stack()
        h = happen.stack()
        z = jump_sizes.stack()
        x = tf.transpose(x, perm=[1, 2, 0, 3])
        dw = tf.transpose(dw, perm=[1, 2, 0, 3])
        h = tf.transpose(h, perm=[1, 2, 0, 3])
        z = tf.transpose(z, perm=[1, 2, 0, 3])
        return (
            x,
            dw,
            h,
            z,
        )  # [B, M, N, d], [B, M, N-1, d], [B, M, N-1, d], [B, M, N-1, d]


class GBMwithSimpleJump(ItoJumpProcess):
    """
    A subclass of ItoProcess, MertonJumpDiffusionProcess under Q measure, mainly for testing.
    This class implements a multivariate geometric Brownian motion process:
    d S_t = r S_t dt + \sigma S_t dW_t

    The config provide the distribution of:
        -r_range: List[2], the uniform distribution of the risk free rate
        -s_range: List[2], the uniform distribution of the volatility rate
        -lambda_range: List[2], the uniform distribution of the intensity rate
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
        self.lambda_range = self.config.lambda_range
        self.range_list = [self.r_range, self.s_range, self.lambda_range]
        self.val_range_list = [
            self.val_config.r_range,
            self.val_config.s_range,
            self.config.lambda_range,
        ]
        self.representor = ConstantRepresentor(config)

    def get_intensity(self, u_hat: tf.Tensor) -> tf.Tensor:
        """
        from [B, k] shape parameters to get the intensity [B, ]
        return [B, ]
        """
        return u_hat[:, 2]

    def jump_size(self, state: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        """
        This give the relative jump size z where z is the Bernoulli distribution in the case in base class
        """
        dist = tfp.distributions.FiniteDiscrete(self.jump_sizes, probs=self.jump_probs)
        z = dist.sample(tf.shape(state))
        return z

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
            corr = super(GBMwithSimpleJump, self).corr_matrix(
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
        d_jump: tf.Tensor,
        u_hat: tf.Tensor,
    ):
        r = tf.expand_dims(u_hat[..., 0], -1)  # [B, M, N, 1]
        v = tf.expand_dims(u_hat[..., 1], -1)  # [B, M, N, 1]
        state_tensor_after_step = (
            state_tensor
            + r * state_tensor * self.config.dt
            + v * state_tensor * dw
            + d_jump * state_tensor
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
        u_curve = tf.expand_dims(u_hat[..., :3], axis=-2)
        u_curve = self.representor.get_sensor_value(u_curve)
        u_param = u_hat[..., 3:]
        return u_curve, u_param

class StochasticVolJumpModel(ItoJumpProcess):
    pass
