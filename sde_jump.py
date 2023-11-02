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
        super(ItoJumpProcess,self).__init__(config)
        self.range_list = []
    
    def jump_happen(self, state: tf.Tensor, intensity: tf.Tensor, samples: int) -> tf.Tensor:
        """
        This give a {0, 1} variable to tell us whether the certain time point has the jump.
        state: [B, M, d]
        intensity: [B, ] which means a batch of intensities
        return: [B, M, d] -> {0, 1} variable to indicate the happening of jump event
        """
        actual_dim = tf.shape(state)[-1]
        dist = tfp.distributions.Bernoulli(probs = intensity * self.config.dt)
        happen = tf.transpose(dist.sample([samples, actual_dim]), perm = [2, 0, 1])
        return happen
    
    def get_intensity(self, u_hat: tf.Tensor) -> tf.Tensor:
        """
        from [B, k] shape parameters to get the intensity [B, ]
        return [B, ]
        """
        raise NotImplementedError


    def jump_size(self, state: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        """
        This give the relative jump size z where z is the Bernoulli distribution in the case in base class
        """
        raise NotImplementedError
    
    def euler_maryama_step(self, time: tf.Tensor, state: tf.Tensor, u_hat: tf.Tensor):
        """
        We extraly calculate the jump term happen * (e^z - 1) * S_t
        return the tuple: S_{t + dt}: [B, M, d], dW_{t+dt} - dW_t: [B, M, d], happen * (e^z - 1) = jump_noise: [B, M, d]
        """
        newstate_without_jump, noise =  super().euler_maryama_step(time, state, u_hat)
        intensity = self.get_intensity(u_hat)
        samples = tf.shape(state)[1]
        happen = self.jump_happen(state, intensity, samples)
        z = self.jump_size(state, u_hat)
        jump_noise = happen * (tf.exp(z) - 1.0)
        jump_increment = jump_noise * state
        new_state = newstate_without_jump + jump_increment
        return new_state, noise, jump_noise
    
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
        jump_increments = tf.TensorArray(tf.float32, size=time_steps)
        state = self.initial_sampler(u_hat, samples)

        time = tf.constant(0.0)
        state_process = state_process.write(0, state)
        current_index = 1
        while current_index <= time_steps:
            state, dw, d_jump = self.euler_maryama_step(
                time, state, u_hat
            )  # [B, M, d], [B, M, d], [B, M, d]
            state_process = state_process.write(current_index, state)
            brownian_increments = brownian_increments.write(current_index - 1, dw)
            jump_increments = jump_increments.write(current_index - 1, d_jump)
            current_index += 1
            time += stepsize
        x = state_process.stack()
        dw = brownian_increments.stack()
        d_jump = jump_increments.stack()
        x = tf.transpose(x, perm=[1, 2, 0, 3])
        dw = tf.transpose(dw, perm=[1, 2, 0, 3])
        d_jump = tf.transpose(d_jump, perm=[1, 2, 0, 3])
        return x, dw, d_jump  # [B, M, N, d], [B, M, N-1, d], [B, M, N-1, d]
    

