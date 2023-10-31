import tensorflow as tf

class BaseRepresentor:
    """
    This class is to dealing with different kinds of functions, for each child class,
    we fix a parametric form such as exponential family or polynomial family. Then we give to methods:
    1. method for sampling related parameter of the function on both 3 or 4 dim tensors.
    2. method for reference the function value on certain input variables and parameters.
    3. method for inference some useful value which is an abstract tensor for deepOnet input. 
        1. inference values on a set of grids;
        2. calculate integral transformation (kernel operation) the sample on a new grid.

    the hyperparameter we need to use in config is:
    -sensors: int, input size of the deepOnet
    """
    def __init__(self, config):
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.sensors = self.eqn_config.sensors 

    def sampling_parameters(self, hyper_cube: dict)-> tf.Tensor:
        raise NotImplementedError
    
    def get_func_value(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError
    
    def get_sensor_value(self, u_hat: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError
    

class ConstantRepresentor(BaseRepresentor):
    r"""
    This class give the function as the flat function with form:
      f(t,x) = c
    """
    def __init__(self, config):
        super(ConstantRepresentor, self).__init__(config)
        self.t_grid = tf.linspace(0.0, self.eqn_config.T, self.sensors)

    def get_sensor_value(self, u_hat: tf.Tensor) -> tf.Tensor:
        """
        u_hat: [B, M, N, k]
        return: [B, M, N, k * sensors]
        """
        u_curve = tf.tile(u_hat, [1, 1, 1, self.sensors, 1])
        return u_curve
    
    
class QuadraticRepresentor(BaseRepresentor):
    r"""
    This class give the function as the quadratic function with form:
      r(t) = r_0 + r_1 * t + r_2 * t ** 2
    """
    def __init__(self, config):
        super(QuadraticRepresentor, self).__init__(config)
        self.t_grid = tf.linspace(0.0, self.eqn_config.T, self.sensors)
    
    def get_func_value(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        r"""
        This method just do this thing: calculate the output tensor which calculated by:
        r = r0 * t + r1 * t + r2 * t ** 2
        The input tensor can be these two cases:
        t: [B, M, 1] or [B, M, N, 1]
        x: [B, M, d] or [B, M, N, d]
        u_hat: [B, M, k] or [B, M, N, k] 
        """
        # batch = tf.shape(x)[0] # t, x is [B, M, 1] [B, M, d] or [B, M, N, 1] [B, M, N, d] 
        r0 = tf.expand_dims(u_hat[..., 0], axis=-1) # [B, M, 1] or [B, M, N, 1]
        r1 = tf.expand_dims(u_hat[..., 1], axis=-1) # [B, M, 1] or [B, M, N, 1]
        r2 = tf.expand_dims(u_hat[..., 2], axis=-1) # [B, M, 1] or [B, M, N, 1]
        r = r0 + r1 * t + r2 * t ** 2 # [B, M, 1] or [B, M, N, 1]
        return r
    
    def get_sensor_value(self, u_hat: tf.Tensor) -> tf.Tensor:
        """
        this u_hat give the tensor corresponded to [r_0, r_1, r_2] 
        then the actual dim is [B, M, N, 3]
        """
        t = tf.reshape(self.t_grid, [1, 1, 1, self.sensors, 1])
        B_0 = tf.shape(u_hat)[0]
        B_1 = tf.shape(u_hat)[1]
        B_2 = tf.shape(u_hat)[2]
        t = tf.tile(t, [B_0, B_1, B_2, 1, 1])
        r0 = tf.reshape(u_hat[..., 0], [B_0, B_1, B_2, 1, 1]) # [B, M, N, 1, 1]
        r0 = tf.tile(r0, [1, 1, 1, self.sensors, 1]) # [B, M, N, sensors, 1]
        r1 = tf.reshape(u_hat[..., 1], [B_0, B_1, B_2, 1, 1]) # [B, M, N, 1, 1]
        r1 = tf.tile(r1, [1, 1, 1, self.sensors, 1]) # [B, M, N, sensors, 1]
        r2 = tf.reshape(u_hat[..., 2], [B_0, B_1, B_2, 1, 1]) # [B, M, N, 1, 1]
        r2 = tf.tile(r2, [1, 1, 1, self.sensors, 1]) # [B, M, N, sensors, 1]
        r_curve = r0 + r1 * t + r2 * t**2 # [B, M, N, sensors, 1]
        return r_curve # [B, M, N, sensors, 1]
    

class ExponentialDecayRepresentor(BaseRepresentor):
    r"""
    This class give the function as the quadratic function with form:
      s(t) = s0 * exp(-\beta (T - t))
    """
    def __init__(self, config):
        super(ExponentialDecayRepresentor, self).__init__(config)
        self.t_grid = tf.linspace(0.0, self.eqn_config.T, self.sensors)
    
    def get_func_value(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        r"""
        This method just do this thing: calculate the output tensor which calculated by:
        s(t) = s0 * exp(-\beta (T - t))
        The input tensor can be these two cases:
        t: [B, M, 1] or [B, M, N, 1]
        x: [B, M, d] or [B, M, N, d]
        u_hat: [B, M, k] or [B, M, N, k] 
        """
        # batch = tf.shape(x)[0] # t, x is [B, M, 1] [B, M, d] or [B, M, N, 1] [B, M, N, d] 
        s_0 = tf.expand_dims(u_hat[..., 0], axis=-1) # [B, M, 1] or [B, M, N, 1]
        beta = tf.expand_dims(u_hat[..., 1], axis=-1) # [B, M, 1] or [B, M, N, 1]
        T = self.eqn_config.T
        s = s_0 * tf.exp(-beta * (T - t)) # [B, M, 1] or [B, M, N, 1]
        return s
    
    def get_sensor_value(self, u_hat: tf.Tensor) -> tf.Tensor:
        """
        this u_hat give the tensor corresponded to [s_0, beta] 
        then the actual dim is [B, M, N, 2]
        """
        t = tf.reshape(self.t_grid, [1, 1, 1, self.sensors, 1])
        B_0 = tf.shape(u_hat)[0]
        B_1 = tf.shape(u_hat)[1]
        B_2 = tf.shape(u_hat)[2]
        T = self.eqn_config.T
        s0 = tf.reshape(u_hat[..., 0], [B_0, B_1, B_2, 1, 1]) # [B, M, N, 1, 1]
        s0 = tf.tile(s0, [1, 1, 1, self.sensors, 1]) # [B, M, N, sensors, 1]
        beta = tf.reshape(u_hat[..., 1], [B_0, B_1, B_2, 1, 1]) # [B, M, N, 1, 1]
        beta = tf.tile(beta, [1, 1, 1, self.sensors, 1]) # [B, M, N, sensors, 1]
        s_curve = s0 * tf.exp(beta * (t - T))
        return s_curve # [B, M, N, 1, 1]
