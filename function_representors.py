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
        u_hat: [B, 1, k] or [B, M, N, k] 
        """
        # batch = tf.shape(x)[0] # t, x is [B, M, 1] [B, M, d] or [B, M, N, 1] [B, M, N, d] 
        r0 = tf.expand_dims(u_hat[..., 0], axis=-1) # [B, M, 1] or [B, M, N, 1]
        r1 = tf.expand_dims(u_hat[..., 1], axis=-1) # [B, M, 1] or [B, M, N, 1]
        r2 = tf.expand_dims(u_hat[..., 2], axis=-1) # [B, M, 1] or [B, M, N, 1]
        r = r0 + r1 * t + r2 * t ** 2 # [B, M, 1] or [B, M, N, 1]
        return r
    
    def get_sensor_value(self, u_hat: tf.Tensor) -> tf.Tensor:
        t = tf.reshape(self.t_grid, [1, 1, 1, self.sensors, 1])
        B_0 = tf.shape(u_hat)[0]
        B_1 = tf.shape(u_hat)[1]
        B_2 = tf.shape(u_hat)[2]
        t = tf.tile(t, [B_0, B_1, B_2, 1, 1])
        r0 = tf.reshape(u_hat[..., 0], [B_0, B_1, B_2, 1, 1])
        r0 = tf.tile(r0, [1, 1, 1, self.config.sensors, 1])
        r1 = tf.reshape(u_hat[..., 1], [B_0, B_1, B_2, 1, 1])
        r1 = tf.tile(r1, [1, 1, 1, self.config.sensors, 1])
        r2 = tf.reshape(u_hat[..., 2], [B_0, B_1, B_2, 1, 1])
        r2 = tf.tile(r2, [1, 1, 1, self.config.sensors, 1])
        r_curve = r0 + r1 * t * r2 * t**2
        return r_curve

class ConstantRepresentor(BaseRepresentor):
    pass

class ExponentialDecayRepresentor(BaseRepresentor):
    pass
