import tensorflow as tf
from dataclasses import dataclass
import tensorflow_probability as tfp
from sde import HullWhiteModel

tfd = tfp.distributions
dist = tfd.Normal(loc=0., scale=1.)


@dataclass
class BaseOption:
    def __init__(self, config):
        self.config = config.eqn_config
        self.val_config = config.val_config
        self.strike_range = self.config.strike_range
        # self.exer_dates = []
        self.exer_dates = [0] + self.config.exercise_date
        self.steps = self.config.time_steps
        self.time_steps_one_year = self.config.time_steps_one_year
        self.exer_index = [int(num) for num in [k * self.time_steps_one_year for k in self.exer_dates]]

    def payoff(self, x: tf.Tensor, u_hat: tf.Tensor, **kwargs):
        raise NotImplementedError

    def exact_price(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor):
        raise NotImplementedError
    
    def expand_batch_inputs_dim(self, par: tf.Tensor):
        """
        input: the parmeter tensor with shape [batch_size, 1]
        output: the expanded parameter [batch_size, sample_size, time_steps, 1]
        """
        par = tf.reshape(par, [self.config.batch_size, 1, 1, 1])
        par = tf.tile(par, [1, self.config.sample_size, self.config.time_steps + 1, 1])
        return par
    
    def sample_parameters(self, N=100, training=True):  # N is the time of batch size
        if training:
            strike_range = self.strike_range
            num_params = int(N * self.config.batch_size)
            m = tf.math.maximum(tf.random.normal([num_params, 1], strike_range[0], strike_range[1]), strike_range[0] * 0.0001)
            return m
        else:
            strike_range = self.val_config.strike_range
            num_params = int(N * self.val_config.batch_size)
            m = tf.math.maximum(tf.random.normal([num_params, 1], strike_range[0], strike_range[1]), strike_range[0] * 0.0001)
            return m



class EuropeanOption(BaseOption):
    def __init__(self, config):
        super(EuropeanOption, self).__init__(config)
        """
        Parameters
        ----------
        m: moneyness with distribution N(1, 0.2)
        """
        self.strike_range = self.config.strike_range
        self.style = self.config.style
        # self.m_range = [100, 5]

    def payoff(self, x: tf.Tensor, param: tf.Tensor, **kwargs):
        """
        Parameters
        ----------
        x: tf.Tensor
            Asset price path. Tensor of shape (batch_size, sample_size, time_steps, d)
        param: tf.Tensor
            The parameter vectors of brunch net input and the last entry is for strike.
        Returns
        -------
        payoff: tf.Tensor
            basket option payoff. Tensor of shape (batch_size, sample_size, 1)
            when self.config.dim = 1, this reduced to 1-d vanilla call payoff
        """
        k = tf.expand_dims(param[:, :, 0, -1], axis=-1) # (B, M, 1)
        temp = tf.reduce_mean(x[:, :, -1, :], axis=-1, keepdims=True) # (B, M, 1)
        K = k * self.config.x_init
        if self.style == "call":
            return tf.nn.relu(temp - K)
        else:
            return tf.nn.relu(K - temp)

    def exact_price(self, t: tf.Tensor, x: tf.Tensor, params: tf.Tensor):
        """
        Implement the BS formula
        """
        T = self.config.T
        r = tf.expand_dims(params[:, :, :, 0], -1)
        vol = tf.expand_dims(params[:, :, :, 1], -1)
        k = tf.expand_dims(params[:, :, :, 2], -1)
        K = k * self.config.x_init
        d1 = (tf.math.log(x / K) + (r + vol ** 2 / 2) * (T - t)) / (vol * tf.math.sqrt(T - t) + 0.000001)
        d2 = d1 - vol * tf.math.sqrt(T - t)
        return x * dist.cdf(d1) - K * tf.exp(-r * (T - t)) * dist.cdf(d2)
        
    
    def exact_delta(self, t: tf.Tensor, x: tf.Tensor, params: tf.Tensor):
        T = self.config.T
        r = tf.expand_dims(params[:, :, :, 0], -1)
        vol = tf.expand_dims(params[:, :, :, 1], -1)
        k = tf.expand_dims(params[:, :, :, -1], -1)
        K = k * self.config.x_init
        d1 = (tf.math.log(x / K) + (r + vol ** 2 / 2) * (T - t)) / (vol * tf.math.sqrt(T - t) + 0.000001)
        delta =  x * dist.cdf(d1) * vol 
        return delta

    def sample_parameters(self, N=100, training=True):  # N is the time of batch size
        if training:
            strike_range = self.strike_range
            num_params = int(N * self.config.batch_size)
            m = tf.math.maximum(tf.random.uniform([num_params, 1], strike_range[0], strike_range[1]), strike_range[0] * 0.05)
            return m
        else:
            strike_range = self.val_config.strike_range
            num_params = int(N * self.val_config.batch_size)
            m = tf.math.maximum(tf.random.uniform([num_params, 1], strike_range[0], strike_range[1]), strike_range[0] * 0.08)
            return m


class TimeEuropean(EuropeanOption):
    def __init__(self, config):
        super(TimeEuropean, self).__init__(config)
    
    def exact_price(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor):
        """
        Implement the BS formula under time dependent case
        """
        T = self.config.T
        r0 = tf.expand_dims(u_hat[:, :, :, 0], -1)
        r1 = tf.expand_dims(u_hat[:, :, :, 1], -1)
        r2 = tf.expand_dims(u_hat[:, :, :, 2], -1)
        r_T = r0 * T + r1 * T ** 2 /2 + r2 * T ** 3/3 
        r_t = r0 * t + r1 * t ** 2 /2 + r2 * t ** 3/3 
        r = (r_T - r_t)/((T - t) + 0.000001)

        s0 = tf.expand_dims(u_hat[:, :, :, 3], -1)
        beta = tf.expand_dims(u_hat[:, :, :, 4], -1)
        vol2 = s0/2/beta * (1.0 - tf.exp(2*beta*(t - T))) / ((T-t) + 0.000001)

        vol = tf.expand_dims(u_hat[:, :, :, 3], -1)
        K = tf.expand_dims(u_hat[:, :, :, -1], -1)
        d1 = (tf.math.log(x / K) + (r + vol2/ 2) * (T - t)) / (tf.math.sqrt(vol2 * (T-t)))
        d2 = d1 - (tf.math.sqrt(vol2 * (T-t)))
        c = self.config.x_init * (x * dist.cdf(d1) - K * tf.exp(-r * (T - t)) * dist.cdf(d2))
        return c


class EuropeanSwap(EuropeanOption):
    def __init__(self, config):
        super(EuropeanSwap, self).__init__(config)
        self.strike = 0.05 #we assume all products have a unified strike

    def payoff(self, x: tf.Tensor, param: tf.Tensor, **kwargs):
        """
        Parameters
        ----------
        x: tf.Tensor
            Asset price path. Tensor of shape (batch_size, sample_size, time_steps, d)
        param: tf.Tensor
            The parameter vectors of brunch net input and the last entry is for strike.
        Returns
        -------
        payoff: tf.Tensor
            swap payoff. Tensor of shape (batch_size, sample_size, 1)
        """
        k = tf.expand_dims(param[:, :, 0, -1], axis=-1)
        temp = tf.reduce_mean(x[:, :, -1, :self.config.dim], axis=-1, keepdims=True) # (B, M, 1)
        K = k * self.config.x_init
        return temp - K
    
    def exact_price(self, t: tf.Tensor, x: tf.Tensor, params: tf.Tensor):
        """
        Implement the BS formula
        """
        k = tf.expand_dims(params[:, :, :, -1], axis=-1)
        x = tf.reduce_mean(x[:, :, :, :self.config.dim], axis=-1, keepdims=True)
        T = self.config.T
        r = tf.expand_dims(params[:, :, :, 0], -1)
        K = k * self.config.x_init
        c = x - K * tf.exp(-r * (T-t))
        return c
    
    # def exact_delta(self, t: tf.Tensor, x: tf.Tensor, params: tf.Tensor):
    #     s = x[:, :, :, :self.config.dim]
    #     v = x[:, :, :, self.config.dim:]
    #     return tf.math.sqrt(v) * s
    

    


class EuropeanBasketOption(EuropeanOption):
    def __init__(self, config):
        super(EuropeanBasketOption, self).__init__(config)
    
    def payoff(self, x: tf.Tensor, u_hat: tf.Tensor, **kwargs):
        """
        Parameters
        ----------
        x: tf.Tensor
            Asset price path. Tensor of shape (batch_size, sample_size, time_steps, d)
        param: tf.Tensor
            The parameter vectors of brunch net input and the last entry is for strike.
        Returns
        -------
        payoff: tf.Tensor
            basket option payoff. Tensor of shape (batch_size, sample_size, 1)
            when self.config.dim = 1, this reduced to 1-d vanilla call payoff
        """
        k = tf.expand_dims(u_hat[:, :, 0, -1], axis=-1) # (B, M, 1)
        K = k * self.config.x_init
        temp = tf.exp(tf.reduce_mean(tf.math.log(x[:, :, -1, :]), axis=-1, keepdims=True))# (B, M, 1)
        return tf.nn.relu(temp - K)
    
    def exact_price(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor):
        """
        Implement the BS formula
        """
        d = self.config.dim
        T = self.config.T
        r = tf.expand_dims(u_hat[:, :, :, 0], -1)
        vol = tf.expand_dims(u_hat[:, :, :, 1], -1)
        rho = tf.expand_dims(u_hat[:, :, :, 2], -1)
        k = tf.expand_dims(u_hat[:, :, :, 3], -1)
        K = k * self.config.x_init
        vol_bar = vol * tf.math.sqrt(1/d + rho * (1-1/d))
        S_pi = tf.exp(tf.reduce_mean(tf.math.log(x), axis=-1, keepdims=True))
        F = S_pi * tf.exp((r - vol**2/2 + vol_bar**2/2) * (T - t))
        d_1 = (tf.math.log(F/K) + vol_bar ** 2/2 * (T - t))/vol_bar/tf.sqrt(T - t)
        d_2 = d_1 - vol_bar * tf.sqrt(T - t)
        c = F * dist.cdf(d_1) - K * dist.cdf(d_2)
        return c
    
class  TimeEuropeanBasketOption(EuropeanBasketOption):
    def __init__(self, config):
        super(TimeEuropeanBasketOption, self).__init__(config)
    
    def exact_price(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor):
        d = self.config.dim
        T = self.config.T
        r0 = tf.expand_dims(u_hat[:, :, :, 0], -1)
        r1 = tf.expand_dims(u_hat[:, :, :, 1], -1)
        r2 = tf.expand_dims(u_hat[:, :, :, 2], -1)
        r_T = r0 * T + r1 * T ** 2 /2 + r2 * T ** 3/3 
        r_t = r0 * t + r1 * t ** 2 /2 + r2 * t ** 3/3 
        r = (r_T - r_t)/((T - t) + 0.000001)

        s0 = tf.expand_dims(u_hat[:, :, :, 3], -1)
        beta = tf.expand_dims(u_hat[:, :, :, 4], -1)
        vol2 = s0/2/beta * (1.0 - tf.exp(2*beta*(t - T))) / ((T-t) + 0.000001)
        rho = tf.expand_dims(u_hat[:, :, :, 2], -1)
        k = tf.expand_dims(u_hat[:, :, :, -1], -1)
        K = k * self.config.x_init
        vol = tf.math.sqrt(vol2)

        vol_bar = vol * tf.math.sqrt(1/d + rho * (1-1/d))
        S_pi = tf.exp(tf.reduce_mean(tf.math.log(x), axis=-1, keepdims=True))
        F = S_pi * tf.exp((r - vol**2/2 + vol_bar**2/2) * (T - t))
        d_1 = (tf.math.log(F/K) + vol_bar ** 2/2 * (T - t))/vol_bar/tf.sqrt(T - t)
        d_2 = d_1 - vol_bar * tf.sqrt(T - t)
        c = F * dist.cdf(d_1) - K * dist.cdf(d_2)
        return c
    


class LookbackOption(EuropeanOption):
    def __init__(self, config):
        super(LookbackOption, self).__init__(config)

    def markovian_var(self, x: tf.Tensor):
        """
        x is a (B, M, N, d) size
        The output is the cummin on the axis=2
        """
        m_pre = x[:, :, 0, :]
        m_list = [m_pre]
        for i in range(1, tf.shape(x)[2]):
            m_pre = tf.math.minimum(m_pre, x[:, :, i, :])
            m_list.append(m_pre)
        markov = tf.stack(m_list, axis=2)
        return markov

    def payoff(self, x: tf.Tensor, u_hat: tf.Tensor):
        temp = tf.reduce_mean(x[:, :, -1, :self.config.dim], axis=-1, keepdims=True) # (B, M, 1)
        float_min = tf.math.reduce_min(x[:, :, :, :self.config.dim], axis=2, keepdims=True) # (B, M, 1, d)
        temp_min = tf.reduce_mean(float_min, axis=-1) # (B,M,1)
        return temp - temp_min
    
    def exact_price(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor):
        """
        In this x has the dimension:
        (B, M, T, d+d) since this ia a concat with markovian variable
        """
        dim = self.config.dim
        T = self.config.T
        r = tf.expand_dims(u_hat[:, :, :, 0], -1)
        vol = tf.expand_dims(u_hat[:, :, :, 1], -1)
        X_t = x[...,:dim]
        m_t = x[...,dim:]
        a_1 = (tf.math.log(X_t/m_t) + (r + vol ** 2/2) * (T - t))/(vol * tf.math.sqrt(T - t))
        a_2 = a_1 - vol * tf.math.sqrt(T - t)
        a_3 = a_1 - 2 * r/vol * tf.math.sqrt(T - t)
        y_t = X_t * dist.cdf(a_1) - m_t * tf.exp(-r * (T-t)) * dist.cdf(a_2) - \
            X_t * vol ** 2/(2 * r) * (dist.cdf(-a_1) - tf.exp(-r * (T-t)) * (m_t/X_t) ** (2 * r / vol**2) * dist.cdf(-a_3)) 
        y_t = tf.reduce_mean(y_t, axis=-1, keepdims=True)
        return y_t

class GeometricAsian(EuropeanOption):
    def __init__(self, config):
        """
        multidimensional Geomatric Asian option
        Parameters
        ----------
        K: float or torch.tensor
            Strike. Id K is a tensor, it needs to have shape (batch_size)
        """
        super(GeometricAsian, self).__init__(config)
    
    def markovian_var(self, x: tf.Tensor):
        """
        x is a (B, M, N, d) size or (B, M, N, d+d) (SV model)
        The output is the running integral on the axis=2
        The geometric average is:

        $$
        G_t=\exp \left\{\frac{1}{t} \int_0^t \log x_u d u\right\}
        $$
        """
        dt = self.config.dt
        x = x[...,:self.config.dim]
        dt_tensor = tf.math.cumsum(tf.ones(tf.shape(x)) * dt, axis=2)
        sumlog = tf.math.cumsum(tf.math.log(x), axis=2)
        log_average = sumlog * dt / dt_tensor
        geo_average = tf.exp(log_average)
        return geo_average
   
    def payoff(self, x_arg: tf.Tensor, param: tf.Tensor):
        """
        Parameters
        ----------
        x: tf.Tensor
            Asset price path. Tensor of shape (B, M, N, d + d)
        param: tf.Tensor
            The parameter vectors of NO input and the last entry is for strike.
        Returns
        -------
        payoff: tf.Tensor
            Asian option payoff. Tensor of shape (B, M, 1)
        """
        k = tf.expand_dims(param[:, :, 0, -1], axis=-1) # (B, M, 1)
        K = k * self.config.x_init
        dt = self.config.dt
        T = self.config.T
        x = x_arg[...,:self.config.dim]
        geo_mean = tf.reduce_mean(tf.exp(tf.reduce_sum(tf.math.log(x), axis=2) * dt / T), axis=-1, keepdims=True)
        return tf.nn.relu(geo_mean - K)

    def exact_price(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor):
        """
        In this x has the dimension:
        (B, M, T, d+d) since this ia a concat with markovian variable
        """
        T = self.config.T
        r = tf.expand_dims(u_hat[:, :, :, 0], -1)
        vol = tf.expand_dims(u_hat[:, :, :, 1], -1)
        k = tf.expand_dims(u_hat[:, :, :, 2], -1)
        K = k * self.config.x_init
        y_t = x[...,:self.config.dim]
        G_t = x[...,self.config.dim:]
        mu_bar = (r - vol**2/2) * (T - t)**2/2/T
        vol_bar = vol/T * tf.math.sqrt((T - t) ** 3/3)
        d_2 = (t/T * tf.math.log(G_t) + (1 - t/T) * tf.math.log(y_t) + mu_bar - tf.math.log(K))/vol_bar
        d_1 = d_2 + vol_bar
        A = tf.exp(-r * (T - t))*(G_t ** (t/T) * y_t ** (1 - t/T) * tf.exp(mu_bar + vol_bar**2/2) * dist.cdf(d_1) - K * dist.cdf(d_2))
        return A
    
    # def exact_delta(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor):
    #     """
    #     In this x has the dimension:
    #     (B, M, T, d+d) since this ia a concat with markovian variable
    #     """
    #     T = self.config.T
    #     r = tf.expand_dims(u_hat[:, :, :, 0], -1)
    #     vol = tf.expand_dims(u_hat[:, :, :, 1], -1)
    #     k = tf.expand_dims(u_hat[:, :, :, 2], -1)
    #     K = k * self.config.x_init
    #     y_t = x[...,:self.config.dim]
    #     G_t = x[...,self.config.dim:]
    #     with tf.GradientTape(watch_accessed_variables=False) as tape:
    #         tape.watch(y_t)
    #         mu_bar = (r - vol**2/2) * (T - t)**2/2/T
    #         vol_bar = vol/T * tf.math.sqrt((T - t) ** 3/3)
    #         d_2 = (t/T * tf.math.log(G_t) + (1 - t/T) * tf.math.log(y_t) + mu_bar - tf.math.log(K))/vol_bar
    #         d_1 = d_2 + vol_bar
    #         A = tf.exp(-r * (T - t))*(G_t ** (t/T) * y_t ** (1 - t/T) * tf.exp(mu_bar + vol_bar**2/2) * dist.cdf(d_1) - K * dist.cdf(d_2))
    #         delta = tape.gradient(A, y_t)
    #     z = vol * y_t * delta
    #     return z



        
class BermudanBasketPut(BaseOption):
    def __init__(self, config):
        super(BermudanBasketPut, self).__init__(config)
        

    def payoff(self, x: tf.Tensor, u_hat: tf.Tensor, **kwargs):
        """
        Parameters
        ----------
        x: tf.Tensor
            Asset price path. Tensor of shape (batch_size, sample_size, time_steps, d)
        param: tf.Tensor
            The parameter vectors of brunch net input and the last entry is for strike.
        Returns
        -------
        payoff: tf.Tensor
            basket option payoff. Tensor of shape (batch_size, sample_size, 1)
            when self.config.dim = 1, this reduced to 1-d vanilla call payoff
        """
        k = tf.expand_dims(u_hat[:, :, 0, -1], axis=-1) # (B, M, 1)
        temp = tf.reduce_mean(x[:, :, -1, :], axis=-1, keepdims=True) # (B, M, 1)
        return tf.nn.relu(k - temp)
    
    def early_payoff(self, x: tf.Tensor, u_hat: tf.Tensor, **kwargs):
        """
        this function calculate the payoff of early exercise at certain time step,
        x has the shape: [B, M, d]
        u_hat has the shape [B, M, k]
        return [B, M, 1]
        """
        temp = temp = tf.reduce_mean(x, axis=-1, keepdims=True) # [B, M, 1]
        k = tf.expand_dims(u_hat[:, :, -1], axis=-1) # [B, M, 1]
        return tf.nn.relu(k - temp)
    
    def exact_price(self, t: tf.Tensor, x: tf.Tensor, params):
        """
        Implement the BS formula
        """
        if self.config.dim==1:
            T = self.config.T
            r = tf.expand_dims(params[:, :, :, 0], -1)
            vol = tf.expand_dims(params[:, :, :, 1], -1)
            k = tf.expand_dims(params[:, :, :, 2], -1)
            K = k * self.config.x_init
            d1 = (tf.math.log(x / K) + (r + vol ** 2 / 2) * (T - t)) / (vol * tf.math.sqrt(T - t) + 0.000001)
            d2 = d1 - vol * tf.math.sqrt(T - t)
            c = x * dist.cdf(d1) - K * tf.exp(-r * (T - t)) * dist.cdf(d2)
            p = K * tf.exp(-r * (T - t)) * dist.cdf(-d2) - x * dist.cdf(-d1)
            return p
        else:
            return None

class InterestRateSwap(BaseOption):
    def __init__(self, config):
        super(InterestRateSwap, self).__init__(config)
        self.strike_range = self.config.strike_range #the distribution of the fixed coupon rate
        self.leg_dates = self.config.leg_dates #a list with first element $T_0$ as the first 
        # payment date and last element $T_n$ as the last payment date
        self.fix_rate = 0.01
        self.sde = HullWhiteModel(config)
        self.notional = 10

    @property
    def delta_t(self):
        return self.leg_dates[1] - self.leg_dates[0]

    def reset_dates(self, leg_dates: list):
        self.leg_dates = leg_dates


    @property
    def float_start_dates(self):
        return self.leg_dates[:-1]
    
    @property
    def float_end_dates(self):
        return self.leg_dates[1:]
    
    def fix_legs(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        """
        return a tensor containing the leg value at given time and state for all maturities in the list
        t: [B, M, 1]
        x: [B, M, 1]
        u_hat: [B, M, k]
        """
        # fix_rate = tf.expand_dims(u_hat[:, :, -1], -1) # (B, M, 1)
        p_list = [self.fix_rate * self.sde.zcp_value(t, x, u_hat, end_date) * self.delta_t 
                  for end_date in self.float_end_dates]
        fix_legs = tf.concat(p_list, axis=-1)
        return fix_legs
    
    def float_legs(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        """
        return a tensor containing the leg value at given time and state for all maturities in the list
        """
        p_list = [self.one_float_leg(t, x, u_hat, end_date) 
                  for end_date in self.float_end_dates] 
        float_legs = tf.concat(p_list, axis=-1)
        return float_legs
    
    def one_float_leg(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor, terminal_date: tf.Tensor) -> tf.Tensor:
        """
        return P(t, T_{i-1}) - P(t, T_{i}) 
        T_{i-1} + \Delta T = T_{i}
        T_{i-1}: float start date
        T_{i}: float end date
        """
        float_leg = self.sde.zcp_value(t, x, u_hat, terminal_date - self.delta_t) - \
            self.sde.zcp_value(t, x, u_hat, terminal_date)
        return tf.where(t > terminal_date - self.delta_t + 0.001, 0., float_leg)
    
    def swap_value(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        """
        t: (B, M, 1)
        x: (B, M, d), d=1 usually
        u_hat: [B, M, k]
        """
        float_legs = self.float_legs(t, x, u_hat) # (B, M, T)
        fix_legs = self.fix_legs(t, x, u_hat) # (B, M, T)
        return self.notional * tf.reduce_sum(fix_legs - float_legs, axis=-1, keepdims=True) # (B, M, 1)
    
    def payoff_inter(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        p_12 = self.zcp(1.0, x[:,:,-1,:], u_hat[:,:,-1,:], 2.0)
        p_13 = self.zcp(1.0, x[:,:,-1,:], u_hat[:,:,-1,:], 3.0)
        return 1.0 - self.fix_rate * p_12 - (1.0 + self.fix_rate) * p_13
    
    def payoff_at_maturity(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        p_23 = self.zcp(2.0, x[:,:,-1,:], u_hat[:,:,-1,:], 3.0)
        return 1.0 - (1.0 + self.fix_rate) * p_23
    
    def zcp(self, t_s: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor, t_e: tf.Tensor) -> tf.Tensor:
        """
        x: [B, M, 1] must corresponded to the time t_s
        u_hat: [B, M, 1]
        """
        kappa = tf.expand_dims(u_hat[..., 0], axis=-1)
        theta = tf.expand_dims(u_hat[..., 1], axis=-1)
        sigma = tf.expand_dims(u_hat[..., 2], axis=-1)
        B = (1 - tf.exp(-kappa * (t_e - t_s)))/ kappa
        A = tf.exp((B - t_e + t_s)*(kappa**2 * theta - sigma**2/2)/kappa**2 +\
                     (sigma*B)**2/(4*kappa))
        p_se = A * tf.exp(-B * tf.reduce_sum(x, axis=-1, keepdims=True))
        return p_se

    def exact_price(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        """
        t: [B, M, N, 1]
        x: [B, M, N, 1]
        u_hat: [B, M, N, 4]
        """
        kappa = tf.expand_dims(u_hat[..., 0], axis=-1)
        theta = tf.expand_dims(u_hat[..., 1], axis=-1)
        sigma = tf.expand_dims(u_hat[..., 2], axis=-1)
        B1 = (1.0 - tf.exp(-kappa * (1.0 - t)))/ kappa
        A1 = tf.exp((B1 - 1.0 + t)*(kappa**2 * theta - sigma**2/2)/kappa**2 +\
                     (sigma*B1)**2/(4*kappa))
        p_t1 = A1* tf.exp(-B1 * tf.reduce_sum(x, axis=-1, keepdims=True))

        B2 = (1.0 - tf.exp(-kappa * (2.0 - t)))/ kappa
        A2 = tf.exp((B2 - 2.0 + t)*(kappa**2 * theta - sigma**2/2)/kappa**2 +\
                     (sigma*B2)**2/(4*kappa))
        p_t2 = A2 * tf.exp(-B2 * tf.reduce_sum(x, axis=-1, keepdims=True))

        B3 = (1.0 - tf.exp(-kappa * (3.0 - t)))/ kappa
        A3 = tf.exp((B3 - 3.0 + t)*(kappa**2 * theta - sigma**2/2)/kappa**2 +\
                     (sigma*B3)**2/(4*kappa))
        p_t3 = A3 * tf.exp(-B3 * tf.reduce_sum(x, axis=-1, keepdims=True))

        v = p_t1 - self.fix_rate * p_t2 - (1.0 + self.fix_rate) * p_t3
        return v
    
    # def payoff_inter(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
    #     return self.swap_value(t, x, u_hat)
    
    # def payoff_at_maturity(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
    #     return 0.0
    
class ZeroCouponBond(BaseOption):
    def __init__(self, config):
        super(ZeroCouponBond, self).__init__(config)
        self.strike_range = self.config.strike_range #the distribution of the fixed coupon rate
        self.leg_dates = self.config.leg_dates #a list with first element $T_0$ as the first 
        # payment date and last element $T_n$ as the last payment date
        self.terminal_date = self.config.T
        self.sde = HullWhiteModel(config)
        self.fix_rate = 0.05
    
    def payoff(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        # return 1.0
        p_12 = self.sde.zcp_value(1.0, x[:,:,-1,:], u_hat[:,:,-1,:], 2.0)
        return 1.0 - (1.0 + self.fix_rate) * p_12
    
    def exact_price(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        """
        t: [B, M, N, 1]
        x: [B, M, N, 1]
        u_hat: [B, M, N, 4]
        """
        terminal_date = self.config.T
        kappa = tf.expand_dims(u_hat[..., 0], axis=-1)
        theta = tf.expand_dims(u_hat[..., 1], axis=-1)
        sigma = tf.expand_dims(u_hat[..., 2], axis=-1)
        B = (1 - tf.exp(-kappa * (terminal_date - t)))/ kappa
        A = tf.exp((B - terminal_date + t)*(kappa**2 * theta - sigma**2/2)/kappa**2 +\
                     (sigma*B)**2/(4*kappa))
        p_t1 = A * tf.exp(-B * tf.reduce_sum(x, axis=-1, keepdims=True))

        B_ = (1 - tf.exp(-kappa * (2.0 - t)))/ kappa
        A_ = tf.exp((B_ - 2.0 + t)*(kappa**2 * theta - sigma**2/2)/kappa**2 +\
                     (sigma*B_)**2/(4*kappa))
        p_t2 = A_ * tf.exp(-B_ * tf.reduce_sum(x, axis=-1, keepdims=True))
        v = p_t1 - (1 + self.fix_rate) * p_t2
        return v 


    
    



