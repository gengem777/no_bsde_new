import tensorflow as tf
from dataclasses import dataclass
import tensorflow_probability as tfp
from sde import HullWhiteModel

tfd = tfp.distributions
dist = tfd.Normal(loc=0.0, scale=1.0)


@dataclass
class BaseOption:
    r"""
    This class give methods corresponded to a certain derivative product.
    The certain product is developed in child classes. For each class, we have following method:
    payoff: This function dipicts the payoff of each product.
    exact_price: This function give the analytical solution of the product, which serves as benchmark
    sample_parameters: this method sample the parameter related to the product.
                    For example, strike price is somewhat needs to be sampled.
    """

    def __init__(self, config):
        self.config = config.eqn_config
        self.val_config = config.val_config
        self.strike_range = self.config.strike_range
        self.exer_dates = [
            0
        ] + self.config.exercise_date  # a list of early exercise dates, [0, 1] for European
        self.steps = self.config.time_steps
        self.time_steps_one_year = self.config.time_steps_one_year
        self.exer_index = [
            int(num) for num in [k * self.time_steps_one_year for k in self.exer_dates]
        ]  # the list of index of the tensor of time stamp

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
        return par  # [B, 1] -> [B, M, N, 1]

    def sample_parameters(self, N=100, training=True):  # N is the time of batch size
        if training:
            strike_range = self.strike_range
            num_params = int(N * self.config.batch_size)
            m = tf.math.maximum(
                tf.random.normal([num_params, 1], strike_range[0], strike_range[1]),
                strike_range[0] * 0.0001,
            )
            return m  # [B, 1]
        else:
            strike_range = self.val_config.strike_range
            num_params = int(N * self.val_config.batch_size)
            m = tf.math.maximum(
                tf.random.normal([num_params, 1], strike_range[0], strike_range[1]),
                strike_range[0] * 0.0001,
            )
            return m  # [B, 1]


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
        self.epsilon = 1e-6

    def payoff(self, x: tf.Tensor, param: tf.Tensor, **kwargs):
        r"""
        The payoff function is \max{S_T - K, 0} where:
          K is strike price which is included into u_hat
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
        k = tf.expand_dims(param[:, :, 0, -1], axis=-1)  # (B, M, 1)
        temp = tf.reduce_mean(x[:, :, -1, :], axis=-1, keepdims=True)  # (B, M, 1)
        K = k * self.config.x_init
        if self.style == "call":
            return tf.nn.relu(temp - K)
        else:
            return tf.nn.relu(K - temp)

    def exact_price(self, t: tf.Tensor, x: tf.Tensor, params: tf.Tensor):
        r"""
        Implement the BS formula:
                u\left(t, S_t; (r, \sigma, \kappa)\right)=S_t N\left( d_1\right) -  \kappa S_0 e^{-\tilde{r}(T - t)}N\left( d_2\right),
            where
                d_1 = \frac{\log\left(\frac{S_t}{ \kappa S_0}\right)+\left(\tilde{r} + \frac{\tilde{\sigma}^2}{2}\right)(T-t)}{\tilde{\sigma}\sqrt{T-t}},
        \quad d_2 = d_1 - \sigma\sqrt{T-t}.
        """
        T = self.config.T
        r = tf.expand_dims(params[:, :, :, 0], -1)
        vol = tf.expand_dims(params[:, :, :, 1], -1)
        k = tf.expand_dims(params[:, :, :, 2], -1)
        K = k * self.config.x_init
        d1 = (tf.math.log(x / K) + (r + vol**2 / 2) * (T - t)) / (
            vol * tf.math.sqrt(T - t) + self.epsilon
        )
        d2 = d1 - vol * tf.math.sqrt(T - t)
        return x * dist.cdf(d1) - K * tf.exp(-r * (T - t)) * dist.cdf(d2)


class TimeEuropean(EuropeanOption):
    def __init__(self, config):
        super(TimeEuropean, self).__init__(config)

    def exact_price(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor):
        """
        Implement the BS formula under time dependent case
        t: [B, M, N, 1]
        x: [B, M, N, d]
        u_hat: [B, M, N, k]
        return: [B, M, N, 1]
        """
        T = self.config.T
        r0 = tf.expand_dims(u_hat[:, :, :, 0], -1)
        r1 = tf.expand_dims(u_hat[:, :, :, 1], -1)
        r2 = tf.expand_dims(u_hat[:, :, :, 2], -1)
        r_T = r0 * T + r1 * T**2 / 2 + r2 * T**3 / 3
        r_t = r0 * t + r1 * t**2 / 2 + r2 * t**3 / 3
        r = (r_T - r_t) / ((T - t) + self.epsilon)
        s0 = tf.expand_dims(u_hat[:, :, :, 3], -1)
        beta = tf.expand_dims(u_hat[:, :, :, 4], -1)
        vol2 = (
            s0
            / 2
            / beta
            * (1.0 - tf.exp(2 * beta * (t - T)))
            / ((T - t) + self.epsilon)
        )
        K = tf.expand_dims(u_hat[:, :, :, -1], -1)
        d1 = (tf.math.log(x / K) + (r + vol2 / 2) * (T - t)) / (
            tf.math.sqrt(vol2 * (T - t))
        )
        d2 = d1 - (tf.math.sqrt(vol2 * (T - t)))
        c = self.config.x_init * (
            x * dist.cdf(d1) - K * tf.exp(-r * (T - t)) * dist.cdf(d2)
        )
        return c  # [B, M, N, 1]


class EuropeanSwap(EuropeanOption):
    def __init__(self, config):
        super(EuropeanSwap, self).__init__(config)
        self.strike = 0.05  # we assume all products have a unified strike

    def payoff(self, x: tf.Tensor, param: tf.Tensor, **kwargs):
        r"""
        The payoff function is S_T - K where:
          K is forward price which is predetermined which is included into u_hat
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
        temp = tf.reduce_mean(
            x[:, :, -1, : self.config.dim], axis=-1, keepdims=True
        )  # [B, M, 1]
        K = k * self.config.x_init
        return temp - K  # [B, M, 1]

    def exact_price(self, t: tf.Tensor, x: tf.Tensor, params: tf.Tensor):
        """
        Implement the forward analytical formula
        t: [B, M, N, 1]
        x: [B, M, N, d]
        u_hat: [B, M, N, k]
        return: [B, M, N, 1]
        """
        k = tf.expand_dims(params[:, :, :, -1], axis=-1)
        x = tf.reduce_mean(x[:, :, :, : self.config.dim], axis=-1, keepdims=True)
        T = self.config.T
        r = tf.expand_dims(params[:, :, :, 0], -1)
        K = k * self.config.x_init
        c = x - K * tf.exp(-r * (T - t))
        return c  # [B, M, N, 1]


class EuropeanBasketOption(EuropeanOption):
    def __init__(self, config):
        super(EuropeanBasketOption, self).__init__(config)

    def payoff(self, x: tf.Tensor, u_hat: tf.Tensor, **kwargs):
        r"""
        The payoff function is \max{G_T - K, 0} where:
          K is strike price which is included into u_hat
          G_T is the geometric average of the basket at terminal time T.
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
        k = tf.expand_dims(u_hat[:, :, 0, -1], axis=-1)  # [B, M, 1]
        K = k * self.config.x_init
        temp = tf.exp(
            tf.reduce_mean(tf.math.log(x[:, :, -1, :]), axis=-1, keepdims=True)
        )  # [B, M, 1]
        return tf.nn.relu(temp - K)  # [B, M, 1]

    def exact_price(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor):
        """
        Implement the BS formula
        t: [B, M, N, 1]
        x: [B, M, N, d]
        u_hat: [B, M, N, k]
        return: [B, M, N, 1]
        """
        d = self.config.dim
        T = self.config.T
        r = tf.expand_dims(u_hat[:, :, :, 0], -1)
        vol = tf.expand_dims(u_hat[:, :, :, 1], -1)
        rho = tf.expand_dims(u_hat[:, :, :, 2], -1)
        k = tf.expand_dims(u_hat[:, :, :, 3], -1)
        K = k * self.config.x_init
        vol_bar = vol * tf.math.sqrt(1 / d + rho * (1 - 1 / d))
        S_pi = tf.exp(tf.reduce_mean(tf.math.log(x), axis=-1, keepdims=True))
        F = S_pi * tf.exp((r - vol**2 / 2 + vol_bar**2 / 2) * (T - t))
        d_1 = (
            (tf.math.log(F / K) + vol_bar**2 / 2 * (T - t)) / vol_bar / tf.sqrt(T - t)
        )
        d_2 = d_1 - vol_bar * tf.sqrt(T - t)
        c = F * dist.cdf(d_1) - K * dist.cdf(d_2)
        return c  # [B, M, N, 1]


class TimeEuropeanBasketOption(EuropeanBasketOption):
    def __init__(self, config):
        super(TimeEuropeanBasketOption, self).__init__(config)
        self.epsilon = 1e-6

    def exact_price(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor):
        d = self.config.dim
        T = self.config.T
        r0 = tf.expand_dims(u_hat[:, :, :, 0], -1)
        r1 = tf.expand_dims(u_hat[:, :, :, 1], -1)
        r2 = tf.expand_dims(u_hat[:, :, :, 2], -1)
        r_T = r0 * T + r1 * T**2 / 2 + r2 * T**3 / 3
        r_t = r0 * t + r1 * t**2 / 2 + r2 * t**3 / 3
        r = (r_T - r_t) / ((T - t) + self.epsilon)
        s0 = tf.expand_dims(u_hat[:, :, :, 3], -1)
        beta = tf.expand_dims(u_hat[:, :, :, 4], -1)
        vol2 = (
            s0
            / 2
            / beta
            * (1.0 - tf.exp(2 * beta * (t - T)))
            / ((T - t) + self.epsilon)
        )
        rho = tf.expand_dims(u_hat[:, :, :, 2], -1)
        k = tf.expand_dims(u_hat[:, :, :, -1], -1)
        K = k * self.config.x_init
        vol = tf.math.sqrt(vol2)
        vol_bar = vol * tf.math.sqrt(1 / d + rho * (1 - 1 / d))
        S_pi = tf.exp(tf.reduce_mean(tf.math.log(x), axis=-1, keepdims=True))
        F = S_pi * tf.exp((r - vol**2 / 2 + vol_bar**2 / 2) * (T - t))
        d_1 = (
            (tf.math.log(F / K) + vol_bar**2 / 2 * (T - t)) / vol_bar / tf.sqrt(T - t)
        )
        d_2 = d_1 - vol_bar * tf.sqrt(T - t)
        c = F * dist.cdf(d_1) - K * dist.cdf(d_2)
        return c  # [B, M, N, 1]


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
        r"""
        The payoff function is \max{S_T - m_T, 0} where:
          m_T is floating minimum at time T.
        Parameters
        ----------
        x: tf.Tensor
            Asset price path. Tensor of shape [B, M, N, 2d]
            where x[:,:,:,1:] denotes the markovian variable m_t
        param: tf.Tensor
            The parameter vectors of brunch net input and we do not have strike in this product.
        Returns
        -------
        payoff: tf.Tensor
            lookback option payoff. Tensor of shape (batch_size, sample_size, 1)
            when self.config.dim = 1, this reduced to 1-d vanilla call payoff
        """
        temp = tf.reduce_mean(
            x[:, :, -1, : self.config.dim], axis=-1, keepdims=True
        )  # (B, M, 1)
        float_min = tf.math.reduce_min(
            x[:, :, :, : self.config.dim], axis=2, keepdims=True
        )  # (B, M, 1, d)
        temp_min = tf.reduce_mean(float_min, axis=-1)  # (B,M,1)
        return temp - temp_min

    def exact_price(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor):
        r"""
        In this x has the dimension:
        (B, M, T, d+d) since this ia a concat with markovian variable
        The exact formula is:
        \begin{equation*}
		u(t, S_t, m_t; (r, \sigma))=S_t N\left(a_1\right)-m_t e^{-\tilde{r}(T-t)} N\left(a_2\right)-S_t 
        \frac{\tilde{\sigma}^2}{2 \tilde{r}}\left(N\left(-a_1\right)\
        -e^{-\tilde{r}(T-t)}\left(\frac{m_t}{S_t}\right)^{2 \tilde{r} / \tilde{\sigma}^2} N\left(-a_3\right)\right).
	    \end{equation*}
	    where
	    \begin{equation*}
		a_1=\frac{\log \left(S_t / m_t\right)+\left(\tilde{r}+\tilde{\sigma}^2 / 2\right)(T-t)}{\tilde{\sigma} \sqrt{T-t}}, 
        \quad a_2=a_1-\tilde{\sigma} \sqrt{T-t}, 
        \quad a_3=a_1-\frac{2 \tilde{r}}{\tilde{\sigma}} \sqrt{T-t}.
	    \end{equation*}
        t: [B, M, N, 1]
        x: [B, M, N, d]
        u_hat: [B, M, N, k]
        return: [B, M, N, 1]
        """
        dim = self.config.dim
        T = self.config.T
        r = tf.expand_dims(u_hat[:, :, :, 0], -1)
        vol = tf.expand_dims(u_hat[:, :, :, 1], -1)
        X_t = x[..., :dim]
        m_t = x[..., dim:]
        a_1 = (tf.math.log(X_t / m_t) + (r + vol**2 / 2) * (T - t)) / (
            vol * tf.math.sqrt(T - t)
        )
        a_2 = a_1 - vol * tf.math.sqrt(T - t)
        a_3 = a_1 - 2 * r / vol * tf.math.sqrt(T - t)
        y_t = (
            X_t * dist.cdf(a_1)
            - m_t * tf.exp(-r * (T - t)) * dist.cdf(a_2)
            - X_t
            * vol**2
            / (2 * r)
            * (
                dist.cdf(-a_1)
                - tf.exp(-r * (T - t))
                * (m_t / X_t) ** (2 * r / vol**2)
                * dist.cdf(-a_3)
            )
        )
        y_t = tf.reduce_mean(y_t, axis=-1, keepdims=True)
        return y_t  # [B, M, N, 1]


class GeometricAsian(EuropeanOption):
    r"""
    multidimensional Geomatric Asian option
    Parameters
    ----------
    K: float or torch.tensor
        Strike. Id K is a tensor, it needs to have shape (batch_size)
    """

    def __init__(self, config):
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
        x = x[..., : self.config.dim]  # [B, M, N, d]
        dt_tensor = tf.math.cumsum(tf.ones(tf.shape(x)) * dt, axis=2)  # [B, M, N, d]
        sumlog = tf.math.cumsum(tf.math.log(x), axis=2)  # [B, M, N, d]
        log_average = sumlog * dt / dt_tensor  # [B, M, N, d]
        geo_average = tf.exp(log_average)  # [B, M, N, d]
        return geo_average

    def payoff(self, x_arg: tf.Tensor, param: tf.Tensor):
        r"""
        The payoff function of multidimensional Geomatric Asian option is \max{A_T - K, 0} where:
          A_T is geometric average on the path of the asset at time T.
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
        k = tf.expand_dims(param[:, :, 0, -1], axis=-1)  # (B, M, 1)
        K = k * self.config.x_init
        dt = self.config.dt
        T = self.config.T
        x = x_arg[..., : self.config.dim]
        geo_mean = tf.reduce_mean(
            tf.exp(tf.reduce_sum(tf.math.log(x), axis=2) * dt / T),
            axis=-1,
            keepdims=True,
        )  # (B, M, 1)
        return tf.nn.relu(geo_mean - K)  # (B, M, 1)

    def exact_price(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor):
        r"""
        In this x has the dimension:
        (B, M, N d+d) since this ia a concat with markovian variable:
        G_t: [B, M, N, d] the geometric average of the underlying price
        ===============================================================
        The exact price formula is:
        \begin{equation*}
		u\left(t, S_t, A_t; (r, \sigma, \kappa)\right)=e^{-\tilde{r}(T-t)}\left(A_t^{t / T} S_t^{1-t / T} 
        e^{\hat{\mu}+\hat{\sigma}^2 / 2} N\left(d_1\right)-\kappa S_0 N\left(d_2\right)\right),
	    \end{equation*}
	    where
	    \begin{align}
		A_t & =\exp \left\{\frac{1}{t} \int_0^t \log S_s d s\right\}  , \quad \hat{\mu}  =\left(\tilde{r}-\frac{\tilde{\sigma}^2}{2}\right) \frac{1}{2 T}(T-t)^2  , 
        \quad \hat{\sigma}=\frac{\tilde{\sigma}}{T} \sqrt{\frac{1}{3}(T-t)^3}, \\
		d_2 & =\frac{(t / T) \log A_t+(1-t / T) \log S_t+\hat{\mu}-\log K}{\tilde{\sigma}}  , 
        \quad d_1 =d_2+\hat{\sigma} .
	    \end{align}

        t: [B, M, N, 1]
        x: [B, M, N, d]
        u_hat: [B, M, N, k]
        return: [B, M, N, 1]
        """
        T = self.config.T
        r = tf.expand_dims(u_hat[:, :, :, 0], -1)
        vol = tf.expand_dims(u_hat[:, :, :, 1], -1)
        k = tf.expand_dims(u_hat[:, :, :, 2], -1)
        K = k * self.config.x_init
        y_t = x[..., : self.config.dim]
        G_t = x[..., self.config.dim :]
        mu_bar = (r - vol**2 / 2) * (T - t) ** 2 / 2 / T
        vol_bar = vol / T * tf.math.sqrt((T - t) ** 3 / 3)
        d_2 = (
            t / T * tf.math.log(G_t)
            + (1 - t / T) * tf.math.log(y_t)
            + mu_bar
            - tf.math.log(K)
        ) / vol_bar
        d_1 = d_2 + vol_bar
        A = tf.exp(-r * (T - t)) * (
            G_t ** (t / T)
            * y_t ** (1 - t / T)
            * tf.exp(mu_bar + vol_bar**2 / 2)
            * dist.cdf(d_1)
            - K * dist.cdf(d_2)
        )
        return A  # [B, M, N, 1]


class InterestRateSwap(BaseOption):
    """
    This class implement the payoff and exact price of swap and swaption.
    For testing issue, we just implement the simpliest version,
    where swap just happen twice for three years.
    """

    def __init__(self, config):
        super(InterestRateSwap, self).__init__(config)
        self.strike_range = (
            self.config.strike_range
        )  # the distribution of the fixed coupon rate
        self.leg_dates = (
            self.config.leg_dates
        )  # a list with first element $T_0$ as the first
        self.fix_rate = 0.01
        self.sde = HullWhiteModel(config)
        self.notional = 1.0
        self.epsilon = 1e-3

    @property
    def delta_t(self):
        r"""
        This gives the \Delta T
        """
        return self.leg_dates[1] - self.leg_dates[0]

    def reset_dates(self, leg_dates: list):
        self.leg_dates = leg_dates

    @property
    def float_start_dates(self):
        """
        this gives [T_0, ..., T_{M-1}]
        """
        return self.leg_dates[:-1]

    @property
    def float_end_dates(self):
        """
        this gives [T_1, ..., T_{M}]
        """
        return self.leg_dates[1:]

    def fix_legs(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        """
        return a tensor containing the leg value at given time and state for all maturities in the list
        t: [B, M, 1]
        x: [B, M, 1]
        u_hat: [B, M, k]

        denote len(self.leg_dates) = L
        """
        # fix_rate = tf.expand_dims(u_hat[:, :, -1], -1) # (B, M, 1)
        p_list = [
            self.fix_rate * self.sde.zcp_value(t, x, u_hat, end_date) * self.delta_t
            for end_date in self.float_end_dates
        ]  # [[B, M, 1], [B, M, 1], ..., [B, M, 1]] (L-1 times)
        fix_legs = tf.concat(p_list, axis=-1)  # [B, M, L-1]
        return fix_legs

    def float_legs(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        """
        return a tensor containing the leg value at given time and state for all maturities in the list
        """
        p_list = [
            self.one_float_leg(t, x, u_hat, end_date)
            for end_date in self.float_end_dates
        ]  # [[B, M, 1], [B, M, 1], ..., [B, M, 1]] (L-1 times)
        float_legs = tf.concat(p_list, axis=-1)
        return float_legs  # [B, M, L-1]

    def one_float_leg(
        self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor, terminal_date: tf.Tensor
    ) -> tf.Tensor:
        """
        return P(t, T_{i-1}) - P(t, T_{i})
        T_{i-1} + \Delta T = T_{i}
        T_{i-1}: float start date
        T_{i}: float end date
        """
        float_leg = self.sde.zcp_value(
            t, x, u_hat, terminal_date - self.delta_t
        ) - self.sde.zcp_value(t, x, u_hat, terminal_date)
        return tf.where(
            t > terminal_date - self.delta_t + self.epsilon, 0.0, float_leg
        )  # [B, M, 1]

    def swap_value(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        """
        t: (B, M, 1)
        x: (B, M, d), d=1 usually
        u_hat: [B, M, k]
        """
        float_legs = self.float_legs(t, x, u_hat)  # (B, M, T)
        fix_legs = self.fix_legs(t, x, u_hat)  # (B, M, T)
        return self.notional * tf.reduce_sum(
            fix_legs - float_legs, axis=-1, keepdims=True
        )  # (B, M, 1)

    def payoff_inter(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        r"""
        This gives the value of a swaption which swap for 2 times at expiry date.
        It is written as the following form:
         v = 1 - K * P(1, 2) - (1 + K) * P(1, 3)
        x: [B, M, 1] must corresponded to the time t_s
        u_hat: [B, M, 1]
        return the value v: [B, M, 1]
        """
        p_12 = self.zcp(1.0, x[:, :, -1, :], u_hat[:, :, -1, :], 2.0)
        p_13 = self.zcp(1.0, x[:, :, -1, :], u_hat[:, :, -1, :], 3.0)
        return 1.0 - self.fix_rate * p_12 - (1.0 + self.fix_rate) * p_13

    def payoff_at_maturity(
        self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor
    ) -> tf.Tensor:
        r"""
        This gives the value of a swaption which swap for 1 times at expiry date.
        It is written as the following form:
         v = 1 - (1 + K) * P(2, 3)
        x: [B, M, 1] must corresponded to the time t_s
        u_hat: [B, M, 1]
        return the value v: [B, M, 1]
        """
        p_23 = self.zcp(
            self.config.leg_dates[1],
            x[:, :, -1, :],
            u_hat[:, :, -1, :],
            self.config.leg_dates[-1],
        )
        return 1.0 - (1.0 + self.fix_rate) * p_23

    def zcp(
        self, t_s: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor, t_e: tf.Tensor
    ) -> tf.Tensor:
        r"""
        zcp is the abbreviation of zero_coupon_bond. It is written as the following form:
         P(t_s, t_e) = A(t_s, t_e) * \exp{-B(t_s, t_e) * r_{t_s}} where function A and B are functions of t_e - t_s
         with \kappa, \theta, \sigma as parameters.
        t_e: a float scalar or a [B, M, 1] tensor, which represents the bond start time
        x: [B, M, 1] must corresponded to the time t_s
        u_hat: [B, M, 1]
        t_e: a float scalar or a [B, M, 1] tensor, which represents the bond end time
        return the value of zero coupon bond P(t_s, t_e): [B, M, 1]

        However this can also enter 4-D tensors, but the dimension of x and u_hat and t_s and t_e (if it is also tensors)
        must be consistent!
        """
        kappa = tf.expand_dims(u_hat[..., 0], axis=-1)
        theta = tf.expand_dims(u_hat[..., 1], axis=-1)
        sigma = tf.expand_dims(u_hat[..., 2], axis=-1)
        B = (1 - tf.exp(-kappa * (t_e - t_s))) / kappa
        A = tf.exp(
            (B - t_e + t_s) * (kappa**2 * theta - sigma**2 / 2) / kappa**2
            + (sigma * B) ** 2 / (4 * kappa)
        )
        p_se = A * tf.exp(-B * tf.reduce_sum(x, axis=-1, keepdims=True))
        return p_se

    def exact_price(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        r"""
        This yield the analytical value of the swap which will be swaped for twice:
         v_t = p(t, 1) - K * p(t, 2) - (1.0 + K) * p(t, 3)
        t: [B, M, N, 1]
        x: [B, M, N, 1]
        u_hat: [B, M, N, 4]
        return: v: [B, M, N, 1]
        """
        p_start = self.zcp(t, x, u_hat, self.config.leg_dates[0])
        p_mid = self.zcp(t, x, u_hat, self.config.leg_dates[1])
        p_end = self.zcp(t, x, u_hat, self.config.leg_dates[-1])
        v = p_start - self.fix_rate * p_mid - (1.0 + self.fix_rate) * p_end
        return v


class ZeroCouponBond(BaseOption):
    def __init__(self, config):
        super(ZeroCouponBond, self).__init__(config)
        self.strike_range = (
            self.config.strike_range
        )  # the distribution of the fixed coupon rate
        self.leg_dates = (
            self.config.leg_dates
        )  # a list with first element $T_0$ as the first
        self.terminal_date = self.config.T
        self.sde = HullWhiteModel(config)
        self.fix_rate = 0.05

    def payoff(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        r"""
        The payoff is 1 for the bond:
        """
        return 1.0

    def exact_price(self, t: tf.Tensor, x: tf.Tensor, u_hat: tf.Tensor) -> tf.Tensor:
        """
        t: [B, M, N, 1]
        x: [B, M, N, 1]
        u_hat: [B, M, N, 4]
        The price formula is:
        v = p(t, 1): [B, M, N, 1]
        """
        terminal_date = self.config.T
        kappa = tf.expand_dims(u_hat[..., 0], axis=-1)
        theta = tf.expand_dims(u_hat[..., 1], axis=-1)
        sigma = tf.expand_dims(u_hat[..., 2], axis=-1)
        B = (1 - tf.exp(-kappa * (terminal_date - t))) / kappa
        A = tf.exp(
            (B - terminal_date + t) * (kappa**2 * theta - sigma**2 / 2) / kappa**2
            + (sigma * B) ** 2 / (4 * kappa)
        )
        p_t1 = A * tf.exp(-B * tf.reduce_sum(x, axis=-1, keepdims=True))
        v = p_t1
        return v  # [B, M, N, 1]
