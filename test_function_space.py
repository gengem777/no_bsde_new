import tensorflow as tf
from function_space import PermutationInvariantLayer, DeepONetwithPI
import numpy as np


class TestPermutationInvariantLayer(tf.test.TestCase):
    """
    This class test the function of permutation invariant layers
    """
    def testPIlayer(self):
        pi_layer_1 = PermutationInvariantLayer(5)
        pi_layer_2 = PermutationInvariantLayer(3)
        x = np.array(
            [
                [
                    [
                        [
                            [1.0, 2.0, 3.0, 4.0],
                            [4.0, 3.0, 2.0, 1.0],
                            [3.0, 5.0, 4.0, 6.0],
                        ],
                        [
                            [5.0, 4.0, 3.0, 2.0],
                            [2.0, 3.0, 4.0, 5.0],
                            [3.0, 5.0, 4.0, 6.0],
                        ],
                    ]
                ]
            ]
        )
        x_pi = np.array(
            [
                [
                    [
                        [
                            [4.0, 3.0, 2.0, 1.0],
                            [3.0, 5.0, 4.0, 6.0],
                            [1.0, 2.0, 3.0, 4.0],
                        ],
                        [
                            [2.0, 3.0, 4.0, 5.0],
                            [3.0, 5.0, 4.0, 6.0],
                            [5.0, 4.0, 3.0, 2.0],
                        ],
                    ]
                ]
            ]
        )
        y = pi_layer_2(pi_layer_1(x))
        y_pi = pi_layer_2(pi_layer_1(x_pi))
        self.assertAllLessEqual(
            tf.abs(tf.reduce_sum(y, axis=-2) - tf.reduce_sum(y_pi, axis=-2)), 1e-6
        )


class TestDeepONetwithPI(tf.test.TestCase):
    """
    This class test the deep o net with part of trunk net being permutation invariant layers
    """
    def testnumberofparams(self):
        l = 1
        m = 6
        d = 3
        N = 10
        B = 2
        M = 2
        T = 2
        deeponet = DeepONetwithPI([3, 3], [3, 3], [m] * (l + 1), 10)
        assets = tf.random.normal([B, M, T, N * d])
        t = tf.random.uniform([B, M, T, 1])
        u_hat = tf.random.normal([B, M, T, 4])
        y = deeponet((t, assets, u_hat))

        def num_params_pi(m, d, l):
            return m * (d + 1) + m * (m + 1) * (l - 1) + (m + 1) * m

        num_actual = 0
        for v in deeponet.PI_layers.trainable_weights:
            num_actual += tf.size(v)
        self.assertEqual(num_params_pi(m, d, l), num_actual)

    def testPIproperty(self):
        l = 1
        m = 6
        d = 3
        N = 10
        B = 2
        M = 2
        T = 2
        deeponet = DeepONetwithPI([3, 3], [3, 3], [m] * (l + 1), 10)
        assets = tf.random.normal([B, M, T, N * d])
        t = tf.random.uniform([B, M, T, 1])
        u_hat = tf.random.normal([B, M, T, 4])
        y_ni = deeponet((t, assets, u_hat))
        assets_pi = tf.reverse(deeponet.reshape_state(assets), [3])
        assets_pi = tf.reshape(assets_pi, [B, M, T, N * d])
        y_pi = deeponet((t, assets_pi, u_hat))
        self.assertAllLessEqual(tf.reduce_sum(tf.abs(y_ni - y_pi)), 1e-7)


if __name__ == "__main__":
    tf.test.main()
