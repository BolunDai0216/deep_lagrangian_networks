import math
from pdb import set_trace

import numpy as np
import tensorflow as tf


class ReLUDer(tf.keras.Model):
    def __init__(self):
        super(ReLUDer, self).__init__()

    def call(self, x):
        der = tf.clip_by_value(x, 0, 1)
        der = tf.math.ceil(der)

        return der


class LinearDer(tf.keras.Model):
    def __init__(self):
        super(LinearDer, self).__init__()

    def call(self, x):
        der = tf.clip_by_value(x, 1, 1)

        return der


class SoftPlusDer(tf.keras.Model):
    def __init__(self, lower_clip_bound=-20.0, upper_clip_bound=20.0):
        super(SoftPlusDer, self).__init__()
        self.lower_clip_bound = lower_clip_bound
        self.upper_clip_bound = upper_clip_bound

    def call(self, x):
        cx = tf.clip_by_value(x, self.lower_clip_bound, self.upper_clip_bound)
        exp_x = tf.math.exp(cx)
        out = exp_x / (exp_x + 1)

        if tf.reduce_any(tf.math.is_nan(out)):
            print("SoftPlus Forward output is NaN.")
        return out


def Diag(x):

    return tf.linalg.diag(x)


class LowTri:
    def __init__(self, n_dof):
        self.n_dof = n_dof
        self.reverse_list = np.flip(np.arange(1, self.n_dof + 1)).tolist()

    def __call__(self, l):
        batch_size = l.shape[0]
        base = tf.zeros([batch_size, self.n_dof, self.n_dof])
        count = 0

        for i, size in enumerate(self.reverse_list):
            base += tf.linalg.diag(l[:, count : (count + size)], k=-i)
            count += size

        return base


class LowTri3D:
    def __init__(self, n_dof):
        self.n_dof = n_dof
        self.reverse_list = np.flip(np.arange(1, self.n_dof + 1)).tolist()

    def __call__(self, l):
        batch_size = l.shape[0]
        base = tf.zeros([batch_size, self.n_dof, self.n_dof, self.n_dof])
        count = 0

        for i, size in enumerate(self.reverse_list):
            base += tf.linalg.diag(l[:, :, count : (count + size)], k=-i)
            count += size

        return base


class LagrangianLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs, _b0=0.1, activation="relu"):
        super(LagrangianLayer, self).__init__()
        self.activation = activation
        self.num_outputs = num_outputs
        self._b0_init = _b0

        if self.activation == "relu":
            self.g = tf.keras.activations.relu
            self.g_prime = ReLUDer()
        if self.activation == "linear":
            self.g = tf.keras.activations.linear
            self.g_prime = LinearDer()
        if self.activation == "softplus":
            self.g = tf.keras.activations.softplus
            self.g_prime = SoftPlusDer()

    def build(self, input_shape):
        weight_initializer = tf.keras.initializers.GlorotNormal(seed=100)
        bias_initializer = tf.keras.initializers.Constant(value=self._b0_init)

        self.kernel = self.add_weight(
            "kernel",
            shape=[int(input_shape[-1]), int(self.num_outputs)],
            initializer=weight_initializer,
        )
        self.bias = self.add_weight(
            "bias",
            shape=[int(self.num_outputs)],
            initializer=bias_initializer,
        )

    def call(self, inputs, prev_der):
        # For the first layer prev_der is the identity
        a = tf.matmul(inputs, self.kernel) + self.bias
        # [BATCH_SIZE, OUTPUT_SIZE]
        output = self.g(a)
        # [BATCH_SIZE, OUTPUT_SIZE, INPUT_SIZE]
        output_der = tf.linalg.diag(self.g_prime(a)) @ tf.transpose(self.kernel)
        # ∇h_i = diag(g'(a))W∇h_{i-1}
        output_der = output_der @ prev_der
        return output, output_der


class DeepLagrangianNetwork(tf.keras.Model):
    def __init__(self, n_dof, **kwargs):
        super(DeepLagrangianNetwork, self).__init__()

        # Read optional arguments:
        self.n_width = kwargs.get("n_width", 128)
        self.n_hidden = kwargs.get("n_depth", 1)
        self._b0 = kwargs.get("b_init", 0.1)
        self._b0_diag = kwargs.get("b_diag_init", 0.1)

        self._w_init = kwargs.get("w_init", "xavier_normal")
        self._g_hidden = kwargs.get("g_hidden", np.sqrt(2.0))
        self._g_output = kwargs.get("g_hidden", 0.125)
        self._p_sparse = kwargs.get("p_sparse", 0.2)
        self._epsilon = kwargs.get("diagonal_epsilon", 1.0e-5)
        nonlinearity = kwargs.get("activation", "relu")

        # Calculate size of lo output
        lower_size = (n_dof ** 2 - n_dof) / 2

        # Define layers
        self.input_layer = LagrangianLayer(
            self.n_width, _b0=self._b0, activation=nonlinearity
        )
        self.hidden_layer1 = LagrangianLayer(
            self.n_width, _b0=self._b0, activation=nonlinearity
        )
        self.hidden_layer2 = LagrangianLayer(
            self.n_width, _b0=self._b0, activation=nonlinearity
        )
        self.net_g = LagrangianLayer(1, _b0=self._b0, activation="linear")
        self.net_ld = LagrangianLayer(n_dof, _b0=self._b0, activation="relu")
        self.net_lo = LagrangianLayer(lower_size, _b0=self._b0, activation="linear")

        # Define helper functions
        self.lowtri = LowTri(n_dof)
        self.lowtri3D = LowTri3D(n_dof)

    def call(self, state, velocity, acceleration, training=True):
        input_shape = state.shape[-1]  # [batch_size, input_size]
        batch_size = state.shape[0]
        # Initial Identity matrix
        init_eye = tf.eye(input_shape, batch_shape=[batch_size])

        # Forward model
        y1, der1 = self.input_layer(state, init_eye)
        y2, der2 = self.hidden_layer1(y1, der1)
        y3, der3 = self.hidden_layer1(y2, der2)

        # Final outputs
        V, derV = self.net_g(y3, der3)
        l_diag, der_l_diag = self.net_ld(y3, der3)
        l_lower, der_l_lower = self.net_lo(y3, der3)

        # Assemble l and der_l
        l = tf.concat([l_diag, l_lower], 1)
        der_l = tf.concat([der_l_diag, der_l_lower], 1)

        # Compute M
        L = self.lowtri(l)
        LT = tf.transpose(L, perm=[0, 2, 1])
        M = L @ LT + init_eye * self._epsilon

        # Compute dM/dt
        dldt = tf.squeeze(der_l @ velocity[:, :, tf.newaxis])
        dLdt = self.lowtri(dldt)
        dLdtT = tf.transpose(dLdt, perm=[0, 2, 1])
        dMdt = L @ dLdtT + dLdt @ LT

        # Compute dM/dq
        der_lT = tf.transpose(der_l, perm=[0, 2, 1])
        dLdq = self.lowtri3D(der_lT)
        dLdqT = tf.transpose(dLdq, perm=[0, 1, 3, 2])
        dMdq = dLdq @ LT[:, tf.newaxis, :, :] + L[:, tf.newaxis, :, :] @ dLdqT

        # Compute Coriolis and Centrifugal forces
        dMdt_dq = dMdt @ velocity[:, :, tf.newaxis]
        quad_dq = (
            velocity[:, tf.newaxis, tf.newaxis, :]
            @ dMdq
            @ velocity[:, tf.newaxis, :, tf.newaxis]
        )
        C = tf.squeeze(dMdt_dq) - 0.5 * tf.squeeze(quad_dq)

        # Compute Gravitational forces
        G = tf.squeeze(derV)

        # Computer tau_pred
        Mddq = tf.squeeze(M @ acceleration[:, :, tf.newaxis])
        tau_pred = Mddq + C + G

        return tau_pred, M, C, G


def main():
    hyper = {
        "n_width": 64,
        "n_depth": 2,
        "diagonal_epsilon": 0.01,
        "activation": "softplus",
        "b_init": 1.0e-4,
        "b_diag_init": 0.001,
        "w_init": "xavier_normal",
        "gain_hidden": np.sqrt(2.0),
        "gain_output": 0.1,
        "n_minibatch": 512,
        "learning_rate": 5.0e-04,
        "weight_decay": 1.0e-5,
        "max_epoch": 10000,
    }

    n_dof = 2
    delan_model = DeepLagrangianNetwork(n_dof, **hyper)

    state = tf.ones([64, n_dof])
    velocity = tf.ones([64, n_dof])
    acceleration = tf.ones([64, n_dof])

    delan_model(state, velocity, acceleration)


if __name__ == "__main__":
    main()
