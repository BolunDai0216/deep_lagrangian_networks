import tensorflow as tf
import tensorflow_probability as tfp
from pdb import set_trace


def ReLUDer(x):
    der = tf.clip_by_value(x, 0, 1)
    der = tf.math.ceil(der)

    return der


def LinearDer(x):
    der = tf.clip_by_value(x, 1, 1)

    return der


def Diag(x):

    return tf.linalg.diag(x)


def SoftPlusDer(x):
    cx = tf.clip_by_value(x, -20.0, 20.0)
    exp_x = tf.math.exp(cx)
    out = exp_x / (exp_x + 1)

    if tf.reduce_any(tf.math.is_nan(out)):
        print("SoftPlus Forward output is NaN.")
    return out


def LowTri(x):
    low_tri = tfp.math.fill_triangular(x)

    sh = low_tri.get_shape().as_list()
    sh[-1] = 1

    low_tri = tf.concat([low_tri, tf.zeros(sh)], axis=-1)

    sh = low_tri.get_shape().as_list()
    sh[-2] = 1

    low_tri = tf.concat([tf.zeros(sh), low_tri], axis=-2)

    return low_tri


class LagrangianLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs, activation="relu"):
        super(LagrangianLayer, self).__init__()
        self.activation = activation
        self.num_outputs = num_outputs

        if self.activation == "relu":
            self.g = tf.keras.activations.relu
            self.g_prime = ReLUDer
        if self.activation == "linear":
            self.g = tf.keras.activations.linear
            self.g_prime = LinearDer
        if self.activation == "softplus":
            self.g = tf.keras.activations.softplus
            self.g_prime = SoftPlusDer

    def build(self, input_shape):
        self.kernel = self.add_weight(
            "kernel",
            shape=[int(input_shape[-1]), int(self.num_outputs)],
            initializer=tf.keras.initializers.GlorotNormal(),
        )
        self.bias = self.add_weight(
            "bias",
            shape=[int(self.num_outputs)],
            initializer=tf.keras.initializers.Zeros(),
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


def main():
    x = tf.ones([1, 3])
    set_trace()
    out = SoftPlusDer(x)


if __name__ == "__main__":
    main()