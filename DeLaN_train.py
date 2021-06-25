from datetime import datetime
from pdb import set_trace
from time import time

import numpy as np
import tensorflow as tf
import torch

from deep_lagrangian_networks.replay_memory import PyTorchReplayMemory
from deep_lagrangian_networks.utils import init_env, load_dataset
from DeLaN_tensorflow import DeepLagrangianNetwork
from DeLaN_utils import plot_test


class Train:
    def __init__(self):
        # Read the dataset:
        n_dof = 2
        cuda = 1
        train_data, test_data, self.divider = load_dataset()
        (
            self.train_labels,
            self.train_qp,
            self.train_qv,
            self.train_qa,
            self.train_tau,
        ) = train_data
        (
            self.test_labels,
            self.test_qp,
            self.test_qv,
            self.test_qa,
            self.test_tau,
            self.test_m,
            self.test_c,
            self.test_g,
        ) = test_data

        self.hyper = {
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

        self.model = DeepLagrangianNetwork(n_dof, **self.hyper)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.hyper["learning_rate"], amsgrad=True
        )

        # Generate Replay Memory:
        mem_dim = ((n_dof,), (n_dof,), (n_dof,), (n_dof,))
        self.mem = PyTorchReplayMemory(
            self.train_qp.shape[0], self.hyper["n_minibatch"], mem_dim, cuda
        )
        self.mem.add_samples(
            [self.train_qp, self.train_qv, self.train_qa, self.train_tau]
        )

        # Information for saving model
        self.checkpoint = tf.train.Checkpoint(net=self.model)
        self.stamp = datetime.fromtimestamp(time()).strftime("%Y%m%d-%H%M%S")

    def train(self):
        # Print training information
        print("\n\n################################################")
        print("Characters:")
        print("   Test Characters = {0}".format(self.test_labels))
        print("  Train Characters = {0}".format(self.train_labels))
        print("# Training Samples = {0:05d}".format(int(self.train_qp.shape[0])))
        print("")

        # Training Parameters:
        print("\n################################################")
        print("Training Deep Lagrangian Networks (DeLaN):")

        for epoch_i in range(self.hyper["max_epoch"]):
            l_mem, n_batches = 0.0, 0.0

            for q, qd, qdd, tau in self.mem:
                q_tf, qd_tf, qdd_tf, tau_tf = self.convert_to_tf(q, qd, qdd, tau)
                loss = self.opt(q_tf, qd_tf, qdd_tf, tau_tf)
                l_mem += loss
                n_batches += 1

            l_mem /= float(n_batches)

            # if epoch_i == 1 or np.mod(epoch_i, 50) == 0:
            print("*******")
            print("Epoch {0:05d}: ".format(epoch_i), end=" ")
            print("Loss = {0:.3e}".format(l_mem), end=", ")

            if np.mod(epoch_i, 1000) == 0:
                filename = "trained_models/{}/tf_model_{}".format(self.stamp, epoch_i)
                self.checkpoint.save("{}".format(filename))

    def test(self, filename):
        # Load pre-trained model
        self.checkpoint.restore(filename)

        # Get test data
        q = tf.cast(self.test_qp, tf.float32)
        dq = tf.cast(self.test_qv, tf.float32)
        ddq = tf.cast(self.test_qa, tf.float32)

        # Calculate torque using test data
        delan_tau, delan_M, delan_C, delan_G = self.model(q, dq, ddq)
        delan_Mddq = tf.squeeze(delan_M @ ddq[:, :, tf.newaxis])

        # Get test error
        mean_coeff = 1.0 / float(self.test_qp.shape[0])
        err_g = mean_coeff * np.sum((delan_G - self.test_g) ** 2)
        err_m = mean_coeff * np.sum((delan_Mddq - self.test_m) ** 2)
        err_c = mean_coeff * np.sum((delan_C - self.test_c) ** 2)
        err_tau = mean_coeff * np.sum((delan_tau - self.test_tau) ** 2)

        print("\nPerformance:")
        print("                Torque MSE = {0:.3e}".format(err_tau))
        print("              Inertial MSE = {0:.3e}".format(err_m))
        print("Coriolis & Centrifugal MSE = {0:.3e}".format(err_c))
        print("         Gravitational MSE = {0:.3e}".format(err_m))

        plot_test(
            delan_tau,
            delan_Mddq,
            delan_C,
            delan_G,
            self.test_tau,
            self.test_m,
            self.test_c,
            self.test_g,
            self.divider,
            self.test_labels,
        )

    @tf.function
    def opt(self, q_tf, qd_tf, qdd_tf, tau_tf):
        with tf.GradientTape() as tape:
            tau_hat, _, _, _ = self.model(q_tf, qd_tf, qdd_tf)
            err = tf.math.reduce_sum(tf.square(tau_hat - tau_tf), axis=1)
            loss = tf.reduce_mean(err)

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return loss

    def convert_to_tf(self, q, qd, qdd, tau):
        q_tf = tf.convert_to_tensor(q.cpu().numpy())
        qd_tf = tf.convert_to_tensor(qd.cpu().numpy())
        qdd_tf = tf.convert_to_tensor(qdd.cpu().numpy())
        tau_tf = tf.convert_to_tensor(tau.cpu().numpy())

        return q_tf, qd_tf, qdd_tf, tau_tf


def main():
    train = Train()
    # train.train()
    train.test("trained_models/20210623-223506/tf_model_9000-10")


if __name__ == "__main__":
    main()
