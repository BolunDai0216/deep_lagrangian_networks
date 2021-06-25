import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np


def plot_test(
    delan_tau,
    delan_m,
    delan_c,
    delan_g,
    test_tau,
    test_m,
    test_c,
    test_g,
    divider,
    test_labels,
):
    try:
        mp.rc("text", usetex=True)
        mp.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]

    except ImportError:
        pass

    # Alpha of the graphs:
    plot_alpha = 0.8

    # Plot the performance:
    y_t_low = np.clip(
        1.2 * np.min(np.vstack((test_tau, delan_tau)), axis=0), -np.inf, -0.01
    )
    y_t_max = np.clip(
        1.5 * np.max(np.vstack((test_tau, delan_tau)), axis=0), 0.01, np.inf
    )

    y_m_low = np.clip(
        1.2 * np.min(np.vstack((test_m, delan_m)), axis=0), -np.inf, -0.01
    )
    y_m_max = np.clip(1.2 * np.max(np.vstack((test_m, delan_m)), axis=0), 0.01, np.inf)

    y_c_low = np.clip(
        1.2 * np.min(np.vstack((test_c, delan_m)), axis=0), -np.inf, -0.01
    )
    y_c_max = np.clip(1.2 * np.max(np.vstack((test_c, delan_c)), axis=0), 0.01, np.inf)

    y_g_low = np.clip(
        1.2 * np.min(np.vstack((test_g, delan_g)), axis=0), -np.inf, -0.01
    )
    y_g_max = np.clip(1.2 * np.max(np.vstack((test_g, delan_g)), axis=0), 0.01, np.inf)

    plt.rc("text", usetex=True)
    color_i = ["r", "b", "g", "k"]

    ticks = np.array(divider)
    ticks = (ticks[:-1] + ticks[1:]) / 2

    fig = plt.figure(figsize=(24.0 / 1.54, 8.0 / 1.54), dpi=100)
    fig.subplots_adjust(
        left=0.08, bottom=0.12, right=0.98, top=0.95, wspace=0.3, hspace=0.2
    )
    fig.canvas.set_window_title("Seed = {0}".format(42))

    legend = [
        mp.patches.Patch(color=color_i[0], label="DeLaN"),
        mp.patches.Patch(color="k", label="Ground Truth"),
    ]

    # Plot Torque
    ax0 = fig.add_subplot(2, 4, 1)
    ax0.set_title(r"$\boldsymbol{\tau}$")
    ax0.text(
        s=r"\textbf{Joint 0}",
        x=-0.35,
        y=0.5,
        fontsize=12,
        fontweight="bold",
        rotation=90,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax0.transAxes,
    )
    ax0.set_ylabel("Torque [Nm]")
    ax0.get_yaxis().set_label_coords(-0.2, 0.5)
    ax0.set_ylim(y_t_low[0], y_t_max[0])
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(test_labels)
    ax0.vlines(divider, y_t_low[0], y_t_max[0], linestyles="--", lw=0.5, alpha=1.0)
    ax0.set_xlim(divider[0], divider[-1])

    ax1 = fig.add_subplot(2, 4, 5)
    ax1.text(
        s=r"\textbf{Joint 1}",
        x=-0.35,
        y=0.5,
        fontsize=12,
        fontweight="bold",
        rotation=90,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax1.transAxes,
    )

    ax1.text(
        s=r"\textbf{(a)}",
        x=0.5,
        y=-0.25,
        fontsize=12,
        fontweight="bold",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax1.transAxes,
    )

    ax1.set_ylabel("Torque [Nm]")
    ax1.get_yaxis().set_label_coords(-0.2, 0.5)
    ax1.set_ylim(y_t_low[1], y_t_max[1])
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(test_labels)
    ax1.vlines(divider, y_t_low[1], y_t_max[1], linestyles="--", lw=0.5, alpha=1.0)
    ax1.set_xlim(divider[0], divider[-1])

    ax0.legend(
        handles=legend,
        bbox_to_anchor=(0.0, 1.0),
        loc="upper left",
        ncol=1,
        framealpha=1.0,
    )

    # Plot Ground Truth Torque:
    ax0.plot(test_tau[:, 0], color="k")
    ax1.plot(test_tau[:, 1], color="k")

    # Plot DeLaN Torque:
    ax0.plot(delan_tau[:, 0], color=color_i[0], alpha=plot_alpha)
    ax1.plot(delan_tau[:, 1], color=color_i[0], alpha=plot_alpha)

    # Plot Mass Torque
    ax0 = fig.add_subplot(2, 4, 2)
    ax0.set_title(r"$\displaystyle\mathbf{H}(\mathbf{q}) \ddot{\mathbf{q}}$")
    ax0.set_ylabel("Torque [Nm]")
    ax0.set_ylim(y_m_low[0], y_m_max[0])
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(test_labels)
    ax0.vlines(divider, y_m_low[0], y_m_max[0], linestyles="--", lw=0.5, alpha=1.0)
    ax0.set_xlim(divider[0], divider[-1])

    ax1 = fig.add_subplot(2, 4, 6)
    ax1.text(
        s=r"\textbf{(b)}",
        x=0.5,
        y=-0.25,
        fontsize=12,
        fontweight="bold",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax1.transAxes,
    )

    ax1.set_ylabel("Torque [Nm]")
    ax1.set_ylim(y_m_low[1], y_m_max[1])
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(test_labels)
    ax1.vlines(divider, y_m_low[1], y_m_max[1], linestyles="--", lw=0.5, alpha=1.0)
    ax1.set_xlim(divider[0], divider[-1])

    # Plot Ground Truth Inertial Torque:
    ax0.plot(test_m[:, 0], color="k")
    ax1.plot(test_m[:, 1], color="k")

    # Plot DeLaN Inertial Torque:
    ax0.plot(delan_m[:, 0], color=color_i[0], alpha=plot_alpha)
    ax1.plot(delan_m[:, 1], color=color_i[0], alpha=plot_alpha)

    # Plot Coriolis Torque
    ax0 = fig.add_subplot(2, 4, 3)
    ax0.set_title(r"$\displaystyle\mathbf{c}(\mathbf{q}, \dot{\mathbf{q}})$")
    ax0.set_ylabel("Torque [Nm]")
    ax0.set_ylim(y_c_low[0], y_c_max[0])
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(test_labels)
    ax0.vlines(divider, y_c_low[0], y_c_max[0], linestyles="--", lw=0.5, alpha=1.0)
    ax0.set_xlim(divider[0], divider[-1])

    ax1 = fig.add_subplot(2, 4, 7)
    ax1.text(
        s=r"\textbf{(c)}",
        x=0.5,
        y=-0.25,
        fontsize=12,
        fontweight="bold",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax1.transAxes,
    )

    ax1.set_ylabel("Torque [Nm]")
    ax1.set_ylim(y_c_low[1], y_c_max[1])
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(test_labels)
    ax1.vlines(divider, y_c_low[1], y_c_max[1], linestyles="--", lw=0.5, alpha=1.0)
    ax1.set_xlim(divider[0], divider[-1])

    # Plot Ground Truth Coriolis & Centrifugal Torque:
    ax0.plot(test_c[:, 0], color="k")
    ax1.plot(test_c[:, 1], color="k")

    # Plot DeLaN Coriolis & Centrifugal Torque:
    ax0.plot(delan_c[:, 0], color=color_i[0], alpha=plot_alpha)
    ax1.plot(delan_c[:, 1], color=color_i[0], alpha=plot_alpha)

    # Plot Gravity
    ax0 = fig.add_subplot(2, 4, 4)
    ax0.set_title(r"$\displaystyle\mathbf{g}(\mathbf{q})$")
    ax0.set_ylabel("Torque [Nm]")
    ax0.set_ylim(y_g_low[0], y_g_max[0])
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(test_labels)
    ax0.vlines(divider, y_g_low[0], y_g_max[0], linestyles="--", lw=0.5, alpha=1.0)
    ax0.set_xlim(divider[0], divider[-1])

    ax1 = fig.add_subplot(2, 4, 8)
    ax1.text(
        s=r"\textbf{(d)}",
        x=0.5,
        y=-0.25,
        fontsize=12,
        fontweight="bold",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax1.transAxes,
    )

    ax1.set_ylabel("Torque [Nm]")
    ax1.set_ylim(y_g_low[1], y_g_max[1])
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(test_labels)
    ax1.vlines(divider, y_g_low[1], y_g_max[1], linestyles="--", lw=0.5, alpha=1.0)
    ax1.set_xlim(divider[0], divider[-1])

    # Plot Ground Truth Gravity Torque:
    ax0.plot(test_g[:, 0], color="k")
    ax1.plot(test_g[:, 1], color="k")

    # Plot DeLaN Gravity Torque:
    ax0.plot(delan_g[:, 0], color=color_i[0], alpha=plot_alpha)
    ax1.plot(delan_g[:, 1], color=color_i[0], alpha=plot_alpha)

    fig.savefig("DeLaN_Performance.png", format="png")
