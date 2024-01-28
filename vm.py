import numpy as np
import scipy as sp
from scipy import sparse
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import interactive
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import cm
import scipy.stats as stats
from statsmodels.stats.weightstats import DescrStatsW

L = 32.0
rho = 3.0
N = int(rho * L**2)
print(" N", N)

r0 = 1.0
deltat = 1.0
factor = 0.5
v0 = r0 / deltat * factor
iterations = 5000
eta_linspace = np.linspace(0, 1, 50)  # 0.15

load_save = True
create_new_save = not load_save
filename = "1000iterations_50etas"  # "5k_iterations_10etas"


def animate(i):
    global orient

    tree = cKDTree(pos, boxsize=[L, L])
    dist = tree.sparse_distance_matrix(tree, max_distance=r0, output_type="coo_matrix")

    # important 3 lines: we evaluate a quantity for every column j
    data = np.exp(orient[dist.col] * 1j)
    # construct  a new sparse marix with entries in the same places ij of the dist matrix
    neigh = sparse.coo_matrix((data, (dist.row, dist.col)), shape=dist.get_shape())
    # and sum along the columns (sum over j)
    S = np.squeeze(np.asarray(neigh.tocsr().sum(axis=1)))

    orient = np.angle(S) + eta * np.random.uniform(-np.pi, np.pi, size=N)

    cos, sin = np.cos(orient), np.sin(orient)
    pos[:, 0] += cos * v0
    pos[:, 1] += sin * v0

    pos[pos > L] -= L
    pos[pos < 0] += L

    vx = v0 * cos
    vy = v0 * sin

    total_speed = np.sqrt((np.sum(vx)) ** 2 + (np.sum(vy)) ** 2) / N

    print(total_speed)
    print(i)
    # print("eta: " + str(eta))
    total_speeds.append(total_speed)

    qv.set_offsets(pos)
    qv.set_UVC(cos, sin, orient)

    return (qv,)


total_speed_dict = {}

pos_df = pd.DataFrame(columns=["pos"])


def Count_birds_in_box(pos):
    N_df = pd.DataFrame(columns=["average", "std"])
    max_std_N = 0
    leniency_strength = 4
    for box_size in np.linspace(0, L, 100):
        N_list = []
        for i in range(50):
            box_middle = np.random.uniform(0.5 * box_size, L - 0.5 * box_size, 2)

            box_birds = np.where(
                (abs(pos[:, 0] - box_middle[0]) < 0.5 * box_size)
                & (abs(pos[:, 1] - box_middle[1]) < 0.5 * box_size),
                True,
                False,
            )
            # Store the number of birds in the box in the list.
            N_list.append(sum(box_birds))

        average_N = np.mean(N_list)
        std_N = np.std(N_list)

        # Store the list of number of birds in the box in the dictionary for every box size.\
        N_df.at[box_size, "average"] = average_N
        N_df.at[box_size, "std"] = std_N

        if std_N < 0.8 * max_std_N:
            leniency_strength -= 1
            if leniency_strength < 0:
                break
        elif std_N > max_std_N:
            max_std_N = std_N

    return N_df


def exponential(x, a, b, c):
    return a * x**b + c


def Fit_curve(fit_function, x_data, y_data):
    # Use curve_fit to fit the data to the function
    params, covariance = curve_fit(fit_function, x_data, y_data)

    # Extract the fitted parameters
    a_fit, b_fit, c_fit = params
    a_err, b_err, c_err = np.sqrt(np.diag(covariance))

    fit_plot = fit_function(np.array(x_data), a_fit, b_fit, c_fit)
    error_top = fit_function(
        np.array(x_data), a_fit - a_err, b_fit - b_err, c_fit - c_err
    )
    error_bottom = fit_function(
        np.array(x_data), a_fit + a_err, b_fit + b_err, c_fit + c_err
    )

    # Generate y values for the fitted function
    return pd.DataFrame(
        {"fit": fit_plot, "errortop": error_top, "errorbottom": error_bottom}
    ), [b_fit, b_err]


def Plot_N_vs_std():
    colours = cm.rainbow(np.linspace(0, 1, len(eta_linspace)))

    z_exponent_df = pd.DataFrame(columns=["z_exponent", "z_exponent_error"])

    plt.figure(figsize=(12, 6))
    plt.title("Fit to the data for different values of eta")
    plt.xlabel("Average birds in box")
    plt.ylabel("Standard deviation")
    for eta, colour, i in zip(eta_linspace, colours, range(len(eta_linspace))):
        N_df = Count_birds_in_box(pos_df.loc[eta, "pos"])

        eta = round(eta, 2)

        N_df.sort_values(by=["average"], ascending=True, inplace=True)

        fit_data_df, z_exponent = Fit_curve(
            exponential, N_df["average"].tolist(), N_df["std"].tolist()
        )

        z_exponent_df.at[eta, "z_exponent"] = z_exponent[0]
        z_exponent_df.at[eta, "z_exponent_error"] = z_exponent[1]

        plt.scatter(
            N_df["average"],
            N_df["std"],
            color=colour,
            label=f"eta: {eta}; z: {round(z_exponent[0], 2)} +- {round(z_exponent[1], 2)}",
            alpha=0.5,
        )
        plt.plot(
            N_df["average"],
            fit_data_df["fit"],
            color=colour,
            # label=f"Exponential fit eta: {eta}",
        )
        plt.plot(
            N_df["average"],
            fit_data_df["errortop"],
            color=colour,
            alpha=0.5,
            # linewidth=0.5,
            linestyle="dotted",
            # label=f"Standard deviation eta: {eta}",
        )
        plt.plot(
            N_df["average"],
            fit_data_df["errorbottom"],
            color=colour,
            alpha=0.5,
            # linewidth=0.5,
            linestyle="dotted",
        )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")  # , borderaxespad=0.0)
    plt.xlim(left=1)
    plt.ylim(bottom=1)
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    # plt.savefig(f"fitdifferentetas.png", dpi=300)
    plt.show(block=False)
    plt.close()

    return z_exponent_df


if load_save:
    pos_df = pd.read_pickle(f"pos_df_{filename}.pkl")
elif create_new_save:
    for eta in eta_linspace:
        pos = np.random.uniform(0, L, size=(N, 2))
        orient = np.random.uniform(-np.pi, np.pi, size=N)

        fig, ax = plt.subplots(figsize=(6, 6))

        qv = ax.quiver(
            pos[:, 0],
            pos[:, 1],
            np.cos(orient),
            np.sin(orient),
            orient,
            clim=[-np.pi, np.pi],
        )

        total_speeds = []
        # myAnimation = animation.FuncAnimation(
        #     fig, animate, np.arange(1, iterations), interval=1, blit=True, repeat=False
        # )
        myAnimation = animation.FuncAnimation(
            fig, animate, iterations, interval=1, blit=True, repeat=False
        )

        myAnimation.save(
            "vm_noise_" + str(eta) + "rho" + str(rho) + ".gif", writer="pillow"
        )
        plt.show(block=False)
        plt.close()
        total_speed_dict[eta] = total_speeds

        pos_df.at[eta, "pos"] = pos

    pos_df.to_pickle(f"pos_df_{filename}.pkl")


if __name__ == "__main__":
    z_exponent_df = Plot_N_vs_std()

    weighted_z = DescrStatsW(
        z_exponent_df["z_exponent"],
        weights=1 / z_exponent_df["z_exponent_error"] ** 2,
        ddof=0,
    )

    plt.figure()
    plt.title("z exponent vs eta")
    plt.xlabel("eta")
    plt.ylabel("z exponent")
    plt.errorbar(
        z_exponent_df.index.tolist(),
        z_exponent_df["z_exponent"],
        yerr=z_exponent_df["z_exponent_error"],
        fmt="o",
    )
    plt.axhline(y=weighted_z.mean, color="r", linestyle="-")
    plt.axhline(y=weighted_z.mean + weighted_z.std, color="r", linestyle="--", alpha=0.5)
    plt.axhline(y=weighted_z.mean - weighted_z.std, color="r", linestyle="--", alpha=0.5)
    # plt.savefig(f"z_exponent_vs_eta_weighted average.png", dpi=300)
    plt.show()
