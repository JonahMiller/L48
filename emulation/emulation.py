import argparse

import GPy
import matplotlib.pyplot as plt
import numpy as np
from emukit.core import ContinuousParameter, DiscreteParameter, ParameterSpace
from emukit.core.initial_designs.latin_design import LatinDesign
from emukit.experimental_design.acquisitions import (
    IntegratedVarianceReduction,
    ModelVariance,
)
from emukit.experimental_design.experimental_design_loop import ExperimentalDesignLoop
from emukit.model_wrappers import GPyModelWrapper
from emukit.multi_fidelity.convert_lists_to_array import (
    convert_x_list_to_array,
)

import sys
sys.path.append("..")
from low_fidelity.main import HyperParams, simulate
from low_fidelity.lv_param_est import estimate

from FillBetween3d import fill_between_3d


length_scales = {
    # "PRED_REPRODUCTION_CHANCE": 0.1,
    # "PRED_REPRODUCTION_THRESHOLD": 500,
    "PRED_ENERGY_FROM_PREY": 2,
    # "STEPS": 50,
    "NUM_FOOD": 10,
}
space = ParameterSpace(
    [
        # ContinuousParameter("PRED_REPRODUCTION_CHANCE", 0, 1),
        # DiscreteParameter("PRED_REPRODUCTION_THRESHOLD", range(10, 5001)),
        DiscreteParameter("PRED_ENERGY_FROM_PREY", range(40, 61)),
        # DiscreteParameter("STEPS", range(0, 1001)),
        DiscreteParameter("NUM_FOOD", range(101, 401)),
    ]
)
dims = space.dimensionality
design = LatinDesign(space)

x_var = "NUM_FOOD"
assert x_var in space.parameter_names

n_starts = 5
n_opts = 5
n_plot = 1000
n_acq = 2000

X_init = design.get_samples(n_starts)
X_plot = design.get_samples(n_plot)
noise_std = 0.05
kernel = GPy.kern.RBF(
    dims,
    lengthscale=[length_scales[name] for name in space.parameter_names],
    ARD=True,
)
normalizer = GPy.normalizer.Standardize()


def X_to_hp(X: np.ndarray) -> HyperParams:
    """Converts a 1d vector of X values to HyperParams"""
    kwargs = {}
    for param in space.parameters:
        assert param.dimension == 1
        name = param.name
        dim = space.find_parameter_index_in_model(name)[0]
        dtype = int if isinstance(param, DiscreteParameter) else float
        kwargs[name] = dtype(X[dim])
    return HyperParams(**kwargs,
                       STEPS = 200,
                       GRID_X = 10,
                       GRID_Y = 10,
                       PREY_SPAWN_RATE = 0,
                       PRED_SPAWN_RATE = 0,
                       MAX_FOOD = 1000,
                       PREY_DEATH_FROM_PRED = 0.1,
                       PREY_ENERGY = 20,
                       PRED_ENERGY = 50,
                       PREY_STEP_ENERGY = 2,
                       PRED_STEP_ENERGY = 3,
                       PREY_ENERGY_FROM_FOOD = 3,
                    #    PRED_ENERGY_FROM_PREY = 10,
                       PREY_REPRODUCTION_THRESHOLD = 15,
                       PRED_REPRODUCTION_THRESHOLD = 40,
                       PREY_REPRODUCTION_CHANCE = 0.3,
                       PRED_REPRODUCTION_CHANCE = 0.1,
                      )


def f(X: np.ndarray):
    # Objective function of average predator count
    """Function to emulate"""

    avg_preds = []
    for x_vec in X:
        hp = X_to_hp(x_vec)

        n_preds = []
        print(f"Running {hp}")
        for summary in simulate(hp):
            n_preds.append(summary.num_preds)
        avg_preds.append(np.mean(n_preds))

    return np.array(avg_preds).reshape(-1, 1)

def g(X: np.ndarray):
    # Objective function of MSE loss from reconstructed LV model
    """Function to emulate"""

    mses = []
    n_preys = []
    n_preds = []
    for x_vec in X:
        hp = X_to_hp(x_vec)
        print(f"Running {hp}")
        for summary in simulate(hp):
            n_preys.append(summary.num_preys)
            n_preds.append(summary.num_preds)

        est = estimate(n_preys, n_preds, error_bound=1000, success_bound=300)
        error = est.get_mse()
        mses.append(error)

    return np.array(mses).reshape(-1, 1)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-o", "--output", required=True)
    args = argparser.parse_args()

    # --- Init ---
    Y_init = f(X_init)
    gpy_model = GPy.models.GPRegression(X_init, Y_init, kernel.copy(), noise_var=noise_std**2, normalizer=normalizer)
    emukit_model = GPyModelWrapper(gpy_model, n_restarts=5)
    # acquisition = ModelVariance(emukit_model)
    X_acq = design.get_samples(n_acq)
    acquisition = IntegratedVarianceReduction(emukit_model, space, x_monte_carlo=X_acq)
    loop = ExperimentalDesignLoop(space, emukit_model, acquisition)

    # --- Main loop ---
    print(emukit_model.model.kern)
    loop.run_loop(f, n_opts)
    print(emukit_model.model.kern)

    # --- Plot graph ---
    if dims == 1:
        x_dim = space.find_parameter_index_in_model(x_var)[0]
        y_mean, y_var = emukit_model.predict(X_plot)
        y_low = y_mean - 1.96 * np.sqrt(y_var)
        y_high = y_mean + 1.96 * np.sqrt(y_var)
        x_1d_idx = np.argsort(X_plot[:, x_dim])  # Plot the first dimension only
        x = X_plot[x_1d_idx, x_dim]
        y_low = y_low[x_1d_idx, 0]
        y_mid = y_mean[x_1d_idx, 0]
        y_high = y_high[x_1d_idx, 0]

        fig, ax = plt.subplots(1, 1)

        ax.plot(x, y_mid, "k", lw=2)
        ax.fill_between(x, y_low, y_high, alpha=0.5)
        ax.scatter(loop.loop_state.X[:, x_dim], loop.loop_state.Y[:, 0])
        ax.set(
            xlabel=x_var,
            ylabel="Objective",
            title=f"Emulating objective function for {x_var}",
        )
        
    elif dims == 2:
        x1_min, x1_max = space.parameters[0].bounds[0]
        x2_min, x2_max = space.parameters[1].bounds[0]

        x1_plot = np.linspace(x1_min, x1_max, 10)
        x2_plot = np.linspace(x2_min, x2_max, 10)

        x1_plot, x2_plot = np.meshgrid(x1_plot, x2_plot)

        x_plot = np.vstack((x1_plot.flatten(), x2_plot.flatten())).T
        x_plot_low, x_plot_high = np.array_split(convert_x_list_to_array([x_plot, x_plot]), 2)
        
        y_plot_mean_low, y_plot_var_low = emukit_model.predict(x_plot_low)
        y_low = y_plot_mean_low - 1.96*np.sqrt(y_plot_var_low)
        y_plot_mean_high, y_plot_var_high = emukit_model.predict(x_plot_high)
        y_high = y_plot_mean_high + 1.96*np.sqrt(y_plot_var_high)

        low = [x1_plot.reshape(-1), x2_plot.reshape(-1), y_low.reshape(-1)]
        high = [x1_plot.reshape(-1), x2_plot.reshape(-1), y_high.reshape(-1)]

        y_plot_mean, y_plot_var = emukit_model.predict(x_plot)

        fig = plt.figure()

        ax = fig.add_subplot(projection='3d')
        
        ax.plot_wireframe(x1_plot, x2_plot, y_plot_mean.reshape(x1_plot.shape),
                          rcount = 20, ccount = 20, color="k")
        
        ax.plot(*low, alpha=0)
        ax.plot(*high, alpha=0)
        
        ax.set(
            xlabel=space.parameter_names[0],
            ylabel=space.parameter_names[1],
            zlabel="Objective",
            title=f"Bayes opt objective function for {x_var}",
        )

        # 3D fill between, doesn't seem to work great though
        fill_between_3d(ax, *low, *high, mode=1, alpha=0.3)

    fig.savefig(args.output, dpi=300)
    plt.legend()
    plt.show()