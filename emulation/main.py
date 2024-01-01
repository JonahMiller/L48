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

import sys
sys.path.append("..")
from low_fidelity.main import HyperParams, simulate
from low_fidelity.lv_param_est import estimate


length_scales = {
    # "PRED_REPRODUCTION_CHANCE": 0.1,
    # "PRED_REPRODUCTION_THRESHOLD": 500,
    # "PRED_ENERGY_FROM_PREY": 50,
    # "STEPS": 50,
    "NUM_FOOD": 10,
}
space = ParameterSpace(
    [
        # ContinuousParameter("PRED_REPRODUCTION_CHANCE", 0, 1),
        # DiscreteParameter("PRED_REPRODUCTION_THRESHOLD", range(10, 5001)),
        # DiscreteParameter("PRED_ENERGY_FROM_PREY", range(50, 51)),
        # DiscreteParameter("STEPS", range(0, 1001)),
        DiscreteParameter("NUM_FOOD", range(101, 401)),
    ]
)
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
    space.dimensionality,
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
                       PRED_ENERGY_FROM_PREY = 10,
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
    Y_init = g(X_init)
    gpy_model = GPy.models.GPRegression(X_init, Y_init, kernel.copy(), noise_var=noise_std**2, normalizer=normalizer)
    emukit_model = GPyModelWrapper(gpy_model, n_restarts=5)
    # acquisition = ModelVariance(emukit_model)
    X_acq = design.get_samples(n_acq)
    acquisition = IntegratedVarianceReduction(emukit_model, space, x_monte_carlo=X_acq)
    loop = ExperimentalDesignLoop(space, emukit_model, acquisition)

    # --- Main loop ---
    print(emukit_model.model.kern)
    loop.run_loop(g, n_opts)
    print(emukit_model.model.kern)

    # --- Plot graph ---
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

    fig.savefig(args.output, dpi=300)

    plt.show()
