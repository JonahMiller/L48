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

from low_fidelity.main import HyperParams, simulate

length_scales = {
    "PRED_REPRODUCTION_CHANCE": 0.1,
    "PRED_REPRODUCTION_THRESHOLD": 500,
    "PRED_ENERGY_FROM_PREY": 50,
    "STEPS": 50,
    "NUM_FOOD": 25,
}
space = ParameterSpace(
    [
        # ContinuousParameter("PRED_REPRODUCTION_CHANCE", 0, 1),
        # DiscreteParameter("PRED_REPRODUCTION_THRESHOLD", range(10, 5001)),
        # DiscreteParameter("PRED_ENERGY_FROM_PREY", range(50, 51)),
        # DiscreteParameter("STEPS", range(0, 1001)),
        DiscreteParameter("NUM_FOOD", range(0, 201)),
    ]
)
design = LatinDesign(space)

x_var = "NUM_FOOD"
assert x_var in space.parameter_names

n_starts = 5
n_opts = 20
n_plot = 1000
n_acq = 2000

X_init = design.get_samples(n_starts)
X_plot = design.get_samples(n_plot)
noise_std = 0.1
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
    return HyperParams(**kwargs)


def f(X: np.ndarray):
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


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-o", "--output", required=True)
    args = argparser.parse_args()

    # --- Init ---
    Y_init = f(X_init)
    gpy_model = GPy.models.GPRegression(X_init, Y_init, kernel.copy(), noise_var=noise_std**2, normalizer=normalizer)
    emukit_model = GPyModelWrapper(gpy_model)
    # acquisition = ModelVariance(emukit_model)
    X_acq = design.get_samples(n_acq)
    acquisition = IntegratedVarianceReduction(emukit_model, space, x_monte_carlo=X_acq)
    loop = ExperimentalDesignLoop(space, emukit_model, acquisition)

    # --- Main loop ---
    loop.run_loop(f, n_opts)

    # --- Plot graph ---
    # Re-fit a model with the acquired points, less buggy this way
    X_final = loop.loop_state.X
    Y_final = loop.loop_state.Y
    gpy_model = GPy.models.GPRegression(
        X_final, Y_final, kernel.copy(), noise_var=noise_std**2, normalizer=normalizer
    )

    x_dim = space.find_parameter_index_in_model(x_var)[0]
    y_low, y_mid, y_high = gpy_model.predict_quantiles(X_plot, quantiles=(2.5, 50, 97.5))
    x_1d_idx = np.argsort(X_plot[:, x_dim])  # Plot the first dimension only
    x = X_plot[x_1d_idx, x_dim]
    y_low = y_low[x_1d_idx, 0]
    y_mid = y_mid[x_1d_idx, 0]
    y_high = y_high[x_1d_idx, 0]

    fig, ax = plt.subplots(1, 1)

    ax.plot(x, y_mid, "k", lw=2)
    ax.fill_between(x, y_low, y_high, alpha=0.5)
    ax.scatter(X_final[:, x_dim], Y_final[:, 0])
    ax.set(
        xlabel=x_var,
        ylabel="Objective",
        title=f"Emulating objective function for {x_var}",
    )

    fig.savefig(args.output, dpi=300)

    plt.show()
