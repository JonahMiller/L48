import sys
from typing import Optional

import GPy
import numpy as np
from emukit.bayesian_optimization.acquisitions.entropy_search import (
    MultiInformationSourceEntropySearch,
)
from emukit.core import (
    ContinuousParameter,
    DiscreteParameter,
    InformationSourceParameter,
    ParameterSpace,
)
from emukit.core.acquisition import Acquisition
from emukit.core.initial_designs.latin_design import LatinDesign
from emukit.core.loop import FixedIntervalUpdater, OuterLoop, SequentialPointCalculator
from emukit.core.loop.loop_state import create_loop_state
from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.core.optimization.multi_source_acquisition_optimizer import (
    MultiSourceAcquisitionOptimizer,
)
from emukit.experimental_design.experimental_design_loop import ExperimentalDesignLoop
from emukit.model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.convert_lists_to_array import (
    convert_x_list_to_array,
    convert_xy_lists_to_arrays,
)
from emukit.multi_fidelity.kernels import LinearMultiFidelityKernel

sys.path.append("..")
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel

from emulation.emukit_custom import LinearMultiFidelityModel
from high_fidelity.py_interface import HyperParams as HighFidelityHyperParams
from high_fidelity.py_interface import simulate as high_fidelity_simulate
from low_fidelity.main import HyperParams as LowFidelityHyperParams
from low_fidelity.main import simulate as low_fidelity_simulate

space = ParameterSpace(
    [
        # DiscreteParameter("NUM_FOOD", range(0, 101)),
        # DiscreteParameter("MAX_FOOD", range(100, 1001)),
        # ContinuousParameter("PREY_STARVATION", 0.1, 10),
        # ContinuousParameter("PRED_STARVATION", 0.1, 10),
        # DiscreteParameter("STARTING_PREY", range(0, 1001)),
        # DiscreteParameter("STARTING_PREDATOR", range(0, 1001)),
        # ContinuousParameter("PREY_DEATH_FROM_PRED", 0.1, 1),
        # DiscreteParameter("BERRY_ENERGY", range(0, 201)),
        DiscreteParameter("NUM_FOOD", range(0, 1001)),
    ]
)
dims = space.dimensionality
design = LatinDesign(space)


def f_low(X: np.ndarray):
    """Low fidelity function to emulate"""

    avg_preds = []
    for x_vec in X:
        # hp = LowFidelityHyperParams(PREY_ENERGY_FROM_FOOD=x_vec[0], NUM_FOOD=x_vec[1])
        hp = LowFidelityHyperParams(
            NUM_FOOD=x_vec[0], STEPS=400, PREY_DEATH_FROM_PRED=0.1, PRED_SPAWN_RATE=0, PREY_SPAWN_RATE=0
        )

        n_preds = []
        print(f"Running low fidelity with {hp}")
        for summary in low_fidelity_simulate(hp):
            n_preds.append(summary.num_preds)
        avg_preds.append(np.mean(n_preds))

    return np.array(avg_preds).reshape(-1, 1)


def f_high(X: np.ndarray):
    """High fidelity function to emulate"""

    avg_preds = []
    for x_vec in X:
        # hp = HighFidelityHyperParams(BERRY_VALUE=x_vec[0], BERRY_SPAWN_RATE=x_vec[1])
        hp = HighFidelityHyperParams(
            BERRY_SPAWN_RATE=x_vec[0],
            STEPS=400,
            PREDATOR_EATING_PROBABILITY=0.1,
            PREY_SPAWN_RATE=0,
            PREDATOR_SPAWN_RATE=0,
        )

        n_preds = []
        print(f"Running high fidelity with {hp}")
        for summary in high_fidelity_simulate(hp, path_to_out="high_fidelity/out"):
            n_preds.append(summary.num_preds)
        avg_preds.append(np.mean(n_preds))

    return np.array(avg_preds).reshape(-1, 1)


n_starts_low = 20
n_starts_high = 2
n_plot = 1000
n_acq = 2000

low_fidelity_cost = 1
high_fidelity_cost = 5

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    X_train_low = design.get_samples(n_starts_low)
    Y_train_low = f_low(X_train_low)
    X_train_high = np.random.permutation(X_train_low)[:n_starts_high]
    Y_train_high = f_high(X_train_high)

    X_train, Y_train = convert_xy_lists_to_arrays([X_train_low, X_train_high], [Y_train_low, Y_train_high])

    low_kernel = GPy.kern.RBF(dims, variance=1, lengthscale=20)
    # low_kernel.lengthscale.constrain_bounded(0.1, 500)
    err_kernel = GPy.kern.RBF(dims, variance=1, lengthscale=20)
    # err_kernel.lengthscale.constrain_bounded(0.1, 500)

    mf_kernel = LinearMultiFidelityKernel([low_kernel, err_kernel])

    gpy_linear_mf_model = LinearMultiFidelityModel(
        X_train,
        Y_train,
        mf_kernel,
        2,
        GPy.likelihoods.mixed_noise.MixedNoise(
            [
                GPy.likelihoods.Gaussian(variance=0.5**2),
                GPy.likelihoods.Gaussian(variance=0.5**2),
            ]
        ),
        normalizer=True,
    )

    print(gpy_linear_mf_model.kern)
    # gpy_linear_mf_model.optimize_restarts(num_restarts=10, verbose=True)

    # gpy_linear_mf_model.optimize()
    print(gpy_linear_mf_model.kern)

    fig = plt.figure()
    if dims == 1:
        # Plot
        X_plot = np.sort(design.get_samples(n_plot), axis=0)

        X_plot_low, X_plot_high = np.array_split(convert_x_list_to_array([X_plot, X_plot]), 2)
        Y_metadata_low = {"output_index": X_plot_low[:, -1].astype(int)}
        Y_metadata_high = {"output_index": X_plot_high[:, -1].astype(int)}
        Y_plot_mean_low, Y_plot_var_low = gpy_linear_mf_model.predict(X_plot_low, Y_metadata=Y_metadata_low)
        Y_plot_mean_high, Y_plot_var_high = gpy_linear_mf_model.predict(X_plot_high, Y_metadata=Y_metadata_high)
        Y_plot_low_lower = Y_plot_mean_low - 1.96 * np.sqrt(Y_plot_var_low)
        Y_plot_low_upper = Y_plot_mean_low + 1.96 * np.sqrt(Y_plot_var_low)
        Y_plot_high_lower = Y_plot_mean_high - 1.96 * np.sqrt(Y_plot_var_high)
        Y_plot_high_upper = Y_plot_mean_high + 1.96 * np.sqrt(Y_plot_var_high)

        ax = fig.add_subplot()

        ax.scatter(X_train_low.reshape(-1), Y_train_low.reshape(-1))
        ax.scatter(X_train_high.reshape(-1), Y_train_high.reshape(-1))

        ax.plot(X_plot.reshape(-1), Y_plot_mean_low.reshape(-1), color="C0", label="low fidelity")
        ax.fill_between(
            X_plot.reshape(-1),
            Y_plot_low_lower.reshape(-1),
            Y_plot_low_upper.reshape(-1),
            color="C0",
            alpha=0.5,
        )

        ax.plot(X_plot.reshape(-1), Y_plot_mean_high.reshape(-1), color="C1", label="high fidelity")
        ax.fill_between(
            X_plot.reshape(-1),
            Y_plot_high_lower.reshape(-1),
            Y_plot_high_upper.reshape(-1),
            color="C1",
            alpha=0.5,
        )
    if dims == 2:
        # Surface plot
        X1_min, X1_max = space.parameters[0].bounds[0]
        X2_min, X2_max = space.parameters[1].bounds[0]

        X1_plot = np.linspace(X1_min, X1_max, 40)
        X2_plot = np.linspace(X2_min, X2_max, 40)

        X1_plot, X2_plot = np.meshgrid(X1_plot, X2_plot)

        X_plot = np.vstack((X1_plot.flatten(), X2_plot.flatten())).T
        print(X1_plot.shape)
        print(X2_plot.shape)
        print(X_plot.shape)

        X_plot_low, X_plot_high = np.array_split(convert_x_list_to_array([X_plot, X_plot]), 2)

        Y_metadata_high = {"output_index": X_plot_high[:, -1].astype(int)}
        Y_plot_mean_high, Y_plot_var_high = gpy_linear_mf_model.predict(X_plot_high, Y_metadata=Y_metadata_high)

        ax = fig.add_subplot(projection="3d")

        ax.scatter(
            X_train_low[:, 0].reshape(-1),
            X_train_low[:, 1].reshape(-1),
            Y_train_low.reshape(-1),
            color="C0",
            label="low fidelity",
        )
        ax.scatter(
            X_train_high[:, 0].reshape(-1),
            X_train_high[:, 1].reshape(-1),
            Y_train_high.reshape(-1),
            color="C1",
            label="high fidelity",
        )

        print(Y_plot_mean_high.shape)

        ax.plot_wireframe(X1_plot, X2_plot, Y_plot_mean_high.reshape(X1_plot.shape), rcount=20, ccount=20)

        ax.set_xlabel(space.parameter_names[0])
        ax.set_ylabel(space.parameter_names[1])
        ax.set_zlabel("Avg predator count")

    fig.legend()
    fig.savefig("linear_multi_fidelity.png", dpi=300)
    plt.show()
