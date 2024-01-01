import GPy
import numpy as np
from emukit.core import ContinuousParameter, DiscreteParameter, ParameterSpace
from emukit.core.initial_designs.latin_design import LatinDesign
from emukit.experimental_design.experimental_design_loop import ExperimentalDesignLoop
from emukit.model_wrappers import GPyModelWrapper
from emukit.multi_fidelity.convert_lists_to_array import (
    convert_x_list_to_array,
    convert_xy_lists_to_arrays,
)
from emukit.multi_fidelity.kernels import LinearMultiFidelityKernel

from high_fidelity.py_interface import HyperParams as HighFidelityHyperParams
from high_fidelity.py_interface import simulate as high_fidelity_simulate
from low_fidelity.main import HyperParams as LowFidelityHyperParams
from low_fidelity.main import simulate as low_fidelity_simulate


class LinearMultiFidelityModel(GPy.core.GP):
    """
    Copied and modified from GPyLinearMultiFidelityModel
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        kernel: GPy.kern.Kern,
        n_fidelities: int,
        likelihood: GPy.likelihoods.Likelihood | None = None,
        **kwargs,
    ):
        """

        :param X: Training data features with fidelity input appended as last column
        :param Y: Training data targets
        :param kernel: Multi-fidelity kernel
        :param n_fidelities: Number of fidelities in problem
        :param likelihood: GPy likelihood object.
                           Defaults to MixedNoise which has different noise levels for each fidelity
        """

        # Input checks
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be an array")

        if not isinstance(Y, np.ndarray):
            raise ValueError("Y should be an array")

        if X.ndim != 2:
            raise ValueError("X should be 2d")

        if Y.ndim != 2:
            raise ValueError("Y should be 2d")

        if np.any(X[:, -1] >= n_fidelities):
            raise ValueError("One or more points has a higher fidelity index than number of fidelities")

        # Make default likelihood as different noise for each fidelity
        if likelihood is None:
            likelihood = GPy.likelihoods.mixed_noise.MixedNoise(
                [GPy.likelihoods.Gaussian(variance=1.0) for _ in range(n_fidelities)]
            )
        y_metadata = {"output_index": X[:, -1].astype(int)}
        super().__init__(X, Y, kernel, likelihood, Y_metadata=y_metadata, **kwargs)


space = ParameterSpace(
    [
        DiscreteParameter("NUM_FOOD", range(0, 201)),
    ]
)
design = LatinDesign(space)


def f_low(X: np.ndarray):
    """Low fidelity function to emulate"""

    avg_preds = []
    for x_vec in X:
        hp = LowFidelityHyperParams(NUM_FOOD=int(x_vec[0]))

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
        hp = HighFidelityHyperParams(BERRY_SPAWN_RATE=int(x_vec[0]))

        n_preds = []
        print(f"Running high fidelity with {hp}")
        for summary in high_fidelity_simulate(hp):
            n_preds.append(summary.num_preds)
        avg_preds.append(np.mean(n_preds))

    return np.array(avg_preds).reshape(-1, 1)


n_starts_low = 10
n_starts_high = 4
n_opts = 5
n_plot = 1000
n_acq = 2000

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    X_train_low = design.get_samples(n_starts_low)
    Y_train_low = f_low(X_train_low)
    X_train_high = design.get_samples(n_starts_high)
    Y_train_high = f_high(X_train_high)

    X_train, Y_train = convert_xy_lists_to_arrays([X_train_low, X_train_high], [Y_train_low, Y_train_high])

    mf_kernel = LinearMultiFidelityKernel(
        [
            GPy.kern.RBF(1, lengthscale=20),  # Kernel for low fidelity
            GPy.kern.RBF(1, lengthscale=20),  # Kernel for error term
        ]
    )
    gpy_linear_mf_model = LinearMultiFidelityModel(
        X_train,
        Y_train,
        mf_kernel,
        2,
        GPy.likelihoods.mixed_noise.MixedNoise(
            [
                GPy.likelihoods.Gaussian(variance=0.3**2),
                GPy.likelihoods.Gaussian(variance=0.1**2),
            ]
        ),
        normalizer=True,
    )

    print(gpy_linear_mf_model.kern)
    gpy_linear_mf_model.optimize_restarts(num_restarts=10, verbose=True)
    print(gpy_linear_mf_model.kern)

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

    fig, ax = plt.subplots(1, 1)
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

    fig.legend()
    fig.savefig("linear_multi_fidelity.png", dpi=300)
    plt.show()