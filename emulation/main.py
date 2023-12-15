import GPy
import matplotlib.pyplot as plt
import numpy as np
from emukit.core import ContinuousParameter, DiscreteParameter, ParameterSpace
from emukit.experimental_design.acquisitions import (
    IntegratedVarianceReduction,
    ModelVariance,
)
from emukit.experimental_design.experimental_design_loop import ExperimentalDesignLoop
from emukit.model_wrappers import GPyModelWrapper

from low_fidelity.main import simulate


def f(X: np.ndarray):
    init_preys = np.round(X[:, 0])
    final_preys = []

    for init_prey in init_preys:
        print(f"Running simulation with {init_prey} preys")
        steps = 100
        grid_x = 40
        grid_y = 40
        init_pred = 100

        states = simulate(steps, grid_x, grid_y, init_prey, init_pred, save_all=False)
        board = states[-1]
        final_preys.append([len(board._preys)])

    return np.array(final_preys)


p = DiscreteParameter("init_prey", range(0, 5000))
space = ParameterSpace([p])

# Need some samples to initialize the model
X = space.sample_uniform(1)
Y = f(X)
gpy_model = GPy.models.GPRegression(X, Y, GPy.kern.RBF(X.shape[1], lengthscale=100).add(GPy.kern.White(X.shape[1], 1)))

emukit_model = GPyModelWrapper(gpy_model)
acquisition = ModelVariance(emukit_model)
loop = ExperimentalDesignLoop(space, emukit_model, acquisition)

if __name__ == "__main__":
    loop.run_loop(f, 10)

    X_all = space.sample_uniform(1000)
    predicted_y, predicted_var = emukit_model.predict(X_all)
    predicted_std = np.sqrt(predicted_var)

    x_1d = X_all[:, 0]
    y_1d = predicted_y[:, 0]
    y_1d_low = y_1d - 2 * predicted_std[:, 0]
    y_1d_high = y_1d + 2 * predicted_std[:, 0]
    x_1d, y_1d, y_1d_low, y_1d_high = zip(*sorted(zip(x_1d, y_1d, y_1d_low, y_1d_high)))

    fig, ax = plt.subplots(1, 1)

    ax.scatter(loop.loop_state.X[:, 0], loop.loop_state.Y[:, 0])
    ax.plot(x_1d, y_1d, "k", lw=2)
    ax.fill_between(x_1d, y_1d_low, y_1d_high, alpha=0.5)

    fig.savefig("integrated_variance.png", dpi=300)

    plt.show()
