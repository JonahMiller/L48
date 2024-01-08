import numpy as np
import matplotlib.pyplot as plt
from emukit.core import ContinuousParameter, ParameterSpace
from emukit.core.initial_designs import RandomDesign
from emukit.model_wrappers import GPyModelWrapper
import GPy
from emukit.sensitivity.monte_carlo import ModelFreeMonteCarloSensitivity

import sys
sys.path.append("..")
from low_fidelity.main import simulate, HyperParams
from SALib.sample import saltelli
from SALib.analyze import sobol


def perform_sensitivity_analysis(hyperparameter_bounds):
    space = ParameterSpace([ContinuousParameter(name, lower_bound, upper_bound)
                            for name, lower_bound, upper_bound in hyperparameter_bounds])

    design = RandomDesign(space)
    num_data_points = 100
    X = design.get_samples(num_data_points)
    Y = np.zeros((num_data_points, 1))

    for i in range(num_data_points):
        hyperparams = {name: int(X[i, j]) for j, (name, _, _) in enumerate(hyperparameter_bounds)}
        hp = HyperParams(**hyperparams)
        summaries = simulate(hp)
        Y[i] = np.average([summary.num_preds for summary in summaries])

    kernel = GPy.kern.RBF(input_dim=len(hyperparameter_bounds))
    model = GPy.models.GPRegression(X, Y, kernel)
    gpy_model = GPyModelWrapper(model)
    gpy_model.optimize()

    def gp_predict_wrapper(X):
        return gpy_model.predict(X)[0]

    sensitivity = ModelFreeMonteCarloSensitivity(gp_predict_wrapper, space)
    main_effects, total_effects, _ = sensitivity.compute_effects(num_monte_carlo_points=1000)

    print("Main Effects:", main_effects)
    print("Total Effects:", total_effects)


if __name__ == '__main__':
    hyperparameter_bounds = [
        ('PREY_ENERGY_FROM_FOOD', 10, 100),  # (name, lower_bound, upper_bound)
        ('PRED_ENERGY_FROM_PREY', 10, 100),
        ('INIT_PREY', 10, 100)
    ]
    perform_sensitivity_analysis(hyperparameter_bounds)
