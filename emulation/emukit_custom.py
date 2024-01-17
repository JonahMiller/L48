from typing import Callable, Optional

import emukit
import emukit.multi_fidelity
import GPy
import numpy as np
from emukit.core.interfaces import IModel
from emukit.core.loop import LoopState, ModelUpdater


class NoOptimiseUpdater(ModelUpdater):
    """Updates the model inputs without calling model.optimise()."""

    def __init__(self, model: IModel, targets_extractor_fcn: Callable = None) -> None:
        self.model = model

        if targets_extractor_fcn is None:
            self.targets_extractor_fcn = lambda loop_state: loop_state.Y
        else:
            self.targets_extractor_fcn = targets_extractor_fcn

    def update(self, loop_state: LoopState) -> None:
        targets = self.targets_extractor_fcn(loop_state)
        self.model.set_data(loop_state.X, targets)


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
        likelihood: Optional[GPy.likelihoods.Likelihood] = None,
        **kwargs,  # only change is to add **kwargs
    ):
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


class LinearMultiFidelityKernel(emukit.multi_fidelity.kernels.LinearMultiFidelityKernel):
    def to_dict(self):
        # input_dict = super()._save_to_input_dict()
        # input_dict["kernels"] = [k.to_dict() for k in self.kernels]
        return {
            "kernels": [k.to_dict() for k in self.kernels],
            "class": "emukit.multi_fidelity.kernels.LinearMultiFidelityKernel",
            "name": "LinearMultiFidelityKernel",
        }

    @staticmethod
    def from_dict(input_dict):
        """
        Instantiate an object of a derived class using the information
        in input_dict (built by the to_dict method of the derived class).
        More specifically, after reading the derived class from input_dict,
        it calls the method _build_from_input_dict of the derived class.
        Note: This method should not be overrided in the derived class. In case
        it is needed, please override _build_from_input_dict instate.

        :param dict input_dict: Dictionary with all the information needed to
           instantiate the object.
        """
        kernels = []
        for kernel_dict in input_dict["kernels"]:
            kernel_class = kernel_dict.pop("class")
            kernel_dict["name"] = str(kernel_dict["name"])
            import GPy

            kernel = eval(kernel_class)._build_from_input_dict(kernel_class, kernel_dict)
            kernels.append(kernel)

        return LinearMultiFidelityKernel(kernels)

        kernel_class = eval(kernel_class)

        import copy

        input_dict = copy.deepcopy(input_dict)
        kernel_class = input_dict.pop("class")
        input_dict["name"] = str(input_dict["name"])
        import emukit.multi_fidelity.kernels

        return kernel_class._build_from_input_dict(kernel_class, input_dict)

    @staticmethod
    def _build_from_input_dict(kernel_class, input_dict):
        return kernel_class(**input_dict)
