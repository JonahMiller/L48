import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from low_fidelity.state import StateSummary


@dataclass
class HyperParams:
    REPRODUCTION_TYPE: str = "FOOD"
    STEPS: int = 1000
    STARTING_PREY: int = 200
    STARTING_PREDATOR: int = 20
    STARTING_BERRY: int = 0
    MAX_BERRY: int = 500
    BERRY_VALUE: int = 50
    PREY_VALUE: int = 50
    PREY_STARTING_ENERGY: int = 50
    PREDATOR_STARTING_ENERGY: int = 50
    PREY_STARVATION: int = 2
    PREDATOR_STARVATION: int = 2
    PREY_STEP_ENERGY: float = 0
    PREDATOR_STEP_ENERGY: float = 0
    # In low-fidelity animals eat in an area of 2500, we want the same here with a circle
    EATING_RADIUS: float = math.sqrt(2500 / math.pi)
    PREDATOR_EATING_PROBABILITY: float = 1
    PREY_REPRODUCTION_ENERGY_THRESHOLD: int = 100
    PREDATOR_REPRODUCTION_ENERGY_THRESHOLD: int = 100
    PREY_REPRODUCTION_PROBABILITY: float = 1
    PREDATOR_REPRODUCTION_PROBABILITY: float = 1
    SPEED: int = 80
    PREY_SPAWN_RATE: float = 0.05
    PREDATOR_SPAWN_RATE: float = 0.05
    BERRY_SPAWN_RATE: int = 10
    SIMULATION_TIMESTEP: float = 0.5


# Helper function
def split_output(output):
    cnt, prey, predator, berry = [], [], [], []

    for line in output.strip().split("\n"):
        a, b, c, d = map(int, line.split())
        cnt.append(a)
        prey.append(b)
        predator.append(c)
        berry.append(d)

    cnt = np.array(cnt)
    prey = np.array(prey)
    predator = np.array(predator)
    berry = np.array(berry)

    return cnt, prey, predator, berry


# Make sure Callable.java is compiled
def simulate(
    hp: HyperParams,
    path_to_out: Path | str = Path(__file__).parent / "out/production/high_fidelity",
) -> list[StateSummary]:
    command = (
        f"java -cp {path_to_out} PyInterface.Callable "
        f"{hp.REPRODUCTION_TYPE} "
        f"{hp.STEPS} "
        f"{hp.STARTING_PREY} "
        f"{hp.STARTING_PREDATOR} "
        f"{hp.STARTING_BERRY} "
        f"{hp.MAX_BERRY} "
        f"{hp.BERRY_VALUE} "
        f"{hp.PREY_VALUE} "
        f"{hp.PREY_STARTING_ENERGY} "
        f"{hp.PREDATOR_STARTING_ENERGY} "
        f"{hp.PREY_STARVATION} "
        f"{hp.PREDATOR_STARVATION} "
        f"{hp.PREY_STEP_ENERGY} "
        f"{hp.PREDATOR_STEP_ENERGY} "
        f"{hp.EATING_RADIUS} "
        f"{hp.PREDATOR_EATING_PROBABILITY} "
        f"{hp.PREY_REPRODUCTION_ENERGY_THRESHOLD} "
        f"{hp.PREDATOR_REPRODUCTION_ENERGY_THRESHOLD} "
        f"{hp.PREY_REPRODUCTION_PROBABILITY} "
        f"{hp.PREDATOR_REPRODUCTION_PROBABILITY} "
        f"{hp.SPEED} "
        f"{hp.PREY_SPAWN_RATE} "
        f"{hp.PREDATOR_SPAWN_RATE} "
        f"{hp.BERRY_SPAWN_RATE} "
        f"{hp.SIMULATION_TIMESTEP}"
    )

    process = subprocess.Popen(
        command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    stdout, stderr = stdout.decode(), stderr.decode()
    if stderr != "":
        print(stderr)

    summaries = []
    for line in stdout.strip().split("\n"):
        step, prey, predator, berry = map(int, line.split())
        summaries.append(StateSummary(prey, predator, berry))

    return summaries


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    hp = HyperParams(PREDATOR_EATING_PROBABILITY=0.1)
    summaries = simulate(hp)
    prey, predator, berry = [], [], []
    for summary in summaries:
        prey.append(summary.num_preys)
        predator.append(summary.num_preds)
        berry.append(summary.num_foods)

    plt.plot(prey, label="prey")
    plt.plot(predator, label="predator")
    plt.plot(berry, label="berry")
    plt.legend()
    plt.show()
