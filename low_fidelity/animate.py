import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib import animation as animation

plt.rcParams["animation.html"] = "jshtml"


class Animate:
    def __init__(
        self,
        steps: int,
        grid_x: int,
        grid_y: int,
        preys_pos: list[dict[tuple[int, int], int]],
        preds_pos: list[dict[tuple[int, int], int]],
        foods_pos: list[dict[tuple[int, int], int]],
    ):
        self.steps = steps
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.preys_pos = preys_pos
        self.preds_pos = preds_pos
        self.foods_pos = foods_pos

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, self.grid_x - 1)
        self.ax.set_ylim(0, self.grid_y - 1)
        self.ax.set_xticks(np.arange(self.grid_x))
        self.ax.set_yticks(np.arange(self.grid_y))
        plt.grid(True)

        prey_x, prey_y, prey_count = self.coords_at_step(self.preys_pos[0])
        pred_x, pred_y, pred_count = self.coords_at_step(self.preds_pos[0])
        food_x, food_y, food_count = self.coords_at_step(self.foods_pos[0])
        self.prey_scat = self.ax.scatter(prey_x, prey_y, c="g", marker="o", s=prey_count * 10, label="Preys")
        self.pred_scat = self.ax.scatter(pred_x, pred_y, c="r", marker="o", s=pred_count * 10, label="Predators")
        self.food_scat = self.ax.scatter(food_x, food_y, c="b", marker="s", s=10, label="Food")

        self.fig.legend()
        self.txt = self.ax.text(
            0.1,
            0.1,
            "",
            ha="center",
            va="center",
            alpha=0.8,
            transform=self.ax.transAxes,
            fontdict={"color": "black", "backgroundcolor": "white", "size": 10},
        )

    def coords_at_step(self, grid: dict[tuple[int, int], int]):
        xs = []
        ys = []
        counts = []
        for coord, n in grid.items():
            x, y = coord
            xs.append(x)
            ys.append(y)
            counts.append(n)
        return np.array(xs), np.array(ys), np.array(counts)

    def animate(self, i: int):
        prey_x, prey_y, prey_count = self.coords_at_step(self.preys_pos[i])
        pred_x, pred_y, pred_count = self.coords_at_step(self.preds_pos[i])
        food_x, food_y, food_count = self.coords_at_step(self.foods_pos[i])
        self.pred_scat.set_offsets(np.stack([pred_x, pred_y]).T)
        self.pred_scat.set_sizes(pred_count * 10)
        self.prey_scat.set_offsets(np.stack([prey_x, prey_y]).T)
        self.prey_scat.set_sizes(prey_count * 10)
        self.food_scat.set_offsets(np.stack([food_x, food_y]).T)

        n_preys = sum(v for v in self.preys_pos[i].values())
        n_preds = sum(v for v in self.preds_pos[i].values())
        n_foods = sum(v for v in self.foods_pos[i].values())
        self.txt.set_text(f"Step: {i}\nPreys: {n_preys}\nPredators: {n_preds}")
        return self.prey_scat, self.pred_scat, self.txt

    def show(self):
        anim = animation.FuncAnimation(
            fig=self.fig,
            func=self.animate,
            frames=self.steps,
            repeat=True,
            save_count=10,
            blit=True,
        )
        HTML(anim.to_jshtml())
        anim.save("animation.gif", fps=10)
