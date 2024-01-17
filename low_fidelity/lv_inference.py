# Based on https://www.pymc.io/projects/examples/en/latest/ode_models/ODE_Lotka_Volterra_multiple_ways.html

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import arviz as az
import pymc as pm
import pytensor.tensor as pt
from pytensor.compile.ops import as_op

from scipy.integrate import odeint, solve_ivp
from scipy.optimize import least_squares

import sys
sys.path.append("..")
from low_fidelity.lv_param_est import estimate
from low_fidelity.main import simulate, HyperParams

START_STEP = 50
LS_FINAL_STEP = 200 
STEP = 1500

hp = HyperParams(
        STEPS = STEP,
        GRID_X = 10,
        GRID_Y = 10,
        INIT_PREY = 200,
        INIT_PRED = 20,
        NUM_FOOD = 250,
        MAX_FOOD = 1000,
        PREY_DEATH_FROM_PRED = 0.1,
        PREY_ENERGY = 20,
        PRED_ENERGY = 50,
        PREY_STEP_ENERGY = 2,
        PRED_STEP_ENERGY = 3,
        PREY_ENERGY_FROM_FOOD = 3,
        PRED_ENERGY_FROM_PREY = 10,
        PREY_REPRODUCTION_THRESHOLD = 15,
        PRED_REPRODUCTION_THRESHOLD = 20,
        PREY_REPRODUCTION_CHANCE = 0.3,
        PRED_REPRODUCTION_CHANCE = 0.1,
        PREY_SPAWN_RATE = 0,
        PRED_SPAWN_RATE = 0)


def derivative(X, t, theta):
    x, y = X
    a_0, a_1, a_2, a_3, a_4, a_5, x_0, y_0 = theta
    dx_dt = x * (a_0 + a_1*x + a_2*y)
    dy_dt = y * (a_3 + a_4*x + a_5*y)
    return [dx_dt, dy_dt]

def full_lv(theta, prey_data, pred_data, graph_name):
    steps = np.arange(START_STEP, STEP + 1)
    steps2 = np.arange(0, START_STEP)[::-1]
    full_steps = np.arange(0, STEP + 1)
    x_y = odeint(func=derivative, y0=theta[-2:], t=steps, args=(theta, ))
    x_y2 = np.flipud(odeint(func=derivative, y0=theta[-2:], t=steps2, args=(theta, )))
    new_x_y = np.concatenate((x_y2, x_y))
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_data_full(ax, prey_data, pred_data, full_steps)
    plot_model(ax, new_x_y, full_steps, title=graph_name)


def plot_model(ax, x_y, time, alpha=1, lw=2, title="LV model",):
    ax.plot(time, x_y[:, 0], color='navy', alpha=alpha, lw=lw, label="Prey (Model)")
    ax.plot(time, x_y[:, 1], color='red', alpha=alpha, lw=lw, label="Predator (Model)")
    ax.legend(fontsize=14, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(title, fontsize=16)
    ax.grid()
    return ax

def plot_data_full(ax, prey_data, pred_data, time, lw=2, title="LV model data"):
    ax.plot(time, prey_data, color='#1f77b4', lw=lw, marker="+", markersize=2, label="Prey (Data)")
    ax.plot(time, pred_data, color='#ff7f0e', lw=lw, marker="o", markersize=1, label="Predator (Data)")
    ax.set_xlim([0, STEP + 1])
    ax.set_xlabel("Step", fontsize=14)
    ax.set_ylabel("Population size", fontsize=14)
    ax.set_title(title, fontsize=16)
    return ax

# decorator with input and output types a Pytensor double float tensors
@as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
def pytensor_forward_model_matrix(theta):
    return odeint(func=derivative, y0=theta[-2:], t=np.arange(START_STEP, LS_FINAL_STEP + 1), args=(theta,))


class inference:
    def __init__(self, prey_data, pred_data):
        self.prey_data = prey_data[START_STEP:LS_FINAL_STEP + 1]
        self.pred_data = pred_data[START_STEP:LS_FINAL_STEP + 1]
        self.n = len(self.prey_data) + START_STEP
        self.x_0 = self.prey_data[0]
        self.y_0 = self.pred_data[0]
        self.start_step = START_STEP
        self.time = np.arange(self.start_step, self.n, 0.01)
        self.df = pd.DataFrame(dict(
            step = np.arange(self.start_step, self.n),
            prey = self.prey_data,
            pred = self.pred_data
        ))

    def init_theta(self):
        est = estimate(self.prey_data, self.pred_data)
        est.form_a_vectors()
        a_0, a_1, a_2, a_3, a_4, a_5 = est.return_a_vectors()
        self.theta = np.array([a_0, a_1, a_2, a_3, a_4, a_5, self.x_0, self.y_0])
    
    def init_ode_plot(self):
        x_y = odeint(func=derivative, y0=self.theta[-2:], t=self.time, args=(self.theta, ))
        _, ax = plt.subplots()
        self.plot_data(ax, lw=0)
        plot_model(ax, x_y, self.time, title="LV model")

    def ode_model_resid(self, theta):
        return (
            self.df[["prey", "pred"]] - odeint(func=derivative, y0=theta[-2:], t=self.df.step, args=(theta,))
        ).values.flatten()
    
    def least_squares_pred(self):
        results = least_squares(self.ode_model_resid, x0=self.theta)
        self.ls_theta = results.x
        x_y = odeint(func=derivative, y0=self.ls_theta[-2:], t=self.time, args=(self.ls_theta, ))
        fig, ax = plt.subplots()
        self.plot_data(ax, lw=0)
        plot_model(ax, x_y, self.time, title="Least squares LV model")

    def plot_data(self, ax, lw=2, title="LV model data"):
        ax.plot(self.df.step, self.prey_data, color='#1f77b4', lw=lw, marker="+", markersize=4, label="Prey (Data)")
        ax.plot(self.df.step, self.pred_data, color='#ff7f0e', lw=lw, marker="o", markersize=2, label="Predator (Data)")
        ax.set_xlim([self.start_step, self.n])
        ax.set_xlabel("Step", fontsize=14)
        ax.set_ylabel("Population size", fontsize=14)
        ax.set_title(title, fontsize=16)
        return ax
    
    def return_thetas(self):
        return self.theta, self.ls_theta

    def infer(self):
        az.style.use("arviz-whitegrid")
        theta = self.ls_theta
        with pm.Model() as model:
            # Priors
            a_0 = pm.TruncatedNormal("a_0", mu=theta[0], sigma=1,lower=-100, initval=theta[0])
            a_1 = pm.TruncatedNormal("a_1", mu=theta[1], sigma=0.01, lower=-100,initval=theta[1])
            a_2 = pm.TruncatedNormal("a_2", mu=theta[2], sigma=0.1, lower=-100,initval=theta[2])
            a_3 = pm.TruncatedNormal("a_3", mu=theta[3], sigma=1, lower=-100, initval=theta[3])
            a_4 = pm.TruncatedNormal("a_4", mu=theta[4], sigma=0.01, lower=-100, initval=theta[4])
            a_5 = pm.TruncatedNormal("a_5", mu=theta[5], sigma=0.1, lower=-100, initval=theta[5])
            x_0 = pm.TruncatedNormal("x_0", mu=theta[6], sigma=1000, lower=0, initval=theta[6])
            y_0 = pm.TruncatedNormal("y_0", mu=theta[7], sigma=100, lower=0, initval=theta[7])
            sigma = pm.HalfNormal("sigma", 10)
            # Ode solution function
            ode_solution = pytensor_forward_model_matrix(
                pm.math.stack([a_0, a_1, a_2, a_3, a_4, a_5, x_0, y_0])
                )
            # Likelihood
            pm.Normal("Y_obs", mu=ode_solution, sigma=sigma, observed=self.df[["prey", "pred"]].values)

        vars_list = list(model.values_to_rvs.keys())[:-1]
        sampler = "DEMetropolisZ"
        tune = draws = 5000
        with model:
            trace_DEMZ = pm.sample(step=[pm.DEMetropolis(vars_list)], tune=tune, draws=draws)
        trace = trace_DEMZ
        az.summary(trace)
        az.plot_trace(trace, kind="rank_bars")
        plt.suptitle(f"Trace Plot {sampler}")
        az.style.use("default")
        fig, ax = plt.subplots()
        self.plot_inference(ax, trace, 
                            title=f"Data and Inference Model Runs\n{sampler} Sampler")

    def plot_model_trace(self, ax, trace_df, row_idx, lw=1, alpha=0.2):
        cols = ["a_0", "a_1", "a_2", "a_3", "a_4", "a_5", "x_0", "y_0"]
        row = trace_df.iloc[row_idx, :][cols].values
        theta = row
        x_y = odeint(func=derivative, y0=theta[-2:], t=self.time, args=(theta,))
        plot_model(ax, x_y, time=self.time, lw=lw, alpha=alpha)

    def plot_inference(self, ax, trace, num_samples=25, 
                       title="Inference Model", plot_model_kwargs=dict(lw=1, alpha=0.2)):
        trace_df = az.extract(trace, num_samples=num_samples).to_dataframe()
        self.plot_data(ax, lw=0)
        for row_idx in range(num_samples):
            self.plot_model_trace(ax, trace_df, row_idx, **plot_model_kwargs)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_title(title, fontsize=16)


if __name__ == "__main__":
    # Generate predator/prey data from low fidelity model
    n_preys = []
    n_preds = []
    for summary in simulate(hp):
        n_preys.append(summary.num_preys)
        n_preds.append(summary.num_preds)

    inf = inference(prey_data=n_preys, pred_data=n_preds)

    inf.init_theta()
    inf.init_ode_plot()
    inf.least_squares_pred()
    inf.infer()

    theta, ls_theta = inf.return_thetas()
    print(ls_theta)
    full_lv(theta, n_preys, n_preds, "Extrapolated LV model")
    full_lv(ls_theta, n_preys, n_preds, "Extrapolated least squares LV model")