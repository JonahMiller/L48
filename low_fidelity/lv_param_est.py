import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

# hp = HyperParams(
#     STEPS = 500,
#     GRID_X = 10,
#     GRID_Y = 10,
#     INIT_PREY = 200,
#     INIT_PRED = 20,
#     NUM_FOOD = 300,
#     MAX_FOOD = 1000,
#     PREY_DEATH_FROM_PRED = 0.1,
#     PREY_ENERGY = 20,
#     PRED_ENERGY = 50,
#     PREY_STEP_ENERGY = 2,
#     PRED_STEP_ENERGY = 3,
#     PREY_ENERGY_FROM_FOOD = 3,
#     PRED_ENERGY_FROM_PREY = 10,
#     PREY_REPRODUCTION_THRESHOLD = 15,
#     PRED_REPRODUCTION_THRESHOLD = 40,
#     PREY_REPRODUCTION_CHANCE = 0.3,
#     PRED_REPRODUCTION_CHANCE = 0.1,
#     PREY_SPAWN_RATE = 0,
#     PRED_SPAWN_RATE = 0)


class estimate:
    def __init__(self, prey_data, pred_data, error_bound=10_000, success_bound=5_000):
        self.prey_data = prey_data
        self.pred_data = pred_data
        self.n = len(self.prey_data)
        self.x_0 = self.prey_data[0]
        self.y_0 = self.pred_data[0]
        self.error_bound = error_bound
        self.success_bound = success_bound

    def form_a_vectors(self):
        n = self.n
        X = np.empty((3, n - 1))
        l_x, l_y = np.empty((1, n - 1)), np.empty((1, n - 1))
        X[0] = np.ones_like(n)

        for j in range(n - 1):
            l_x[0][j] = np.log(self.prey_data[j+1]) - np.log(self.prey_data[j])
            l_y[0][j] = np.log(self.pred_data[j+1]) - np.log(self.pred_data[j])
            X[1][j] = (self.prey_data[j] + self.prey_data[j+1])/2
            X[2][j] = (self.pred_data[j] + self.pred_data[j+1])/2

        X_ = X.T
        base = np.matmul(np.linalg.inv(np.matmul(X, X_)), X)

        a_x = np.matmul(base, l_x.T)
        a_y = np.matmul(base, l_y.T)

        self.a_0, self.a_1, self.a_2 = a_x[0], a_x[1], a_x[2]
        self.a_3, self.a_4, self.a_5 = a_y[0], a_y[1], a_y[2]

    def derivative(self, X, t):
        x, y = X
        dotx = x * (self.a_0 + self.a_1*x + self.a_2*y)
        doty = y * (self.a_3 + self.a_4*x + self.a_5*y)
        return np.array([dotx, doty]).reshape(-1)
    
    def form_func(self):
        t = np.linspace(0., self.n, self.n*10)
        X0 = [self.x_0, self.y_0]
        res = integrate.odeint(self.derivative, X0, t)
        self.x, self.y = res.T

    def error(self):
        fake_prey = []
        fake_pred = []
        if np.any(np.isnan(self.x)) or np.any(np.isnan(self.y)):
            return self.error_bound
        for i in range(len(self.x)):
            if i % 10 == 0:
                fake_prey.append(self.x[i])
                fake_pred.append(self.y[i])
        return min(mse(self.prey_data, fake_prey),
                   mse(self.pred_data, fake_pred), 
                   self.success_bound)
    
    def get_mse(self):
        self.form_a_vectors()
        self.form_func()
        error = self.error()
        return error
    
    def graph(self):
        plt.figure()
        plt.grid()
        plt.title("Real data")
        plt.plot(np.arange(len(self.n)), self.prey_data, label="Prey")
        plt.plot(np.arange(len(self.n)), self.pred_data, label="Predator")
        plt.legend()
        plt.show()
        plt.clf()

        plt.figure()
        plt.grid()
        plt.title("LV method reconstruction")
        t = np.linspace(0., self.n, self.n*10)
        plt.plot(t, self.x, label="Prey")
        plt.plot(t, self.y, label="Predator")
        plt.legend()
        plt.show()

