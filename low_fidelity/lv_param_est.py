import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

from low_fidelity.main import main, HyperParams

hp = HyperParams()
prey_data, pred_data = main(hp)

print(prey_data)
print(pred_data)

def form_a_vectors(prey_data, pred_data):
    assert len(prey_data) == len(pred_data), "datasets need same length"
    n = len(prey_data)

    X = np.empty((3, n - 1))
    l_x, l_y = np.empty((1, n - 1)), np.empty((1, n - 1))
    X[0] = np.ones_like(n)

    for j in range(n - 1):
        l_x[0][j] = np.log(prey_data[j+1]) - np.log(prey_data[j])
        l_y[0][j] = np.log(pred_data[j+1]) - np.log(pred_data[j])
        X[1][j] = (prey_data[j] + prey_data[j+1])/2
        X[2][j] = (pred_data[j] + pred_data[j+1])/2

    X_ = X.T
    base = np.matmul(np.linalg.inv(np.matmul(X, X_)), X)

    a_x = np.matmul(base, l_x.T)
    a_y = np.matmul(base, l_y.T)

    return a_x, a_y

x_0 = prey_data[0]
y_0 = pred_data[0]

a_x, a_y = form_a_vectors(prey_data=prey_data, pred_data=pred_data)

a_0, a_1, a_2 = a_x[0], a_x[1], a_x[2]
a_3, a_4, a_5 = a_y[0], a_y[1], a_y[2]

def derivative(X, t, a_0, a_1, a_2, a_3, a_4, a_5):
    x, y = X
    dotx = x * (a_0 + a_1*x + a_2*y)
    doty = y * (a_3 + a_4*x + a_5*y)
    return np.array([dotx, doty]).reshape(-1)

n = len(prey_data)

t = np.linspace(0., n, n*10)
X0 = [x_0, y_0]
res = integrate.odeint(derivative, X0, t, args = (a_0, a_1, a_2, a_3, a_4, a_5))
x, y = res.T

def error(real_prey, real_pred, synth_prey, synth_pred):
    fake_prey = []
    fake_pred = []
    n = len(real_prey)
    if np.any(np.isnan(synth_prey)) or np.any(np.isnan(synth_pred)):
        return np.inf, np.inf
    for i in range(len(synth_prey)):
        if i % 10 == 0:
            fake_prey.append(synth_prey[i])
            fake_pred.append(synth_pred[i])
    return mse(real_prey, fake_prey), mse(real_pred, fake_pred)

plt.figure()
plt.grid()
plt.title("Real data")
plt.plot(np.arange(len(prey_data)), prey_data, label="Prey")
plt.plot(np.arange(len(pred_data)), pred_data, label="Predator")
plt.legend()
plt.show()
plt.clf()

plt.figure()
plt.grid()
plt.title("LV method reconstruction")
plt.plot(t, x, label="Prey")
plt.plot(t, y, label="Predator")
plt.legend()
plt.show()

prey_error, pred_error = error(prey_data, pred_data, x, y)
print(f"Prey error is {prey_error}")
print(f"Pred error is {pred_error}")
