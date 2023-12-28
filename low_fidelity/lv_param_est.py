import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

from low_fidelity.main import main

# prey_data = [200, 190, 181, 168, 164, 160, 154, 151, 142, 127, 118, 114, 103, 95, 83, 79, 75, 74, 64, 64, 56, 48, 48, 46, 43, 31, 30, 30, 27, 25, 22, 19, 17, 13, 15, 11, 12, 9, 9, 8, 8, 8, 7, 7, 7, 5, 3, 5, 4, 7, 6, 4, 4, 5, 5, 5, 7, 6, 6, 6, 10, 15, 21, 22, 25, 27, 29, 34, 40, 56, 64, 73, 80, 84, 87, 89, 93, 102, 107, 117, 125, 127, 130, 128, 129, 134, 141, 140, 146, 146, 134, 125, 124, 117, 98, 87, 79, 72, 64, 49, 43, 31, 21, 17, 20, 17, 17, 14, 12, 11, 11, 9, 8, 12, 12, 7, 6, 5, 2, 1, 1, 1, 1, 2, 2, 3, 5, 4, 4, 4, 4, 5, 6, 8, 9, 8, 9, 10, 15, 14, 20, 22, 25, 29, 33, 37, 42, 45, 50, 51, 60, 75, 99, 110, 112, 121, 124, 137, 142, 150, 152, 155, 158, 154, 149, 154, 157, 153, 160, 149, 127, 114, 105, 86, 68, 58, 49, 36, 35, 26, 21, 18, 14, 11, 8, 5, 3, 1, 2, 1, 2, 1, 1, 1]
# pred_data = [20, 20, 22, 27, 31, 36, 41, 43, 51, 62, 68, 72, 77, 84, 90, 92, 94, 97, 103, 104, 110, 116, 117, 119, 124, 128, 129, 130, 132, 133, 136, 136, 138, 142, 140, 137, 134, 135, 132, 126, 126, 123, 120, 120, 117, 117, 112, 107, 104, 98, 93, 90, 87, 85, 83, 75, 73, 71, 66, 65, 60, 56, 52, 49, 37, 37, 33, 32, 32, 32, 33, 33, 34, 33, 36, 35, 39, 40, 45, 55, 58, 62, 68, 73, 76, 79, 83, 94, 101, 106, 117, 126, 130, 140, 155, 164, 170, 176, 181, 187, 191, 198, 204, 207, 210, 214, 213, 213, 210, 209, 208, 208, 208, 207, 203, 202, 197, 194, 190, 184, 177, 171, 163, 151, 139, 135, 130, 123, 120, 118, 116, 114, 108, 88, 83, 80, 75, 70, 60, 56, 52, 52, 50, 47, 44, 42, 39, 33, 34, 34, 35, 35, 37, 39, 41, 43, 51, 55, 60, 64, 71, 75, 86, 97, 107, 120, 131, 138, 143, 150, 163, 178, 188, 203, 217, 223, 227, 232, 236, 237, 242, 242, 248, 248, 249, 250, 251, 252, 246, 243, 241, 235, 225, 222]

prey_data, pred_data = main(200, 20, 20, 200, 20)

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
X0 = [200, 20]
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
