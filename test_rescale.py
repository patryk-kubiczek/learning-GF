from learning import rescale, back_rescale
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt, tanh

alpha =  1 

x = np.linspace(0, 1, num=1001)

def f(x):
    rho = 0.5 * (1 - tanh(alpha))
    return (-rho + 0.5 * (tanh(alpha * (2 * x - 1)) + 1)) / (1 - 2 * rho)

plt.plot(x, [f(xx) for xx in x])
plt.show()
plt.close()

data = back_rescale([sqrt(1 - (2 * (xx - 0.5))**2) for xx in x], 2)
plt.plot(x, data)

data_rescaled = rescale(data, alpha)
plt.plot(x, data_rescaled)

data_back = back_rescale(data_rescaled, alpha)
plt.plot(x, data_back)
plt.legend()
plt.show()
