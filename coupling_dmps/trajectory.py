import numpy as np
import matplotlib.pyplot as plt


Y = np.zeros((1001, 14))
T = np.linspace(0, 1, len(Y))
sigmoid = 0.5 * (np.tanh(1.5 * np.pi * (T - 0.5)) + 1.0)  # 双曲正切函数 sigmoid=0.5*(tanh(pi*(t-0.5)+1.0)
radius = 0.5
print(T)
print(type(sigmoid))
plt.plot(T, sigmoid, label="Sigmoid Curve")
plt.xlabel("T")
plt.ylabel("Sigmoid Value")
plt.legend()
plt.grid()
plt.show()