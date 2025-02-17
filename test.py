import numpy as np
start_y = np.array([0, 0], dtype=float)
print(start_y)
print(type(start_y))
print(start_y[0])
print(type(start_y[0]))
print(type([0, 0]))
print([0, 0][0])
print(type([0, 0][0]))

random_state = np.random.RandomState(1)
print(random_state)

n_rows, n_cols = 2, 4
n_subplots = n_rows * n_cols
x_range = -0.2, 1.2
y_range = -0.2, 1.2

position = np.copy(start_y)
positions = [position]  # [[0., 0.]]
path = np.array(positions)
path1 = np.array(position)
pass