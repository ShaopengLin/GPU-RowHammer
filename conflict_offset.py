import matplotlib.pyplot as plt
import numpy as np
import sys

x = []

with open(sys.argv[1]) as f:
    for line in f:
        xt = line.strip()
        x.append(int(xt, 16))

distances = [1]
prev = x[0]
for value in x[1:]:
    if value == prev:
        distances[-1] += 1
    else:
        distances.append(1)
        prev = value
plt.scatter(
    np.arange(len(distances)),
    distances,
    color=["red", "blue"] * ((len(distances) // 2)) + ["red"],
    alpha=0.7,
)
plt.plot(np.arange(len(distances)), distances)
plt.xlabel("Hit(Blue) or Conflict(Red)")
plt.ylabel("Count")
plt.title("Plot of The Hit and Conflicts")
plt.show()
