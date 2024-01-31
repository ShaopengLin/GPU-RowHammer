import matplotlib.pyplot as plt
import numpy as np

x = []
y = []
with open("./out/build/o16") as f:
    for line in f:
        xt, yt = line.strip().split('\t')
        x.append(int(xt, 16))
        y.append(float(yt))

counts, bins = np.histogram(y)
plt.stairs(counts, bins)
plt.show()