import matplotlib.pyplot as plt
import numpy as np
import sys

x = []
y = []

with open(sys.argv[1]) as f:
    for line in f:
        xt, yt = line.strip().split('\t')
        x.append(int(xt, 16))
        y.append(float(yt))
spike = []
for e in y:
    if e > 400:
        spike.append(e)
print(len(y))
print(len(spike))
print(100 * len(spike) / len(y))
counts, bins = np.histogram(y)

plt.figure(figsize=(10, 10))
plt.tight_layout()

counts, edges, bars = plt.hist(y)
plt.bar_label(bars)
plt.xlabel('Time Spent for ld (ns)')
plt.ylabel('Amount of addresses')
plt.title('Row Buffer Conflict Histogram')

plt.show()

plt.scatter(x, y, color='blue', alpha=0.7)

# Adding labels and title
plt.xlabel('Memory Addresses (offset)')
plt.ylabel('Fetch Time (ns)')
plt.title('Memory Addresses vs Fetch Time Scatter Plot')

# Display the plot
plt.show()