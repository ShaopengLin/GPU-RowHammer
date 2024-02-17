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

print(len(y))

counts, bins = np.histogram(y)

plt.figure(figsize=(10, 10))
plt.tight_layout()

counts, edges, bars = plt.hist(y)
plt.bar_label(bars)
plt.xlabel('Time Spent for ld (ns)')
plt.ylabel('Amount of addresses (64 bytes per address)')
plt.title('Row Buffer Conflict Histogram')

plt.show()

for i in range(len(y)):
    if y[i] < 665:
        y[i] = 0

distances = []

# Count until reach > 0 number, and add distance traveled. Skip until next == 0
# to start counting again.
counter = 0
i = 0
while i != len(y):
    if y[i] == 0:
        counter += 1
        i += 1
    else:
        distances.append(counter)
        counter = 0
        i += 1
        while i != len(y) and y[i] != 0:
            i += 1
# distances = list(filter(lambda a: a != 8, distances))
print(distances)
plt.figure(figsize=(30,10))
plt.scatter(np.arange(len(distances)), distances, color='blue', alpha=0.7)
for i in range(len(distances)):
    plt.annotate(distances[i], (i, distances[i]))
plt.plot(np.arange(len(distances)), distances)
# Adding labels and title
plt.xlabel('Enumerated Conflicts (256 bytes each)')
plt.ylabel('Memory address offset (64 bytes each)')
plt.title('Distances Between Row Buffer Conflicts')

# Display the plot
plt.show()