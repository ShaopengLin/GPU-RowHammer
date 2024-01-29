import matplotlib.pyplot as plt

FNAME = "stride_latencies.txt"

if __name__ == '__main__':
    hist = {}
    with open(FNAME, "r") as f:
        lines = f.readlines()
        for line in lines:
            data = line.split(" ")
            latency = float(data[1])

            if latency // 10 in hist:
                hist[latency // 10] += 1
            else:
                hist[latency // 10] = 1
    print(hist)
    plt.figure()
    plt.bar(hist.keys(), hist.values())
    plt.title(f'Histogram of frequency of access latencies')
    plt.xlabel('latencies')
    plt.ylabel('# accesses')
    plt.show()
    # plt.savefig(f'hist_same_ct.png')  # Create a new figure
