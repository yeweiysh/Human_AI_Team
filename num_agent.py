import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

def histPlot(data):
    mu = np.mean(data) # mean of distribution
    sigma = np.std(data) # standard deviation of distribution
    num_bins = 10
    # the histogram of the data
    n, bins, patches = plt.hist(data, num_bins, facecolor='blue', alpha=0.5)
    # plt.xlim(0.0, 3)
    # plt.ylim(0.0, 5)
    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'b-')


    plt.xlabel('number of agent consulted')
    plt.ylabel('PDF')
    plt.legend(loc='upper right')
    plt.show()
    plt.clf()


if __name__ == '__main__':
	num_agent=[3, 8, 3, 4, 4, 4, 1, 4, 7, 3, 4, 1, 4, 4, 6, 6, 3, 1, 4, 2, 7, 1, 9, 4, 4, 4, 0, 3, 6, 2]
	histPlot(num_agent)