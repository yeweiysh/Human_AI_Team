import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

if __name__ == '__main__':
    
    w = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    cent_loss_mean = [0.69, 0.65, 0.61, 0.57, 0.53, 0.49, 0.44, 0.40, 0.38, 0.38, 0.40, ]
    cent_loss_var = [0.22, 0.21, 0.19, 0.18, 0.17, 0.17, 0.17, 0.18, 0.19, 0.20, 0.24]
    
    w = np.asarray(w)
    cent_loss_mean = np.asarray(cent_loss_mean)
    cent_loss_var = np.asarray(cent_loss_var)
    
    plt.plot(w, cent_loss_mean, lw=2, color='blue')
    plt.fill_between(w, cent_loss_mean+cent_loss_var, cent_loss_mean-cent_loss_var, facecolor='blue', alpha=0.5)
    # ax.legend(loc='upper left')
    plt.xlabel('w')
    plt.ylabel('Loss')
    # ax.grid()
    plt.show()