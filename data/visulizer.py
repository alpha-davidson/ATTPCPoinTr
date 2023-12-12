import numpy as np
from matplotlib import pyplot as plt

def save_img(event):

    xs = event[:, 0]
    ys = event[:, 1]
    zs = event[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(xs, ys, zs)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.savefig('zSlice256.png')
    
    
    return


if __name__ == '__main__':

    file = 'data/Mg22-Ne20pp/simulated/512c/256p/2cut_scaled_Mg22_zsliced_size256_train.npy'

    f = np.load(file)

    save_img(f[0])