import numpy as np
import os


def get_qs(file):

    empty = []

    for i, values in enumerate(file):
        empty.append((i, abs(values[3])))

    return empty


def get_idxs(keepers, nFinal):

    idx_arr = np.empty((nFinal,), int)

    for i in range(nFinal):
        idx_arr[i] = keepers[i][0]

    return idx_arr


def get_xyzs(idx_arr, event, test, nFinal):

    if test:
        new_evt = np.empty((nFinal, 8))
    else:
        new_evt = np.empty((nFinal, 7))

    for i, idx in enumerate(idx_arr):

        new_evt[i] = event[idx]

    if test:
        new_evt = np.reshape(new_evt, (1, nFinal, 8))
    else:
        new_evt = np.reshape(new_evt, (1, nFinal, 7))

    return new_evt


def get_sliced_event(f, event_idx, nFinal, nInit, test=False):

    empty = get_qs(f[event_idx])

    dtype = [('index', int), ('charge', float)]

    arr = np.array(empty, dtype=dtype)

    arr = np.sort(arr, order='charge')

    keepers = arr[nInit-nFinal:]

    idx_arr = get_idxs(keepers, nFinal)

    sliced = get_xyzs(idx_arr, f[event_idx], test, nFinal)

    return sliced
    


if __name__ == '__main__':

    root = 'data/Mg22-Ne20pp/data/'

    files = [root+'Mg22_size128_convertXYZQ_train.npy',
             root+'Mg22_size128_convertXYZQ_test.npy',
             root+'Mg22_size128_convertXYZQ_val.npy']
    
    nInit = 128
    nFinal = 32

    for file in files:

        if 'test' in file:
            test = True
            split = 'test.npy'
        else:
            test = False

        if 'train' in file:
            split = 'train.npy'
        elif 'val' in file:
            split = 'val.npy'

        f = np.load(file)

        sliced = get_sliced_event(f, 0, nFinal, nInit, test=test)

        for i in range(1, len(f)):

            next_event = get_sliced_event(f, i, nFinal, nInit, test=test)
            sliced = np.vstack((sliced, next_event))

        middle = '/%dp/qSliced_128Mg22_size%d_' % (nFinal, nFinal)
        new_file = root + middle + split
        np.save(new_file, sliced)