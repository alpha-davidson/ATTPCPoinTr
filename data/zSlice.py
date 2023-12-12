import numpy as np
import os

def restructure_and_sort(event):

    dtype = [('x', float), ('y', float), ('z', float)]

    restructured = np.empty((len(event), ), dtype=dtype)
    xs = event[:, 0]
    ys = event[:, 1]
    zs = event[:, 2]

    for i in range(len(event)):
        restructured[i] = (xs[i], ys[i], zs[i])

    restructured = np.sort(restructured, order='z')

    return restructured


def z_slice_event(event, nSliced, slices):

    nSliced //= slices

    while slices != 0:

        rng = np.random.default_rng()

        slice_idx = int(rng.random() * len(event))

        while slice_idx > (len(event) - nSliced) or slice_idx < nSliced:
            slice_idx = int(rng.random() * len(event))

        last_idx = slice_idx + nSliced

        sliced_event = np.delete(event, np.s_[slice_idx:last_idx:], axis=0)

        slices -= 1

    np.random.default_rng().shuffle(sliced_event)

    return sliced_event


def revert_structure(event):

    reverted = np.empty((1, len(event), 3))

    for i in range(len(event)):
        reverted[0, i, 0] = event[i]['x']
        reverted[0, i, 1] = event[i]['y']
        reverted[0, i, 2] = event[i]['z']

    return reverted



if __name__ == '__main__':
    
    root = '/home/DAVIDSON/bewagner/summer2023/ATTPCPoinTr/data/'

    files = [root+'Mg22-Ne20pp/simulated/512c/scaled_Mg22_size512_train.npy',
             root+'Mg22-Ne20pp/simulated/512c/scaled_Mg22_size512_test.npy',
             root+'Mg22-Ne20pp/simulated/512c/scaled_Mg22_size512_val.npy']
    
    nSliced = 256
    slices = 2
    
    for file in files:

        f = np.load(file)

        cut_file = revert_structure(z_slice_event(restructure_and_sort(f[0]), nSliced, slices))

        for i in range(1, len(f)):
            cut_event = restructure_and_sort(f[i])
            cut_event = z_slice_event(cut_event, nSliced, slices)
            cut_event = revert_structure(cut_event)
            cut_file = np.vstack((cut_file, cut_event))

        if 'test' in file:
            split = 'test.npy'
        elif 'train' in file:
            split = 'train.npy'
        elif 'val' in file:
            split = 'val.npy'
        else:
            assert False, 'train, test, or val not in filename'

        np.save(root+'Mg22-Ne20pp/simulated/512c/256p/2cut_scaled_Mg22_zsliced_size256_'+split, cut_file)

        
