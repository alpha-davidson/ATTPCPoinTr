import numpy as np
import os


def cut_event(origin, event, k, include_q=True):
    '''
    Cuts the k-nearest points to an origin in an event
    '''

    dists = np.ndarray(len(event))

    for i, point in enumerate(event):
        d = np.sqrt( np.square(point[0] - origin[0]) + np.square(point[1] - origin[1]) )
        dists[i] = d

    idxs = np.argsort(dists)
    cut = event[idxs[k:]]

    feat_lim = 4 if include_q else 3

    if cut.shape[-1] > feat_lim:
        return cut[:, :feat_lim]
    elif cut.shape[-1] == feat_lim:
        return cut
    else:
        raise AttributeError("Missing x, y, z, or q")


if __name__ == '__main__':

    NUM_POINTS_COMPLETE = 512
    NUM_TO_CUT = 256
    ORIGIN = (0, 0)

    assert (NUM_POINTS_COMPLETE - NUM_TO_CUT) > 0, "Would result in 0 or fewer points"
    assert (ORIGIN[0] >= -270 and ORIGIN[0] <= 270), "Origin x value out of bounds"
    assert (ORIGIN[1] >= -270 and ORIGIN[1] <= 270), "Origin y value out of bounds"

    in_root = os.getcwd() + f'/data/Mg22-Ne20pp/simulated/{NUM_POINTS_COMPLETE}c/Mg22_size512_convertXYZQ_'
    out_root = os.getcwd() + f'/data/Mg22-Ne20pp/simulated/{NUM_POINTS_COMPLETE}c/{NUM_POINTS_COMPLETE - NUM_TO_CUT}p/center_cut_Mg22_'
    stems = ['train.npy', 'test.npy', 'val.npy']

    for stem in stems:

        dset = np.load(in_root+stem)
        assert dset.shape[1] == NUM_POINTS_COMPLETE, 'Make sure to update NUM_POINTS_COMPLETE'
        
        cut_dset = np.ndarray((len(dset), NUM_POINTS_COMPLETE-NUM_TO_CUT, 4))
        
        for i, event in enumerate(dset):
            cut_dset[i] = cut_event(ORIGIN, event, NUM_TO_CUT)

        np.save(out_root+stem, cut_dset)