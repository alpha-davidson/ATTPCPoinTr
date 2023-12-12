
import torch
import h5py
import numpy as np
# import MinkowskiEngine as ME


class O16Dataset(torch.utils.data.Dataset):
    def __init__(self, fold, quantization_size=1.0):
        """ Loads the underlying numpy array. """
        assert (fold == 'train') or (fold == 'test')
        super().__init__()

        self.fold = fold
        self.quantization_size = quantization_size

        # indices: x, y, z, amplitude, event index, label, event length
        # only loading event index and labels
        data = np.load(f'O16_{fold}.npy')[:,0,4:6].astype(int)

        # processing labels
        self.labels = np.array(list(map(self._transform_label, data[:, 1])))
        
        # loading and processing events (i.e., features)
        self.evt_ids = data[:, 0]
        group_names = [f'Event_[{i}]' for i in self.evt_ids]
        self.events = []

        # load all points in events for which we have labels
        with h5py.File('O16_run160.h5') as f:
            for key in group_names:
                xs, ys, zs, As = self._transform_features(xs=f[key]['x'].astype('float32'),
                                                          ys=f[key]['y'].astype('float32'),
                                                          zs=f[key]['z'].astype('float32'),
                                                          As = f[key]['A'].astype('float32'))
                self.events.append(list(zip(xs, ys, zs, As)))

    def _transform_features(self, xs, ys, zs, As):
        RANGES = {
            'MIN_X': -270.0,
            'MAX_X': 270.0,
            'MIN_Y': -270.0,
            'MAX_Y': 270.0,
            'MIN_Z': -185.0,
            'MAX_Z': 1155.0,
            'MIN_LOG_A': 0.0,
            'MAX_LOG_A': 8.60
        }
        # xs = (xs - RANGES['MIN_X'])/(RANGES['MAX_X'] - RANGES['MIN_X'])
        # ys = (ys - RANGES['MIN_Y'])/(RANGES['MAX_Y'] - RANGES['MIN_Y'])
        # zs = (zs - RANGES['MIN_Z'])/(RANGES['MAX_Z'] - RANGES['MIN_Z'])
        As = (np.log(As) - RANGES['MIN_LOG_A'])/(RANGES['MAX_LOG_A'] - RANGES['MIN_LOG_A'])
        return xs, ys, zs, As

    def _transform_label(self, label):
        if label <= 2:
            return 0
        elif label == 3:
            return 1
        else:
            return 2

    def __getitem__(self, i):
        """ Return i^th event. """
        xyza = np.array(self.events[i])
        xyz = xyza[:, :3]
        a = xyza[:, 3].reshape(-1, 1)
        label = self.labels[i]

        coords, features = ME.utils.sparse_quantize(coordinates=xyz, 
                                                    features=a,
                                                    quantization_size=self.quantization_size)
        return coords, features, label

    def __len__(self):
        return len(self.events)


if __name__ == '__main__':
    train_data = O16Dataset('train')
    print(train_data[0])

    test_data = O16Dataset('test')
    print(test_data[0])