import numpy as np
from matplotlib import pyplot as plt
from random import randrange

def make_line(nPoints, nSliced, ySlope=1.5, zSlope=-0.5, yInt=0, zInt=0):

    '''
    Description:
        Generates a line of points in 3d space and then removes a sequential chunk
    Parameters:
        - nPoints: int, number of points in line
        - nSliced: int, number of points to be cut out
        - ySlope: float, slope in the y direction
        - zSlope: float, slope in the z direction
    Returns:
        - matrix: 2D array representing the points in the whole line
        - new_matrix: 2D array representing the points in the cut up line
    '''

    rng = np.random.default_rng()

    xs = rng.random(nPoints)
    xs = np.sort(xs)
    ys = ySlope*xs + yInt
    zs = zSlope*xs + zInt

    matrix = np.vstack((xs, ys, zs))
    matrix = np.transpose(matrix)

    cutoff = rng.random()
    slice_idx = (nPoints * cutoff)
    slice_idx = int(slice_idx)

    # Prevents intial nSliced and finial nSliced points from being sliced
    while slice_idx > (nPoints - nSliced) or slice_idx < nSliced:
        cutoff = rng.random()
        slice_idx = (nPoints * cutoff)//1
        slice_idx = int(slice_idx)

    last_idx = slice_idx + nSliced

    new_matrix = np.delete(matrix, np.s_[slice_idx:last_idx:], axis=0)
    assert new_matrix.shape == (nPoints - nSliced, 3)

    return new_matrix, matrix


def make_helix(nPoints, nSliced, noisiness=0.1):

    totPoints = 3*(2*nPoints-nSliced)

    rng = np.random.default_rng()

    noise = np.random.default_rng().normal(0, 1, totPoints)
    noise *= noisiness

    zs = rng.random(nPoints)
    zs = np.sort(zs)
    zs = 4*np.pi*zs
    xs = np.cos(zs)
    ys = np.sin(zs)

    matrix = np.vstack((xs, ys, zs))
    matrix = np.transpose(matrix)

    cutoff = rng.random()
    slice_idx = nPoints * cutoff
    slice_idx = int(slice_idx)

    while slice_idx > (nPoints - nSliced) or slice_idx < nSliced:
        cutoff = rng.random()
        slice_idx = nPoints * cutoff
        slice_idx = int(slice_idx)

    last_idx = slice_idx + nSliced

    new_matrix = np.delete(matrix, np.s_[slice_idx:last_idx:], axis=0)
    assert new_matrix.shape == (nPoints - nSliced, 3)

    for i in range(nPoints):
        for j in range(3):
            matrix[i, j] += noise[randrange(totPoints)]

    for i in range(nPoints-nSliced):
        for j in range(3):
            new_matrix[i, j] += noise[randrange(totPoints)]


    return new_matrix, matrix


def visualize(matrix=None, file=None, name=None, event=0):

    if file != None and matrix != None:
        print('Error: Too many point clouds provided!')
        exit(1)    
    elif file != None:
        f = np.load(file)
        xs = f[event, :, 0]
        ys = f[event, :, 1]
        zs = f[event, :, 2]
    else:
        xs = matrix[:, 0]
        ys = matrix[:, 1]
        zs = matrix[:, 2]
    # elif file == None and matrix == None:
    #     print('Error: Too few point clouds provided!')
    #     exit(1)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(xs, ys, zs)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if file != None and name == None:
        plt.savefig(file[:-4]+'.png')
    else:
        plt.savefig(str(name)+'.png')

def make_dataset(nTrain=3000, nTest=1000, nVal=1000, nPoints=512, nSliced=128):

    with open('train.txt', 'w') as txt:

        for i in range(nTrain):
            partial, complete = make_line(nPoints, nSliced, ySlope=0.3)
            i_str = str(i).zfill(4)
            with open('./partial/line-'+i_str+'.npy', 'wb') as f:
                np.save(f, partial)
            with open('./complete/line-'+i_str+'.npy', 'wb') as f:
                np.save(f, complete)
            txt.write('line-'+i_str+'.npy\n')

    with open('test.txt', 'w') as txt:

        for i in range(nTrain, nTrain+nTest):
            partial, complete = make_line(nPoints, nSliced, ySlope=0.3)
            with open('./partial/line-'+str(i)+'.npy', 'wb') as f:
                np.save(f, partial)
            with open('./complete/line-'+str(i)+'.npy', 'wb') as f:
                np.save(f, complete)
            txt.write('line-'+str(i)+'.npy\n')

    with open('val.txt', 'w') as txt:

        for i in range(nTrain+nTest, nTrain+nTest+nVal):
            partial, complete = make_line(nPoints, nSliced, ySlope=0.3)
            with open('./partial/line-'+str(i)+'.npy', 'wb') as f:
                np.save(f, partial)
            with open('./complete/line-'+str(i)+'.npy', 'wb') as f:
                np.save(f, complete)
            txt.write('line-'+str(i)+'.npy\n')


def total():

    with open('./total/lines-partial-train.npy', 'wb') as file:

        mat = np.load('./partial/line-0000.npy')
        mat = np.resize(mat, (1, 384, 3))

        for i in range(1, 3000):
            i_str = str(i).zfill(4)
            f = np.load('./partial/line-'+i_str+'.npy')
            f = np.resize(f, (1, 384, 3))
            mat = np.vstack([mat, f])

        assert mat.shape == (3000, 384, 3)

        np.save(file, mat)
    
    with open('./total/lines-partial-test.npy', 'wb') as file:

        mat = np.load('./partial/line-3000.npy')
        mat = np.resize(mat, (1, 384, 3))

        for i in range(3001, 4000):
            i_str = str(i).zfill(4)
            f = np.load('./partial/line-'+i_str+'.npy')
            f = np.resize(f, (1, 384, 3))
            mat = np.vstack([mat, f])

        assert mat.shape == (1000, 384, 3)

        np.save(file, mat)

    with open('./total/lines-partial-val.npy', 'wb') as file:

        mat = np.load('./partial/line-4000.npy')
        mat = np.resize(mat, (1, 384, 3))

        for i in range(4001, 5000):
            i_str = str(i).zfill(4)
            f = np.load('./partial/line-'+i_str+'.npy')
            f = np.resize(f, (1, 384, 3))
            mat = np.vstack([mat, f])

        assert mat.shape == (1000, 384, 3)

        np.save(file, mat)

    with open('./total/lines-complete-train.npy', 'wb') as file:

        mat = np.load('./complete/line-0000.npy')
        mat = np.resize(mat, (1, 512, 3))

        for i in range(1, 3000):
            i_str = str(i).zfill(4)
            f = np.load('./complete/line-'+i_str+'.npy')
            f = np.resize(f, (1, 512, 3))
            mat = np.vstack([mat, f])

        assert mat.shape == (3000, 512, 3)

        np.save(file, mat)
    
    with open('./total/lines-complete-test.npy', 'wb') as file:

        mat = np.load('./complete/line-3000.npy')
        mat = np.resize(mat, (1, 512, 3))

        for i in range(3001, 4000):
            i_str = str(i).zfill(4)
            f = np.load('./complete/line-'+i_str+'.npy')
            f = np.resize(f, (1, 512, 3))
            mat = np.vstack([mat, f])

        assert mat.shape == (1000, 512, 3)

        np.save(file, mat)

    with open('./total/lines-complete-val.npy', 'wb') as file:

        mat = np.load('./complete/line-4000.npy')
        mat = np.resize(mat, (1, 512, 3))

        for i in range(4001, 5000):
            i_str = str(i).zfill(4)
            f = np.load('./complete/line-'+i_str+'.npy')
            f = np.resize(f, (1, 512, 3))
            mat = np.vstack([mat, f])

        assert mat.shape == (1000, 512, 3)

        np.save(file, mat)


def make_helix_dataset(nTrain=3000, nTest=1000, nVal=1000, nPoints=512, nSliced=128):

    pointsINpartial = nPoints - nSliced

    partial, complete = make_helix(nPoints, nSliced)
    partial_stack = np.resize(partial, (1, pointsINpartial, 3))
    complete_stack = np.resize(complete, (1, nPoints, 3))

    for i in range(1, nTrain):
        partial, complete = make_helix(nPoints, nSliced)
        partial = np.resize(partial, (1, pointsINpartial, 3))
        complete = np.resize(complete, (1, nPoints, 3))

        partial_stack = np.vstack([partial_stack, partial])
        complete_stack = np.vstack([complete_stack, complete])

    assert partial_stack.shape == (nTrain, pointsINpartial, 3)
    assert complete_stack.shape == (nTrain, nPoints, 3)

    with open('./noisy/noisy-helix-partial-train.npy', 'wb') as file:
        np.save(file, partial_stack)

    with open('./noisy/noisy-helix-complete-train.npy', 'wb') as file:
        np.save(file, complete_stack)

    
    partial, complete = make_helix(nPoints, nSliced)
    partial_stack = np.resize(partial, (1, pointsINpartial, 3))
    complete_stack = np.resize(complete, (1, nPoints, 3))

    for i in range(1, nTest):
        partial, complete = make_helix(nPoints, nSliced)
        partial = np.resize(partial, (1, pointsINpartial, 3))
        complete = np.resize(complete, (1, nPoints, 3))

        partial_stack = np.vstack([partial_stack, partial])
        complete_stack = np.vstack([complete_stack, complete])

    assert partial_stack.shape == (nTest, pointsINpartial, 3)
    assert complete_stack.shape == (nTest, nPoints, 3)

    with open('./noisy/noisy-helix-partial-test.npy', 'wb') as file:
        np.save(file, partial_stack)

    with open('./noisy/noisy-helix-complete-test.npy', 'wb') as file:
        np.save(file, complete_stack)



    partial, complete = make_helix(nPoints, nSliced)
    partial_stack = np.resize(partial, (1, pointsINpartial, 3))
    complete_stack = np.resize(complete, (1, nPoints, 3))

    for i in range(1, nVal):
        partial, complete = make_helix(nPoints, nSliced)
        partial = np.resize(partial, (1, pointsINpartial, 3))
        complete = np.resize(complete, (1, nPoints, 3))

        partial_stack = np.vstack([partial_stack, partial])
        complete_stack = np.vstack([complete_stack, complete])

    assert partial_stack.shape == (nVal, pointsINpartial, 3)
    assert complete_stack.shape == (nVal, nPoints, 3)

    with open('./noisy/noisy-helix-partial-val.npy', 'wb') as file:
        np.save(file, partial_stack)

    with open('./noisy/noisy-helix-complete-val.npy', 'wb') as file:
        np.save(file, complete_stack)



if __name__ == '__main__':

    make_helix_dataset(nPoints=128, nSliced=32)