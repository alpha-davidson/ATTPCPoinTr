import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import abc
from pointnet2_ops import pointnet2_utils

def jitter_points(pc, std=0.01, clip=0.05):
    bsize = pc.size()[0]
    for i in range(bsize):
        jittered_data = pc.new(pc.size(1), 3).normal_(
            mean=0.0, std=std
        ).clamp_(-clip, clip)
        pc[i, :, 0:3] += jittered_data
    return pc

def random_sample(data, number):
    '''
        data B N 3
        number int
    '''
    assert data.size(1) > number
    assert len(data.shape) == 3
    ind = torch.multinomial(torch.rand(data.size()[:2]).float(), number).to(data.device)
    data = torch.gather(data, 1, ind.unsqueeze(-1).expand(-1, -1, data.size(-1)))
    return data

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def build_lambda_sche(opti, config, last_epoch=-1):
    if config.get('decay_step') is not None:
        # lr_lbmd = lambda e: max(config.lr_decay ** (e / config.decay_step), config.lowest_decay)
        warming_up_t = getattr(config, 'warmingup_e', 0)
        lr_lbmd = lambda e: max(config.lr_decay ** ((e - warming_up_t) / config.decay_step), config.lowest_decay) if e >= warming_up_t else max(e / warming_up_t, 0.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opti, lr_lbmd, last_epoch=last_epoch)
    else:
        raise NotImplementedError()
    return scheduler

def build_lambda_bnsche(model, config, last_epoch=-1):
    if config.get('decay_step') is not None:
        bnm_lmbd = lambda e: max(config.bn_momentum * config.bn_decay ** (e / config.decay_step), config.lowest_decay)
        bnm_scheduler = BNMomentumScheduler(model, bnm_lmbd, last_epoch=last_epoch)
    else:
        raise NotImplementedError()
    return bnm_scheduler
    
def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.

    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.
    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.
    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum
    return fn

class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def get_momentum(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        return self.lmbd(epoch)



def seprate_point_cloud(xyz, num_points, crop, fixed_points = None, padding_zeros = False):
    '''
     seprate point cloud: usage : using to generate the incomplete point cloud with a setted number.
    '''
    _,n,c = xyz.shape

    assert n == num_points
    assert c == 3
    if crop == num_points:
        return xyz, None
        
    INPUT = []
    CROP = []
    for points in xyz:
        if isinstance(crop,list):
            num_crop = random.randint(crop[0],crop[1])
        else:
            num_crop = crop

        points = points.unsqueeze(0)

        if fixed_points is None:       
            center = F.normalize(torch.randn(1,1,3),p=2,dim=-1).cuda()
        else:
            if isinstance(fixed_points,list):
                fixed_point = random.sample(fixed_points,1)[0]
            else:
                fixed_point = fixed_points
            center = fixed_point.reshape(1,1,3).cuda()

        distance_matrix = torch.norm(center.unsqueeze(2) - points.unsqueeze(1), p =2 ,dim = -1)  # 1 1 2048

        idx = torch.argsort(distance_matrix,dim=-1, descending=False)[0,0] # 2048

        if padding_zeros:
            input_data = points.clone()
            input_data[0, idx[:num_crop]] =  input_data[0,idx[:num_crop]] * 0

        else:
            input_data = points.clone()[0, idx[num_crop:]].unsqueeze(0) # 1 N 3

        crop_data =  points.clone()[0, idx[:num_crop]].unsqueeze(0)

        if isinstance(crop,list):
            INPUT.append(fps(input_data,2048))
            CROP.append(fps(crop_data,2048))
        else:
            INPUT.append(input_data)
            CROP.append(crop_data)

    input_data = torch.cat(INPUT,dim=0)# B N 3
    crop_data = torch.cat(CROP,dim=0)# B M 3

    return input_data.contiguous(), crop_data.contiguous()

def get_ptcloud_img(ptcloud):
    fig = plt.figure(figsize=(8, 8))

    # x, z, y = ptcloud.transpose(1, 0)
    x = ptcloud[:, 0]
    y = ptcloud[:, 1]
    z = ptcloud[:, 2]
    try:
        ax = fig.gca(projection=Axes3D.name, adjustable='box')
    except:
        ax = fig.add_subplot(projection=Axes3D.name, adjustable='box')
    # ax.axis('off')
    # ax.axis('scaled')
    # ax.view_init(30, 45)
    maximum, minimum = np.max(ptcloud), np.min(ptcloud)
    ax.set_xbound(minimum, maximum)
    ax.set_ybound(minimum, maximum)
    ax.set_zbound(minimum, maximum)
    ax.scatter(x, y, z, zdir='z')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    plt.close()

    return img



def visualize_KITTI(path, data_list, titles = ['input','pred'], cmap=['bwr','autumn'], zdir='y', 
                         xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1) ):
    fig = plt.figure(figsize=(6*len(data_list),6))
    cmax = data_list[-1][:,0].max()

    for i in range(len(data_list)):
        data = data_list[i][:-2048] if i == 1 else data_list[i]
        color = data[:,0] /cmax
        ax = fig.add_subplot(1, len(data_list) , i + 1, projection='3d')
        ax.view_init(30, -120)
        b = ax.scatter(data[:, 0], data[:, 1], data[:, 2], zdir=zdir, c=color,vmin=-1,vmax=1 ,cmap = cmap[0],s=4,linewidth=0.05, edgecolors = 'black')
        ax.set_title(titles[i])

        ax.set_axis_off()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.2, hspace=0)
    if not os.path.exists(path):
        os.makedirs(path)

    pic_path = path + '.png'
    fig.savefig(pic_path)

    np.save(os.path.join(path, 'input.npy'), data_list[0].numpy())
    np.save(os.path.join(path, 'pred.npy'), data_list[1].numpy())
    plt.close(fig)


def random_dropping(pc, e):
    up_num = max(64, 768 // (e//50 + 1))
    pc = pc
    random_num = torch.randint(1, up_num, (1,1))[0,0]
    pc = fps(pc, random_num)
    padding = torch.zeros(pc.size(0), 2048 - pc.size(1), 3).to(pc.device)
    pc = torch.cat([pc, padding], dim = 1)
    return pc
    

def random_scale(partial, gt, scale_range=[0.8, 1.2]):
    scale = torch.rand(1).cuda() * (scale_range[1] - scale_range[0]) + scale_range[0]
    return partial * scale, gt * scale



from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


def better_img(event):

    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(projection='3d')

    xs = event[0, :, 0]
    ys = event[0, :, 1]
    zs = event[0, :, 2]

    RANGES = {
            'MIN_X': -270.0,
            'MAX_X': 270.0,
            'MIN_Y': -270.0,
            'MAX_Y': 270.0,
            'MIN_Z': -185.0,
            'MAX_Z': 1155.0
        }
    xs = xs * (RANGES['MAX_X'] - RANGES['MIN_X']) + RANGES['MIN_X']
    ys = ys * (RANGES['MAX_Y'] - RANGES['MIN_Y']) + RANGES['MIN_Y']
    zs = zs * (RANGES['MAX_Z'] - RANGES['MIN_Z']) + RANGES['MIN_Z']

    ax.scatter(xs, zs, ys)

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')

    ax.set_xlim(xmin=RANGES['MIN_X'], xmax=RANGES['MAX_X'])
    ax.set_ylim(ymin=RANGES['MIN_Z'], ymax=RANGES['MAX_Z'])
    ax.set_zlim(zmin=RANGES['MIN_Y'], zmax=RANGES['MAX_Y'])

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    plt.close()

    return img


def get_extremes(event):

    maxes = []
    mins = []

    for i in range(3):
        maxes.append(np.max(event[:, i]))
        mins.append(np.min(event[:, i]))

    return maxes, mins


def rescale_feats(xs, ys, zs):
    RANGES = {
            'MIN_X': -270.0,
            'MAX_X': 270.0,
            'MIN_Y': -270.0,
            'MAX_Y': 270.0,
            'MIN_Z': -185.0,
            'MAX_Z': 1155.0
        }
    xs = xs * (RANGES['MAX_X'] - RANGES['MIN_X']) + RANGES['MIN_X']
    ys = ys * (RANGES['MAX_Y'] - RANGES['MIN_Y']) + RANGES['MIN_Y']
    zs = zs * (RANGES['MAX_Z'] - RANGES['MIN_Z']) + RANGES['MIN_Z']

    return xs, ys, zs


def triplet_img(input_pc, output_pc, gt_pc, idx, path, cfg):

    assert os.getcwd() == '/home/DAVIDSON/bewagner/summer2023/ATTPCPoinTr', f'Current Directory == {os.getcwd()}'

    RANGES = {
            'MIN_X': -270.0,
            'MAX_X': 270.0,
            'MIN_Y': -270.0,
            'MAX_Y': 270.0,
            'MIN_Z': -185.0,
            'MAX_Z': 1155.0
        }

    fig, (input_ax, output_ax, gt_ax) = plt.subplots(1, 3, figsize=(12,6), subplot_kw=dict(projection='3d'))

    input_xs, input_ys, input_zs = rescale_feats(input_pc[:, 0], input_pc[:, 1], input_pc[:, 2])
    output_xs, output_ys, output_zs = rescale_feats(output_pc[:, 0], output_pc[:, 1], output_pc[:, 2])
    gt_xs, gt_ys, gt_zs = rescale_feats(gt_pc[:, 0], gt_pc[:, 1], gt_pc[:, 2])

    input_ax.scatter(input_xs, input_zs, input_ys, s=4)
    output_ax.scatter(output_xs, output_zs, output_ys, s=4)
    gt_ax.scatter(gt_xs, gt_zs, gt_ys, s=4)

    # axes = [input_ax, output_ax, gt_ax]

    # for ax in axes:
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Z')
    #     ax.set_zlabel('Y')

    #     ax.set_xlim(xmin=RANGES['MIN_X'], xmax=RANGES['MAX_X'])
    #     ax.set_ylim(ymin=RANGES['MIN_Z'], ymax=RANGES['MAX_Z'])
    #     ax.set_zlim(zmin=RANGES['MIN_Y'], zmax=RANGES['MAX_Y'])

    input_ax.set_xlabel('X')
    input_ax.set_ylabel('Z')
    input_ax.set_zlabel('Y')

    output_ax.set_xlabel('X')
    output_ax.set_ylabel('Z')
    output_ax.set_zlabel('Y')

    gt_ax.set_xlabel('X')
    gt_ax.set_ylabel('Z')
    gt_ax.set_zlabel('Y')

    input_ax.set_xlim(xmin=RANGES['MIN_X'], xmax=RANGES['MAX_X'])
    input_ax.set_ylim(ymin=RANGES['MIN_Z'], ymax=RANGES['MAX_Z'])
    input_ax.set_zlim(zmin=RANGES['MIN_Y'], zmax=RANGES['MAX_Y'])

    output_ax.set_xlim(xmin=RANGES['MIN_X'], xmax=RANGES['MAX_X'])
    output_ax.set_ylim(ymin=RANGES['MIN_Z'], ymax=RANGES['MAX_Z'])
    output_ax.set_zlim(zmin=RANGES['MIN_Y'], zmax=RANGES['MAX_Y'])

    gt_ax.set_xlim(xmin=RANGES['MIN_X'], xmax=RANGES['MAX_X'])
    gt_ax.set_ylim(ymin=RANGES['MIN_Z'], ymax=RANGES['MAX_Z'])
    gt_ax.set_zlim(zmin=RANGES['MIN_Y'], zmax=RANGES['MAX_Y'])

    input_ax.set_title('Input')
    output_ax.set_title('Output')
    gt_ax.set_title('Ground Truth')

    fig.suptitle('Event '+str(idx).zfill(4))

    if path == '':
        path = '/'.join(cfg.split('/')[:-1]) + '/imgs/'

    plt.savefig(path+'event'+str(idx).zfill(4)+'.png')
    plt.close()


def experimental_img(input_pc, output_pc, idx, path, cfg):

    assert os.getcwd() == '/home/DAVIDSON/bewagner/summer2023/ATTPCPoinTr', f'Current Directory == {os.getcwd()}'

    RANGES = {
            'MIN_X': -270.0,
            'MAX_X': 270.0,
            'MIN_Y': -270.0,
            'MAX_Y': 270.0,
            'MIN_Z': -185.0,
            'MAX_Z': 1155.0
        }

    fig, (input_ax, output_ax) = plt.subplots(1, 2, figsize=(12,6), subplot_kw=dict(projection='3d'))

    input_xs, input_ys, input_zs = rescale_feats(input_pc[:, 0], input_pc[:, 1], input_pc[:, 2])
    output_xs, output_ys, output_zs = rescale_feats(output_pc[:, 0], output_pc[:, 1], output_pc[:, 2])

    input_ax.scatter(input_xs, input_zs, input_ys, s=4)
    output_ax.scatter(output_xs, output_zs, output_ys, s=4)

    # axes = [input_ax, output_ax]

    # for ax in axes:
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Z')
    #     ax.set_zlabel('Y')

    #     ax.set_xlim(xmin=RANGES['MIN_X'], xmax=RANGES['MAX_X'])
    #     ax.set_ylim(ymin=RANGES['MIN_Z'], ymax=RANGES['MAX_Z'])
    #     ax.set_zlim(zmin=RANGES['MIN_Y'], zmax=RANGES['MAX_Y'])

    input_ax.set_xlabel('X')
    input_ax.set_ylabel('Z')
    input_ax.set_zlabel('Y')

    output_ax.set_xlabel('X')
    output_ax.set_ylabel('Z')
    output_ax.set_zlabel('Y')

    input_ax.set_xlim(xmin=RANGES['MIN_X'], xmax=RANGES['MAX_X'])
    input_ax.set_ylim(ymin=RANGES['MIN_Z'], ymax=RANGES['MAX_Z'])
    input_ax.set_zlim(zmin=RANGES['MIN_Y'], zmax=RANGES['MAX_Y'])

    output_ax.set_xlim(xmin=RANGES['MIN_X'], xmax=RANGES['MAX_X'])
    output_ax.set_ylim(ymin=RANGES['MIN_Z'], ymax=RANGES['MAX_Z'])
    output_ax.set_zlim(zmin=RANGES['MIN_Y'], zmax=RANGES['MAX_Y'])

    input_ax.set_title('Input')
    output_ax.set_title('Output')

    fig.suptitle('Event '+str(idx).zfill(4))

    if path == '':
        path = '/'.join(cfg.split('/')[:-1]) + '/imgs/'

    plt.savefig(path+'event'+str(idx).zfill(4)+'.png')
    plt.close()
