import torch
import numpy as np
import torch.nn as nn
import os
import json
from tools import mybuilder as builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader) = builder.dataset_builder(args, config.dataset.train)
    (_, test_dataloader) = builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # from IPython import embed; embed()
    
    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # print model info
    print_log('Trainable_parameters:', logger = logger)
    print_log('=' * 25, logger = logger)
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)
    
    print_log('Untrainable_parameters:', logger = logger)
    print_log('=' * 25, logger = logger)
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer = builder.build_optimizer(base_model, config)
    
    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()


    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)
    scheduler = builder.build_scheduler(base_model, optimizer, config, last_epoch=start_epoch-1)

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['SparseLoss', 'DenseLoss'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        testing_flag = False
        for idx, (feats, labels) in enumerate(train_dataloader):
            testing_flag = True
            data_time.update(time.time() - batch_start_time)

            in_pc = feats.cuda()
            gt_pc = labels.cuda()

            num_iter += 1
           
            out_pcs = base_model(in_pc)
            
            sparse_loss, dense_loss = base_model.module.get_loss(out_pcs, gt_pc, epoch)
         
            _loss = sparse_loss + dense_loss 
            _loss.backward()

            assert np.sum(np.isnan(out_pcs[-1].squeeze().cpu().numpy())) == 0, f'After Backward -- NaNs in dataset at epoch == {epoch} and batch == {idx}'
            assert np.sum(np.isinf(out_pcs[-1].squeeze().cpu().numpy())) == 0, f'After Backward -- Infs in dataset at epoch == {epoch} and batch == {idx}'
            assert np.isnan(sparse_loss.item()) == False, f'After _loss.backward -- NaN found in sparse_loss at epoch {epoch} and batch {idx}'
            assert np.isinf(sparse_loss.item()) == False, f'After _loss.backward -- Inf found in sparse_loss at epoch {epoch} and batch {idx}'
            assert np.isnan(dense_loss.item()) == False, f'After _loss.backward -- NaN found in dense_loss at epoch {epoch} and batch {idx}'
            assert np.isinf(dense_loss.item()) == False, f'After _loss.backward -- Inf found in dense_loss at epoch {epoch} and batch {idx}'

            # forward
            if num_iter == config.step_per_update:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), getattr(config, 'grad_norm_clip', 10), norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                sparse_loss = dist_utils.reduce_tensor(sparse_loss, args)
                dense_loss = dist_utils.reduce_tensor(dense_loss, args)
                losses.update([sparse_loss.item(), dense_loss.item()])
            else:
                losses.update([sparse_loss.item(), dense_loss.item()])

            assert np.sum(np.isnan(losses.val())) == 0, f'NaN found in losses.val() at epoch {epoch}, batch {idx}'
            assert np.sum(np.isinf(losses.val())) == 0, f'Inf found in losses.val() at epoch {epoch}, batch {idx}'

            if args.distributed:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx
            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Sparse', sparse_loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/Dense', dense_loss.item(), n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 100 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)

            if config.scheduler.type == 'GradualWarmup':
                if n_itr < config.scheduler.kwargs_2.total_epoch:
                    scheduler.step()

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
        else:
            scheduler.step()
        epoch_end_time = time.time()

        assert testing_flag, 'Did not go into data_loader loop'
        assert losses._count[0] != 0, f'losses._count == {losses._count}'

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Sparse', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Dense', losses.avg(1), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)

        if epoch % args.val_freq == 0:
            # Validate the current model
            assert np.sum(np.isnan(out_pcs[-1].squeeze().cpu().numpy())) == 0, f'Before Validate -- NaNs in dataset at epoch == {epoch} and batch == {idx}'
            assert np.sum(np.isinf(out_pcs[-1].squeeze().cpu().numpy())) == 0, f'Before Validate -- Infs in dataset at epoch == {epoch} and batch == {idx}'
            metrics = validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger=logger)

            # Save ckeckpoints
            if  metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)
        if epoch % 20 == 0:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch{epoch}', args, logger = logger)
        if (config.max_epoch - epoch) < 2:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)     
    if train_writer is not None and val_writer is not None:
        train_writer.close()
        val_writer.close()

def validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    n_samples = len(test_dataloader) # bs is 1

    with torch.no_grad():
        for idx, (feats, labels) in enumerate(test_dataloader):
            in_pc = feats.cuda()
            gt_pc = labels.cuda()
            
            event_id = str(idx).zfill(4)

            coarse_points, dense_points = base_model(in_pc)

            sparse_loss_l1 =  ChamferDisL1(coarse_points, gt_pc)
            sparse_loss_l2 =  ChamferDisL2(coarse_points, gt_pc)
            dense_loss_l1 =  ChamferDisL1(dense_points, gt_pc)
            dense_loss_l2 =  ChamferDisL2(dense_points, gt_pc)

            if args.distributed:
                sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, args)
                sparse_loss_l2 = dist_utils.reduce_tensor(sparse_loss_l2, args)
                dense_loss_l1 = dist_utils.reduce_tensor(dense_loss_l1, args)
                dense_loss_l2 = dist_utils.reduce_tensor(dense_loss_l2, args)

            test_losses.update([sparse_loss_l1.item(), sparse_loss_l2.item(), dense_loss_l1.item(), dense_loss_l2.item()])

            _metrics = Metrics.get(dense_points, gt_pc)
            if args.distributed:
                _metrics = [dist_utils.reduce_tensor(_metric, args).item() for _metric in _metrics]
            else:
                _metrics = [_metric.item() for _metric in _metrics]


            if val_writer is not None and idx % 200 == 0:

                gt_ptcloud = gt_pc.squeeze().cpu().numpy()
                gt_ptcloud_img = misc.better_img(gt_ptcloud)
                val_writer.add_image('Event%02d/DenseGT' % idx, gt_ptcloud_img, epoch, dataformats='HWC')

                input_pc = in_pc.squeeze().detach().cpu().numpy()
                input_pc = misc.better_img(input_pc)
                val_writer.add_image('Event%02d/Input'% idx , input_pc, epoch, dataformats='HWC')

                dense = dense_points.squeeze().cpu().numpy()
                assert np.sum(np.isnan(dense)) == 0, f'{np.sum(np.isnan(dense))} NaNs found in predicted cloud at epoch {epoch}, val batch {idx}'
                dense_img = misc.better_img(dense)
                val_writer.add_image('Event%02d/Dense' % idx, dense_img, epoch, dataformats='HWC')
                
            print_log('Test[%d/%d] Sample = %s Losses = %s Metrics = %s' %
                        (idx + 1, n_samples, event_id, ['%.4f' % l for l in test_losses.val()], 
                        ['%.4f' % m for m in _metrics]), logger=logger)
            test_metrics.update(_metrics)


        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()
     
    # Print testing results
    print_log('============================ TEST RESULTS ============================',logger=logger)

    msg = ''
    msg += 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/Sparse', test_losses.avg(0), epoch)
        val_writer.add_scalar('Loss/Epoch/Dense', test_losses.avg(2), epoch)
        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)

    return Metrics(config.consider_metric, test_metrics.avg())


crop_ratio = {
    'easy': 1/4,
    'median' :1/2,
    'hard':3/4
}

def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
 
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger = logger)
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger=logger)

def test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger = None):

    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1

    with torch.no_grad():
        for idx, (feats, labels) in enumerate(test_dataloader):

            event_id = str(idx).zfill(4)
            in_pc = feats.cuda()
            gt_pc = labels.cuda()

            coarse_points, dense_points = base_model(in_pc)

            sparse_loss_l1 = ChamferDisL1(coarse_points, gt_pc)
            sparse_loss_l2 = ChamferDisL2(coarse_points, gt_pc)
            dense_loss_l1 = ChamferDisL1(dense_points, gt_pc)
            dense_loss_l2 = ChamferDisL2(dense_points, gt_pc)

            test_losses.update([sparse_loss_l1, sparse_loss_l2, dense_loss_l1, dense_loss_l2])
            _metrics = Metrics.get(dense_points, gt_pc)
            test_metrics.update(_metrics)

            if (idx+1) % 200 == 0:
                print_log('Test[%d/%d] Event = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, event_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
    
        print_log('[TEST] Metrics = %s' % (['%.4f' % m for m in test_metrics.avg()]), logger=logger)

     

    # Print testing results
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    for metric in test_metrics.items:
        msg += metric + '\t'
    print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall \t\t'
    for value in test_metrics.avg():
        msg += '%.5f \t' % value
    print_log(msg, logger=logger)
    return 
