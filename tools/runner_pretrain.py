import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
from torchvision import transforms

from tools import builder
from utils import misc, dist_utils
from utils.logger import *
from utils.config import *
from utils.AverageMeter import AverageMeter
from datasets import data_transforms


train_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        # data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudRotatePerturbation(),
        # data_transforms.PointcloudTranslate(),
        # data_transforms.PointcloudJitter(),
        # data_transforms.PointcloudRandomInputDropout(),
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict

def evaluate_svm(train_features, train_labels, test_features, test_labels, C):    
    # C = 0.0093 C = 0.018
    model_tl = SVC(C = C, kernel ='linear')
    model_tl.fit(train_features, train_labels)
    test_accuracy = model_tl.score(test_features, test_labels)
    return test_accuracy

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    print_log('Start Pre-training ... ', logger=logger)
    # build dataset for pre-training
    config.dataset.train.others.whole = True
    train_sampler, train_dataloader = builder.dataset_builder(args, config.dataset.train)    
    # build dataset for validate

    print_log("Load extra data to validate ...")
    (_, extra_train_dataloader), (_, extra_test_dataloader) = builder.dataset_builder(args, config.dataset.extra_train), \
                                    builder.dataset_builder(args, config.dataset.extra_val)    
    
    # build model
    base_model = builder.model_builder(config.model)
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)   
    
    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)
    
    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metric)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)
    # resume optimizer   
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # DDP
    if args.use_gpu:
        base_model.to(args.local_rank)  
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

    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()  # set model to training mode

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['Loss'])

        num_iter = 0

        n_batches = len(train_dataloader)
        train_data_bar = tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader), smoothing=0.9)

        for idx, data in train_data_bar:            
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train.others.npoints
            dataset_name = config.dataset.train._base_.NAME

            if dataset_name == 'ShapeNet':
                points = data.cuda()
            elif dataset_name == 'ModelNet':
                points = data[0].cuda()
                points = misc.fps(points, npoints)   
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            assert points.size(1) == npoints

            points = train_transforms(points)
            loss = base_model(points)

            try:
                loss.backward()
                # print("Using one GPU")
            except:
                loss = loss.mean()
                loss.backward()
                # print("Using multi GPUs")

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                losses.update([loss.item()*1000])
            else:
                loss = loss.mean()
                losses.update([loss.item()*1000])


            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            train_data_bar.set_description('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']))
            
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
            
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
        
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) losses = %s lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
             optimizer.param_groups[0]['lr']), logger = logger)
                
        # trainval
        if epoch % args.val_freq == 0 and epoch >=0:
            # Validate the current model
            metrics = validate(base_model, extra_train_dataloader, extra_test_dataloader, args, config, logger=logger)

            # Add testing results to TensorBoard
            if val_writer is not None:
                val_writer.add_scalar('Metric/ACC', metrics.acc, epoch)

            # Save ckeckpoints
            if metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger=logger)

            print_log('[Validation] EPOCH: %d Acc = %.4f best_Acc = %.4f' % (epoch, metrics.acc * 100, best_metrics.acc * 100), logger=logger)

        # Save last checkpoint
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)

        if epoch % 25 ==0 and epoch >=250:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args,
                                    logger=logger)
    
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def validate(base_model, train_dataloader, test_dataloader, args, config, logger=None):
    base_model.eval()
    features = {'train': [], 'test': []}
    labels = {'train': [], 'test': []}
    npoints = config.npoints

    with torch.no_grad():
        # Use a for loop to process training and test data
        for phase, dataloader in [('train', train_dataloader), ('test', test_dataloader)]:
            data_bar = tqdm(enumerate(dataloader, 0), total=len(dataloader), smoothing=0.9)
            for idx, (points, label) in data_bar:
                points = points.cuda()
                label = label.cuda()
                points = misc.fps(points, npoints)
                assert points.size(1) == npoints
                feature = base_model(points, eval=True)
                target = label.view(-1)

                features[phase].append(feature.detach())
                labels[phase].append(target.detach())

        # Combine the list of features and labels into a tensor
        for phase in ['train', 'test']:
            features[phase] = torch.cat(features[phase], dim=0)
            labels[phase] = torch.cat(labels[phase], dim=0)

        if args.distributed:
            for phase in ['train', 'test']:
                features[phase] = dist_utils.gather_tensor(features[phase], args)
                labels[phase] = dist_utils.gather_tensor(labels[phase], args)

        # Evaluating accuracy using SVM
        acc = evaluate_svm(features['train'].data.cpu().numpy(), labels['train'].data.cpu().numpy(),
                           features['test'].data.cpu().numpy(), labels['test'].data.cpu().numpy(), C = config.SVM_C)
        
        if args.distributed:
            torch.cuda.synchronize()

    return Acc_Metric(acc)