import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *

import cv2
import numpy as np


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Visualization start ... ', logger = logger)
    config.dataset.test.others.vis = True
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    base_model = builder.model_builder(config.model)
    # base_model.load_model_from_ckpt(args.ckpts)
    builder.load_model(base_model, args.ckpts, logger = logger)

    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP
    if args.distributed:
        raise NotImplementedError()

    test(base_model, test_dataloader, args, config, logger=logger)


# visualization
def test(base_model, test_dataloader, args, config, logger=None):
    base_model.eval()
    target = './vis'
    useful_cate = [
        "02691156", #plane
        "04379243", #table
        "03790512", #motorbike
        "03948459", #pistol
        "03642806", #laptop
        "03467517", #guitar
        "03261776", #earphone
        "03001627", #chair
        "02958343", #car
        "04090263", #rifle
        "03759954", #microphone
    ]
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            if taxonomy_ids[0] not in useful_cate:
                continue
            
            a, b = {
                "02691156": (90, 135),
                "04379243": (30, 30),
                "03642806": (30, -45),
                "03467517": (0, 90),
                "03261776": (0, 75),
                "03001627": (30, -45)
            }.get(taxonomy_ids[0], (0, 0))
            
            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')
            
            # rec_mae, full_mae, rec_m2ae, full_m2ae, gt_mae, gt_m2ae = base_model(points, vis=True)
            (gt, mae_vis, mae_rebuild, mae_full,
             m2ae_vis, m2ae_rebuild, m2ae_full, pts) = base_model(points, vis=True)
            
            data_path = f'./vis/{taxonomy_ids[0]}/{taxonomy_ids[0]}_{idx}'
            os.makedirs(data_path, exist_ok=True)
            points = points.squeeze()
            
            # Save point clouds
            np.savetxt(os.path.join(data_path, 'points.txt'), points.cpu().numpy(), delimiter=',')
            np.savetxt(os.path.join(data_path, 'gt.txt'), gt.cpu().numpy(), delimiter=',')
            np.savetxt(os.path.join(data_path, 'mae_vis.txt'), mae_vis.cpu().numpy().reshape(-1, 3), delimiter=',')
            np.savetxt(os.path.join(data_path, 'mae_rebuild.txt'), mae_rebuild.cpu().numpy().reshape(-1, 3), delimiter=',')
            np.savetxt(os.path.join(data_path, 'mae_full.txt'), mae_full.cpu().numpy().reshape(-1, 3), delimiter=',')
            np.savetxt(os.path.join(data_path, 'm2ae_vis.txt'), m2ae_vis.cpu().numpy().reshape(-1, 3), delimiter=',')
            np.savetxt(os.path.join(data_path, 'm2ae_rebuild.txt'), m2ae_rebuild.cpu().numpy().reshape(-1, 3), delimiter=',')
            np.savetxt(os.path.join(data_path, 'm2ae_full.txt'), m2ae_full.cpu().numpy().reshape(-1, 3), delimiter=',')
            
            
            # Visualize and save images
            def save_image(points, file_name):
                img = misc.get_ptcloud_img(points.cpu().numpy(), a, b)
                cv2.imwrite(os.path.join(data_path, file_name), img[150:650, 150:675, :])
            
            save_image(points, 'points.jpg')
            save_image(gt, 'gt.jpg')
            save_image(mae_vis, 'mae_vis.jpg')
            save_image(mae_rebuild, 'mae_rebuild.jpg')
            save_image(mae_full, 'mae_full.jpg')
            save_image(m2ae_vis, 'm2ae_vis.jpg')
            save_image(m2ae_rebuild, 'm2ae_rebuild.jpg')
            save_image(m2ae_full, 'm2ae_full.jpg')
            
            if idx > 520:
                break
        return

