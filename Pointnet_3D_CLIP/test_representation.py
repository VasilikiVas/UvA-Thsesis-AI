"""
Author: Benny
Date: Nov 2019
"""
from data_utils.ABODataLoader import ABODataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
import provider
import collections

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=124, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    return parser.parse_args()


def test(model, criterion, loader, num_class=40, vote_num=1):
    point3d = model.eval()

    for j, (points, img_emb, txt_emb) in tqdm(enumerate(loader), total=len(loader)):

        points = points.data.numpy()
        points = provider.random_point_dropout(points)
        points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        points = torch.Tensor(points)
        points = points.transpose(2, 1)
        img_emb = img_emb.squeeze(1)
        txt_emb = txt_emb.squeeze(1)
        if not args.use_cpu:
            points, img_emb, txt_emb = points.cuda(), img_emb.cuda(), txt_emb.cuda()


        features_3d, _ = point3d(points)
        loss = criterion(features_3d, img_emb, txt_emb)
        print(loss)
    return features_3d, loss


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/ep400_bs124_data962_toptxt/pointnet_feat'

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = '/projects/0/vvasileiou4/ABO/'

    test_dataset = ABODataLoader(root=data_path, args=args, split='train', process_data=False)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    point3d = model.get_model(num_class, normal_channel=args.use_normals)
    criterion = model.get_loss(margin=1)
    if not args.use_cpu:
        point3d = point3d.cuda()
        criterion = criterion.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', map_location=torch.device('cpu'))
    #print(checkpoint['model_state_dict'])
    new_dict = collections.OrderedDict((k.replace("module.", ""), v) for k, v in checkpoint['model_state_dict'].items())    
    point3d.load_state_dict(new_dict)

    with torch.no_grad():
        features_3d, loss = test(point3d.eval(), criterion, testDataLoader, vote_num=args.num_votes, num_class=num_class)
        print(features_3d.shape, loss)
        #log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))


if __name__ == '__main__':
    args = parse_args()
    main(args)